"""Soil-water balance simulator and irrigation policy comparison.

IMPORTANT: this is a pedagogical water-balance model driven by *calculated*
FAO-56 crop demand and *measured/simulated* weather.  It is NOT a calibrated
field experiment and must not be presented as real agronomic validation.

Model (single layer, daily step, FAO-56 style):

    Dr(t)        = TAW - S(t)                root-zone depletion, mm
    ETc(t)       = Kc * ET0(t)               crop water demand, mm
    Ks(t)        = 1                          if Dr(t-1) <= RAW
                 = (TAW - Dr(t-1))/(TAW - RAW) if Dr(t-1) >  RAW   (FAO-56 eq. 22)
    ETa(t)       = Ks(t) * ETc(t)            actual evapotranspiration, mm
    S(t)         = clip(S(t-1) + Peff(t) + I(t) - ETa(t), 0, TAW)

where TAW = total available water of the root zone (mm), RAW = MAD * TAW is
the readily available amount, and irrigation I(t) is decided by a policy.

Penalties (both expressed in "cost units"; weights configurable):
    water cost = water_penalty  * sum(I)          unnecessary pumping is bad
    stress cost = stress_penalty * sum(ETc - ETa) crop thirst is worse
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .water_demand import effective_precipitation_usda_scs


@dataclass
class SoilConfig:
    """Root-zone soil water parameters (mm of stored water)."""

    total_available_water_mm: float = 150.0   # TAW: refill point -> wilting point
    mad: float = 0.50                          # management allowed depletion
    initial_storage_mm: float | None = None    # None -> start full
    application_efficiency: float = 0.70

    def validate(self) -> None:
        if self.total_available_water_mm <= 0:
            raise ValueError("total_available_water_mm must be positive.")
        if not 0.05 <= self.mad <= 0.95:
            raise ValueError("mad must be within [0.05, 0.95].")
        if not 0.0 < self.application_efficiency <= 1.0:
            raise ValueError("application_efficiency must be within (0, 1].")
        if self.initial_storage_mm is not None and not (
            0 <= self.initial_storage_mm <= self.total_available_water_mm
        ):
            raise ValueError("initial_storage_mm must be within [0, TAW].")


@dataclass
class CostWeights:
    water_penalty: float = 0.05    # cost per mm of irrigation applied
    stress_penalty: float = 1.0    # cost per mm of crop water deficit


@dataclass
class SimResult:
    policy: str
    frame: pd.DataFrame           # daily state trace
    total_irrigation_mm: float
    total_stress_mm: float        # sum(ETc - ETa): crop thirst realised
    stress_days: int
    water_cost: float
    stress_cost: float

    @property
    def total_cost(self) -> float:
        return self.water_cost + self.stress_cost

    def summary(self) -> dict[str, float | str]:
        return {
            "policy": self.policy,
            "total_irrigation_mm": round(self.total_irrigation_mm, 2),
            "total_stress_mm": round(self.total_stress_mm, 2),
            "stress_days": self.stress_days,
            "water_cost": round(self.water_cost, 2),
            "stress_cost": round(self.stress_cost, 2),
            "total_cost": round(self.total_cost, 2),
        }


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class IrrigationPolicy:
    """Interface: given the day's state, return irrigation to apply (mm)."""

    name = "base"

    def decide(
        self,
        day: pd.Series,
        storage_mm: float,
        taw_mm: float,
        raw_mm: float,
    ) -> float:  # pragma: no cover - interface only
        raise NotImplementedError


class NoIrrigationPolicy(IrrigationPolicy):
    """Baseline A: never irrigate (rainfed)."""

    name = "No irrigation (rainfed baseline)"

    def decide(self, day, storage_mm, taw_mm, raw_mm) -> float:
        return 0.0


class ThresholdPolicy(IrrigationPolicy):
    """Baseline B: irrigate when depletion reaches a fixed fraction of TAW,
    refilling the root zone to field capacity (ignores weather forecasts)."""

    def __init__(self, trigger_fraction: float = 0.5, application_efficiency: float = 0.70):
        if not 0.05 <= trigger_fraction <= 1.0:
            raise ValueError("trigger_fraction must be within [0.05, 1.0].")
        if not 0.0 < application_efficiency <= 1.0:
            raise ValueError("application_efficiency must be within (0, 1].")
        self.trigger_fraction = trigger_fraction
        self.application_efficiency = application_efficiency
        self.name = f"Threshold policy (trigger at {int(trigger_fraction * 100)}% depletion)"

    def decide(self, day, storage_mm, taw_mm, raw_mm) -> float:
        depletion = taw_mm - storage_mm
        if depletion >= self.trigger_fraction * taw_mm:
            return depletion / self.application_efficiency
        return 0.0


class PredictedDemandPolicy(IrrigationPolicy):
    """Proposed policy: irrigate only when the modelled net irrigation
    requirement exceeds a threshold, and never more than the day's need
    plus a small safety margin (capped by available soil capacity)."""

    def __init__(self, demand_threshold_mm: float = 3.0, safety_margin_mm: float = 2.0, application_efficiency: float = 0.70):
        if demand_threshold_mm < 0 or safety_margin_mm < 0:
            raise ValueError("Thresholds must be non-negative.")
        self.demand_threshold_mm = demand_threshold_mm
        self.safety_margin_mm = safety_margin_mm
        self.application_efficiency = application_efficiency
        self.name = (
            f"AquaCrop demand-aware policy "
            f"(trigger > {demand_threshold_mm} mm net need)"
        )

    def decide(self, day, storage_mm, taw_mm, raw_mm) -> float:
        net_need = float(day.get("net_irrig_mm", 0.0))
        if net_need <= self.demand_threshold_mm:
            return 0.0
        # Never irrigate into a nearly-full profile (rain has covered the need).
        free_capacity = taw_mm - storage_mm
        if free_capacity <= 1.0:
            return 0.0
        gross = (net_need + self.safety_margin_mm) / self.application_efficiency
        # Apply at most what the root zone can store.
        return min(gross, free_capacity)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    weather: pd.DataFrame,
    policy: IrrigationPolicy,
    soil: SoilConfig | None = None,
    costs: CostWeights | None = None,
) -> SimResult:
    """Run the daily soil-water balance under one irrigation policy.

    ``weather`` must contain net_irrig_mm and etc_mm (from
    :func:`src.water_demand.crop_water_demand`) and either peff_mm or
    precip_mm.
    """
    soil = soil or SoilConfig()
    costs = costs or CostWeights()
    soil.validate()

    df = weather.sort_index().copy()
    if "etc_mm" not in df.columns:
        raise ValueError("weather frame must include etc_mm (run crop_water_demand first).")
    if "peff_mm" not in df.columns:
        if "precip_mm" in df.columns:
            df["peff_mm"] = effective_precipitation_usda_scs(df["precip_mm"])
        else:
            raise ValueError("weather frame needs peff_mm or precip_mm.")

    taw = soil.total_available_water_mm
    raw = soil.mad * taw
    storage = taw if soil.initial_storage_mm is None else float(soil.initial_storage_mm)

    # Inject efficiency so threshold policy can convert net to gross.
    if isinstance(policy, ThresholdPolicy):
        policy.application_efficiency = soil.application_efficiency

    rows = []
    total_irrig = total_stress = 0.0
    stress_days = 0

    for dt, day in df.iterrows():
        etc = float(day["etc_mm"])
        peff = float(day["peff_mm"])
        depletion_before = taw - storage

        # Stress coefficient uses yesterday's depletion (FAO-56 delay of one day).
        if raw > 0 and depletion_before > raw:
            ks = max(0.0, (taw - depletion_before) / (taw - raw))
        else:
            ks = 1.0
        eta = ks * etc

        irrig_gross = float(np.clip(policy.decide(day, storage, taw, raw), 0.0, taw))
        irrig_net = irrig_gross * soil.application_efficiency

        storage = float(np.clip(storage + peff + irrig_net - eta, 0.0, taw))
        stress = etc - eta

        total_irrig += irrig_gross
        total_stress += stress
        if stress > 0.1:
            stress_days += 1

        rows.append(
            {
                "date": dt,
                "storage_mm": round(storage, 2),
                "depletion_mm": round(taw - storage, 2),
                "etc_mm": etc,
                "eta_mm": round(eta, 2),
                "stress_mm": round(stress, 2),
                "rain_mm": float(day.get("precip_mm", 0.0)),
                "peff_mm": peff,
                "irrigation_mm": round(irrig_gross, 2),
                "ks": round(ks, 3),
            }
        )

    frame = pd.DataFrame(rows).set_index("date")
    return SimResult(
        policy=policy.name,
        frame=frame,
        total_irrigation_mm=total_irrig,
        total_stress_mm=total_stress,
        stress_days=stress_days,
        water_cost=costs.water_penalty * total_irrig,
        stress_cost=costs.stress_penalty * total_stress,
    )


def default_policies(demand_threshold_mm: float = 3.0) -> list[IrrigationPolicy]:
    """The three policies compared in the dashboard."""
    return [
        NoIrrigationPolicy(),
        ThresholdPolicy(trigger_fraction=0.5),
        PredictedDemandPolicy(demand_threshold_mm=demand_threshold_mm),
    ]


def compare_policies(
    weather: pd.DataFrame,
    policies: list[IrrigationPolicy] | None = None,
    soil: SoilConfig | None = None,
    costs: CostWeights | None = None,
) -> pd.DataFrame:
    """Simulate every policy on the same weather and return a summary table."""
    policies = policies or default_policies()
    results = [simulate(weather, p, soil=soil, costs=costs) for p in policies]
    table = pd.DataFrame([r.summary() for r in results]).set_index("policy")
    return table.sort_values("total_cost")
