"""AquaCrop — Smart Irrigation Planner (Streamlit dashboard).

Run with:  streamlit run app.py

Pages: Dashboard | Weather Data | Prediction | Clustering |
       Semi-Supervised Experiment | Irrigation Simulator | Model Evaluation
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import CACHED_SAMPLE_PATH, Location, PUNE, generate_synthetic_sample
from src.evaluation import classification_metrics, confusion_matrix_frame, flag_suspicious_predictions
from src.irrigation_simulator import (
    CostWeights,
    PredictedDemandPolicy,
    SoilConfig,
    compare_policies,
    default_policies,
    simulate,
)
from src.models import describe_cluster, train_logistic_regression
from src.pipeline import PreparedDataset, missing_report, prepare_dataset, weather_summary
from src.preprocessing import build_feature_frame, chronological_split, set_global_seed
from src.semi_supervised import self_training_classification, self_training_regression
from src.water_demand import crop_water_demand

st.set_page_config(page_title="AquaCrop — Smart Irrigation Planner", page_icon="💧", layout="wide")

LOCATION_PRESETS = {
    "Pune, Maharashtra, India": PUNE,
    "Nashik, Maharashtra, India": Location("Nashik, Maharashtra, India", 19.9975, 73.7898, 662.0),
    "Vidisha, Madhya Pradesh, India": Location("Vidisha, Madhya Pradesh, India", 23.5263, 77.8062, 487.0),
    "Custom coordinates...": None,
}


# ---------------------------------------------------------------------------
# Cached data layer
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading weather data and training models...")
def cached_prepare(
    lat: float,
    lon: float,
    elev: float,
    start: str,
    end: str,
    offline: bool,
    kc: float,
    efficiency: float,
    irr_threshold: float,
    n_clusters: int,
    poly_degree: int,
    knn_k: int,
    seed: int,
) -> PreparedDataset:
    loc = Location("Custom", latitude=lat, longitude=lon, elevation_m=elev)
    return prepare_dataset(
        location=loc,
        start=start,
        end=end,
        offline=offline,
        kc=kc,
        application_efficiency=efficiency,
        irrigation_threshold_mm=irr_threshold,
        n_clusters=n_clusters,
        poly_degree=poly_degree,
        knn_k=knn_k,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

st.sidebar.title("💧 AquaCrop")
st.sidebar.caption("Smart Irrigation Planner — ML college project")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Weather Data",
        "Prediction",
        "Clustering",
        "Semi-Supervised Experiment",
        "Irrigation Simulator",
        "Model Evaluation",
    ],
)

st.sidebar.header("Data & model settings")

preset_name = st.sidebar.selectbox("Location preset", list(LOCATION_PRESETS.keys()))
preset = LOCATION_PRESETS[preset_name]
if preset is None:
    col1, col2 = st.sidebar.columns(2)
    lat = col1.number_input("Latitude", -90.0, 90.0, 18.5204, format="%.4f")
    lon = col2.number_input("Longitude", -180.0, 180.0, 73.8567, format="%.4f")
    elev = st.sidebar.number_input("Elevation (m)", 0.0, 6000.0, 560.0)
    location = Location("Custom", latitude=float(lat), longitude=float(lon), elevation_m=float(elev))
else:
    lat, lon, elev = preset.latitude, preset.longitude, preset.elevation_m or 560.0
    location = preset

offline = st.sidebar.checkbox(
    "Offline mode (use bundled sample dataset)",
    value=CACHED_SAMPLE_PATH.exists(),
    help="The bundled sample is real NASA POWER data for Pune (2022-2024). "
         "Offline mode never contacts the API; untick it to fetch live data.",
)
default_start, default_end = date(2022, 1, 1), date(2024, 12, 31)
col1, col2 = st.sidebar.columns(2)
start_d = col1.date_input("Start date", value=default_start, min_value=date(2015, 1, 1), max_value=date(2030, 12, 31))
end_d = col2.date_input("End date", value=default_end, min_value=date(2015, 1, 1), max_value=date(2030, 12, 31))

st.sidebar.subheader("Agronomy")
kc = st.sidebar.slider("Crop coefficient (Kc)", 0.4, 1.4, 1.0, 0.05,
                       help="1.0 = reference grass (ETc = ET0). E.g. maize ~1.15, wheat ~0.7 mid-season.")
efficiency = st.sidebar.slider("Application efficiency", 0.4, 0.95, 0.70, 0.05)
irr_threshold = st.sidebar.slider("Irrigation trigger (mm net need)", 1.0, 8.0, 3.0, 0.5)

st.sidebar.subheader("Model settings")
n_clusters = st.sidebar.slider("K-Means clusters", 2, 6, 4)
poly_degree = st.sidebar.slider("Polynomial degree", 2, 3, 2)
knn_k = st.sidebar.slider("KNN neighbours", 3, 31, 15, 2)
seed = st.sidebar.number_input("Random seed", 0, 9999, 42)

if end_d <= start_d:
    st.sidebar.error("End date must be after start date.")
    st.stop()

try:
    prepared = cached_prepare(
        lat, lon, elev, str(start_d), str(end_d), offline, kc, efficiency,
        irr_threshold, n_clusters, poly_degree, knn_k, seed,
    )
except Exception as exc:  # noqa: BLE001 - show any data error in the UI
    st.error(f"Data preparation failed: {exc}")
    st.stop()

enriched = prepared.enriched
raw = prepared.raw


def model_predict_day(models: dict, row: pd.Series) -> dict[str, float]:
    """Run each trained regressor on a single weather row."""
    X = pd.DataFrame([row[["tmax_c", "tmin_c", "rh_pct", "precip_mm", "solar_mj", "wind_ms"]].astype(float)])
    return {name: float(max(0.0, m.estimator.predict(X)[0])) for name, m in models.items()}


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

if page == "Dashboard":
    st.title("Dashboard")
    st.markdown(
        """
**AquaCrop** estimates daily crop water demand from NASA POWER weather data using the
FAO-56 Penman–Monteith method, learns predictive models (Multiple Linear, Polynomial and
KNN regression; Logistic classification; K-Means clustering), runs a semi-supervised
learning experiment, and compares irrigation policies in a soil-water balance simulator.

**Data honesty:** weather values are *measured* (NASA POWER). ET₀/ETc, the
"irrigation required" label and every simulator state are *calculated or simulated* —
never observed field measurements.
"""
    )
    st.info(f"Data source: {prepared.source_note}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days loaded", f"{len(raw):,}")
    c2.metric("Date range", f"{raw.index.min().date()} → {raw.index.max().date()}")
    c3.metric("Mean ET₀", f"{enriched['et0_mm'].mean():.2f} mm/day")
    c4.metric("Mean daily rain", f"{raw['precip_mm'].mean():.1f} mm")

    st.subheader("Weather summary (measured + calculated)")
    st.dataframe(weather_summary(enriched), width="stretch")

    st.subheader("Monthly overview")
    monthly = enriched.resample("ME").agg(
        et0_mm=("et0_mm", "mean"),
        rain_mm=("precip_mm", "sum"),
        tmax_c=("tmax_c", "mean"),
    )
    fig = go.Figure()
    fig.add_bar(x=monthly.index, y=monthly["rain_mm"], name="Monthly rainfall (mm)", marker_color="#4c9be8")
    fig.add_scatter(x=monthly.index, y=monthly["et0_mm"], name="Mean ET₀ (mm/day)",
                    yaxis="y2", line=dict(color="#e8744c", width=3))
    fig.update_layout(
        height=380,
        yaxis=dict(title="Rainfall (mm)"),
        yaxis2=dict(title="ET₀ (mm/day)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Irrigation recommendation for a selected day")
    data_days = enriched.index
    chosen = st.date_input(
        "Select a day",
        value=min(data_days[-1].date(), date(2024, 5, 15)),
        min_value=data_days.min().date(),
        max_value=data_days.max().date(),
        key="dash_day",
    )
    chosen_ts = pd.Timestamp(chosen)
    if chosen_ts not in enriched.index:
        st.warning("Selected date is not in the loaded dataset (it may be a gap or outside the range).")
    else:
        day = enriched.loc[chosen_ts]
        preds = model_predict_day(prepared.regressors, day)
        best_name = min(prepared.regressors, key=lambda n: prepared.regressors[n].metrics["mae"])
        best_pred = preds[best_name]
        net_need = float(day["net_irrig_mm"])
        if net_need <= irr_threshold:
            rec = f"**No irrigation needed.** Net crop-water deficit after effective rainfall is {net_need:.1f} mm (trigger: {irr_threshold} mm)."
        else:
            gross = net_need / efficiency
            rec = (
                f"**Irrigate ~{gross:.0f} mm gross** (net need {net_need:.1f} mm ÷ efficiency {efficiency:.0%}). "
                f"Rainfall today: {day['precip_mm']:.1f} mm; ETc: {day['etc_mm']:.1f} mm."
            )
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(rec)
            st.caption(
                f"FAO-56 calculated net need: {net_need:.2f} mm | "
                f"Best model ({best_name}) predicts ET₀ = {best_pred:.2f} mm/day "
                f"(all models: " + ", ".join(f"{v:.2f}" for v in preds.values()) + ")"
            )
        with col2:
            weather_card = pd.DataFrame(
                {
                    "Measured weather": [
                        f"{day['tmax_c']:.1f} / {day['tmin_c']:.1f} °C",
                        f"{day['rh_pct']:.0f} %",
                        f"{day['precip_mm']:.1f} mm",
                        f"{day['solar_mj']:.1f} MJ/m²",
                        f"{day['wind_ms']:.1f} m/s",
                    ]
                },
                index=["Temp max/min", "Humidity", "Rain", "Solar", "Wind"],
            )
            st.table(weather_card)

# ---------------------------------------------------------------------------
# Page: Weather Data
# ---------------------------------------------------------------------------

elif page == "Weather Data":
    st.title("Weather Data")
    st.markdown(
        "Daily weather is downloaded from the [NASA POWER API](https://power.larc.nasa.gov/) "
        "for the configured location and date range. If the API is unreachable (or offline "
        "mode is on), the bundled sample — real POWER data for Pune 2022–2024 — is used."
    )
    st.info(f"Current source: {prepared.source_note}")

    tab1, tab2, tab3 = st.tabs(["Preview", "Missing data", "Download"])
    with tab1:
        st.dataframe(raw.head(10).style.format("{:.2f}"), width="stretch")
        st.dataframe(raw.tail(5).style.format("{:.2f}"), width="stretch")
    with tab2:
        rep = missing_report(prepared)
        st.dataframe(rep, width="stretch", hide_index=True)
        n_missing = int(raw.isna().sum().sum())
        if n_missing == 0:
            st.success(f"No missing values in {len(raw)} rows. No imputation was required.")
        else:
            st.warning(f"{n_missing} missing cell(s). Rows with missing targets are dropped before modelling.")
        st.caption(
            "NASA POWER marks unavailable readings with -999; these are converted to NaN "
            "and reported here."
        )
    with tab3:
        csv = raw.copy()
        csv.index.name = "date"
        st.download_button(
            "Download weather CSV",
            data=csv.to_csv().encode("utf-8"),
            file_name="aquacrop_weather.csv",
            mime="text/csv",
        )
        if st.button("Regenerate synthetic fallback sample (offline backup)"):
            syn = generate_synthetic_sample()
            st.dataframe(syn.head(), width="stretch")
            st.caption("Seeded synthetic data — plausible but NOT measurements.")

# ---------------------------------------------------------------------------
# Page: Prediction
# ---------------------------------------------------------------------------

elif page == "Prediction":
    st.title("Prediction — daily reference evapotranspiration (ET₀)")
    st.markdown(
        "Three regressors are trained on chronological splits of the FAO-56 ET₀ target: "
        "**Multiple Linear Regression** (baseline), **Polynomial Regression** and **KNN Regression**. "
        "Pick a past day or type weather values to see each model's prediction."
    )

    split = prepared.regression_split
    st.subheader("Test-set comparison")
    rows = []
    for name, res in prepared.regressors.items():
        rows.append({"Model": name, **{k: round(v, 3) for k, v in res.metrics.items()},
                     "Note": res.notes})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

    mode = st.radio("Input mode", ["Select a date from the test set", "Enter weather manually"], horizontal=True)
    if mode.startswith("Select"):
        day_options = split.X_test.index
        picked = st.selectbox("Test-set day", day_options, index=0, format_func=lambda d: str(d.date()))
        X_one = split.X_test.loc[[picked]]
        actual_et0 = float(split.y_test.loc[picked])
    else:
        c = st.columns(6)
        vals = {
            "tmax_c": c[0].number_input("Tmax (°C)", 0.0, 55.0, 32.0),
            "tmin_c": c[1].number_input("Tmin (°C)", -10.0, 40.0, 20.0),
            "rh_pct": c[2].number_input("Humidity (%)", 0.0, 100.0, 60.0),
            "precip_mm": c[3].number_input("Rain (mm)", 0.0, 300.0, 0.0),
            "solar_mj": c[4].number_input("Solar (MJ/m²)", 0.0, 35.0, 18.0),
            "wind_ms": c[5].number_input("Wind (m/s)", 0.0, 15.0, 2.0),
        }
        if vals["tmin_c"] >= vals["tmax_c"]:
            st.error("Tmin must be below Tmax.")
            st.stop()
        X_one = pd.DataFrame([vals], columns=split.X_train.columns)
        actual_et0 = None

    st.subheader("Predictions")
    preds = {name: float(max(0.0, m.estimator.predict(X_one)[0])) for name, m in prepared.regressors.items()}
    cols = st.columns(len(preds) + (1 if actual_et0 is not None else 0))
    for i, (name, val) in enumerate(preds.items()):
        cols[i].metric(name.replace(" (deg=", " (d="), f"{val:.2f} mm/day")
    if actual_et0 is not None:
        cols[-1].metric("FAO-56 calculated ET₀", f"{actual_et0:.2f} mm/day")

    fig = go.Figure([go.Bar(x=list(preds.keys()), y=list(preds.values()),
                            marker_color=["#4c9be8", "#e8744c", "#5cb88a"])])
    if actual_et0 is not None:
        fig.add_hline(y=actual_et0, line_dash="dash", line_color="black",
                      annotation_text="FAO-56 ET₀", annotation_position="top right")
    fig.update_layout(height=350, yaxis_title="ET₀ (mm/day)", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Actual vs predicted on the held-out test period")
    best_name = min(prepared.regressors, key=lambda n: prepared.regressors[n].metrics["mae"])
    best = prepared.regressors[best_name]
    scatter = px.scatter(
        x=split.y_test.to_numpy(), y=np.asarray(best.y_pred, dtype=float),
        labels={"x": "FAO-56 ET₀ (mm/day)", "y": f"{best_name} prediction (mm/day)"},
        opacity=0.6,
    )
    lims = [0, max(split.y_test.max(), float(np.max(best.y_pred))) * 1.05]
    scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="perfect fit",
                                 line=dict(dash="dash", color="black")))
    scatter.update_layout(height=420, margin=dict(t=20))
    st.plotly_chart(scatter, width="stretch")
    warn = flag_suspicious_predictions(best.y_pred)
    if warn:
        st.warning(warn)

# ---------------------------------------------------------------------------
# Page: Clustering
# ---------------------------------------------------------------------------

elif page == "Clustering":
    st.title("Clustering — weather patterns with K-Means")
    km = prepared.clusters
    st.markdown(
        f"K-Means groups the {len(km.labels):,} days into **{n_clusters}** clusters of similar "
        f"weather. Silhouette score: **{km.silhouette:.3f}** (closer to 1 = better separated). "
        "Features are standardised first so no single unit dominates."
    )

    plot_df = raw.loc[km.labels.index].copy()
    plot_df["cluster"] = km.labels.map(lambda i: f"Cluster {i}")
    fig = px.scatter(
        plot_df.reset_index(), x="tmax_c", y="rh_pct", color="cluster",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=["precip_mm", "solar_mj", "wind_ms"],
        labels={"tmax_c": "Max temperature (°C)", "rh_pct": "Relative humidity (%)"},
        opacity=0.7,
    )
    fig.update_layout(height=480, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Cluster characteristics (raw-unit centroids)")
    prof = km.profile.copy()
    prof["pattern"] = [describe_cluster(km, i) for i in range(len(km.profile))]
    st.dataframe(prof.round(2), width="stretch", hide_index=True)

    st.subheader("Plain-language interpretation")
    for i in range(n_clusters):
        share = float(km.profile.loc[i, "share_pct"])
        st.markdown(f"- **Cluster {i}** ({share}% of days): {describe_cluster(km, i)}")
    st.caption(
        "Typical Pune interpretation: the hot/dry cluster dominates the pre-monsoon months "
        "(highest irrigation demand), the wet clusters cover the monsoon, and the cool/moderate "
        "cluster covers winter — when ET₀ is low but non-zero."
    )

# ---------------------------------------------------------------------------
# Page: Semi-Supervised Experiment
# ---------------------------------------------------------------------------

elif page == "Semi-Supervised Experiment":
    st.title("Semi-supervised learning experiment (self-training)")
    st.markdown(
        """
The training window is split into a small **labelled** subset and a larger **unlabelled**
subset. A base model learns from the labelled part, predicts the unlabelled part, and only
pseudo-labels that pass a documented confidence test are kept; the model is then retrained
on everything accepted. The test set is never used for pseudo-labelling.

- **Classification** confidence = the model's own max class probability (Logistic Regression).
- **Regression** confidence = standard deviation of the k nearest training targets (KNN);
  low dispersion means the neighbourhood agrees and the pseudo-label is trusted.
"""
    )
    c1, c2, c3 = st.columns(3)
    labelled_fraction = c1.slider("Labelled fraction of training data", 0.05, 0.5, 0.10, 0.05)
    conf_threshold = c2.slider("Classification: min class probability", 0.50, 0.99, 0.90, 0.01)
    max_std = c3.slider("Regression: max neighbour std (mm/day)", 0.1, 2.0, 0.8, 0.1)

    set_global_seed(seed)

    with st.spinner("Running classification self-training..."):
        ssl_cls = self_training_classification(
            prepared.classification_split.X_train, prepared.classification_split.y_train,
            prepared.classification_split.X_test, prepared.classification_split.y_test,
            labelled_fraction=labelled_fraction, confidence_threshold=conf_threshold, seed=seed,
        )
    with st.spinner("Running regression self-training..."):
        ssl_reg = self_training_regression(
            prepared.regression_split.X_train, prepared.regression_split.y_train,
            prepared.regression_split.X_test, prepared.regression_split.y_test,
            labelled_fraction=labelled_fraction, max_std_mm=max_std, seed=seed,
        )

    tab1, tab2 = st.tabs(["Classification (irrigation required?)", "Regression (ET₀)"])
    with tab1:
        m = pd.DataFrame({"Supervised only": ssl_cls.baseline_metrics, "Semi-supervised": ssl_cls.ssl_metrics}).T
        st.dataframe(m.round(3), width="stretch")
        st.caption(
            f"Labelled: {ssl_cls.n_labelled} days | Unlabelled: {ssl_cls.n_unlabelled} | "
            f"Pseudo-labels accepted: {ssl_cls.n_pseudo_accepted} ({ssl_cls.acceptance_pct}%) | "
            f"{ssl_cls.notes}"
        )
        fig = go.Figure([
            go.Bar(name="Supervised only", x=["Accuracy", "Precision", "Recall", "F1"],
                   y=[ssl_cls.baseline_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]),
            go.Bar(name="Semi-supervised", x=["Accuracy", "Precision", "Recall", "F1"],
                   y=[ssl_cls.ssl_metrics[k] for k in ["accuracy", "precision", "recall", "f1"]]),
        ])
        fig.update_layout(barmode="group", height=350, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    with tab2:
        m = pd.DataFrame({"Supervised only": ssl_reg.baseline_metrics, "Semi-supervised": ssl_reg.ssl_metrics}).T
        st.dataframe(m.round(3), width="stretch")
        st.caption(
            f"Labelled: {ssl_reg.n_labelled} days | Unlabelled: {ssl_reg.n_unlabelled} | "
            f"Pseudo-labels accepted: {ssl_reg.n_pseudo_accepted} ({ssl_reg.acceptance_pct}%) | "
            f"{ssl_reg.notes}"
        )
        fig = go.Figure([
            go.Bar(name="Supervised only", x=["MAE", "RMSE"],
                   y=[ssl_reg.baseline_metrics["mae"], ssl_reg.baseline_metrics["rmse"]]),
            go.Bar(name="Semi-supervised", x=["MAE", "RMSE"],
                   y=[ssl_reg.ssl_metrics["mae"], ssl_reg.ssl_metrics["rmse"]]),
        ])
        fig.update_layout(barmode="group", height=350, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Honest interpretation")
    st.markdown(
        f"""
With these settings, semi-supervised training changed classification F1 by
**{ssl_cls.improvement['f1']:+.3f}** and regression MAE by **{ssl_reg.improvement['mae']:+.3f} mm/day**
compared with models trained on the identical labelled subsets.

This dataset has a *strong, nearly deterministic* relationship between weather inputs and
FAO-56 ET₀ (R² ≈ 0.95–0.99 even for the linear baseline), so even a small labelled subset
already captures most of the signal. Pseudo-labels therefore mostly re-encode what the
model already knows, and under the chronological test split (later months, different
weather regime) they can add slight bias — self-training reinforces its own errors on
ambiguous days. Pseudo-labeling tends to help when labels are scarce *and* the model is
far from its capacity limit; that is not the situation here, and we report the outcome as
measured rather than tuning until it looks good.
"""
    )

# ---------------------------------------------------------------------------
# Page: Irrigation Simulator
# ---------------------------------------------------------------------------

elif page == "Irrigation Simulator":
    st.title("Irrigation Simulator (soil-water balance)")
    st.markdown(
        """
A transparent single-layer daily water balance: rain fills the root zone, the crop
consumes water (FAO-56 ETc, reduced by a stress coefficient when the soil dries past the
management-allowed depletion), and irrigation policies refill the profile. Costs:
water applied × `water_penalty` + crop deficit × `stress_penalty`.

**This is a teaching model, not a calibrated field trial.**
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    taw = c1.slider("Root-zone TAW (mm)", 50, 300, 150, 10)
    mad = c2.slider("Management allowed depletion", 0.2, 0.8, 0.5, 0.05)
    init_frac = c3.slider("Initial storage (% of TAW)", 0, 100, 80, 5)
    water_pen = c4.slider("Water cost per mm applied", 0.0, 0.5, 0.05, 0.01)
    stress_pen = st.slider("Stress cost per mm of crop deficit", 0.1, 5.0, 1.0, 0.1)

    sim_days = st.slider("Simulation length (days, ending at dataset end)", 30, min(365, len(enriched)), 120, 10)
    sim_df = enriched.iloc[-sim_days:]
    st.caption(f"Simulating {sim_df.index.min().date()} → {sim_df.index.max().date()}")

    soil = SoilConfig(
        total_available_water_mm=float(taw),
        mad=float(mad),
        initial_storage_mm=float(taw) * init_frac / 100.0,
        application_efficiency=efficiency,
    )
    costs = CostWeights(water_penalty=float(water_pen), stress_penalty=float(stress_pen))
    policies = default_policies(demand_threshold_mm=irr_threshold)

    results = [simulate(sim_df, p, soil=soil, costs=costs) for p in policies]
    table = pd.DataFrame([r.summary() for r in results]).set_index("policy").sort_values("total_cost")
    st.subheader("Policy comparison (lower total cost is better)")
    st.dataframe(table.style.format("{:.2f}"), width="stretch")

    best = table.index[0]
    st.success(f"Best policy under these cost weights: **{best}** "
               f"(total cost {table.loc[best, 'total_cost']:.2f}).")

    sel = st.selectbox("Plot policy", [r.policy for r in results])
    res = next(r for r in results if r.policy == sel)
    trace = res.frame

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trace.index, y=trace["storage_mm"], name="Soil storage (mm)",
                             line=dict(color="#8a5a44", width=2)))
    fig.add_hline(y=taw, line_dash="dot", annotation_text="Field capacity (TAW)")
    fig.add_hline(y=soil.mad * taw, line_dash="dot", line_color="orange",
                  annotation_text="Depletion threshold (RAW)")
    fig.update_layout(height=360, yaxis_title="Soil water storage (mm)", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    fig2 = go.Figure()
    fig2.add_bar(x=trace.index, y=trace["rain_mm"], name="Rainfall (mm)", marker_color="#4c9be8")
    fig2.add_bar(x=trace.index, y=trace["irrigation_mm"], name="Irrigation applied (mm)", marker_color="#5cb88a")
    fig2.add_scatter(x=trace.index, y=trace["stress_mm"], name="Daily water stress (mm)",
                     yaxis="y2", line=dict(color="#e8744c", width=2))
    fig2.update_layout(
        barmode="overlay", height=380,
        yaxis=dict(title="Water input (mm)"),
        yaxis2=dict(title="Stress (mm)", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12), margin=dict(t=30),
    )
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Daily simulation trace"):
        st.dataframe(trace.style.format("{:.2f}"), width="stretch")

# ---------------------------------------------------------------------------
# Page: Model Evaluation
# ---------------------------------------------------------------------------

elif page == "Model Evaluation":
    st.title("Model Evaluation")
    split = prepared.regression_split
    cls_split = prepared.classification_split

    st.subheader("Regression metrics — chronological held-out test set")
    reg_rows = []
    for name, res in prepared.regressors.items():
        reg_rows.append({"Model": name, **{k: round(v, 3) for k, v in res.metrics.items()}})
    reg_table = pd.DataFrame(reg_rows).set_index("Model")
    st.dataframe(reg_table, width="stretch")
    st.caption(
        f"Train: {split.n_train} days ({split.X_train.index.min().date()} → {split.X_train.index.max().date()}) | "
        f"Test: {split.n_test} days ({split.X_test.index.min().date()} → {split.X_test.index.max().date()}). "
        "MAE/RMSE/R² are computed from the models' own test predictions."
    )

    fig = go.Figure()
    fig.add_bar(x=reg_table.index, y=reg_table["mae"], name="MAE (mm/day)", marker_color="#4c9be8")
    fig.add_bar(x=reg_table.index, y=reg_table["rmse"], name="RMSE (mm/day)", marker_color="#e8744c")
    fig.update_layout(barmode="group", height=350, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Classification — Logistic Regression (derived target)")
    set_global_seed(seed)
    logres = train_logistic_regression(cls_split, seed=seed)
    st.dataframe(pd.DataFrame([logres.metrics]).T.rename(columns={0: "value"}).round(3), width="stretch")
    st.dataframe(confusion_matrix_frame(cls_split.y_test, logres.y_pred), width="stretch")
    st.caption(logres.notes)

    coefs = pd.Series(
        logres.estimator.named_steps["model"].coef_[0],
        index=cls_split.X_train.columns,
    ).sort_values()
    fig = go.Figure([go.Bar(x=coefs.values, y=coefs.index, orientation="h",
                            marker_color=np.where(coefs.values >= 0, "#5cb88a", "#e8744c"))])
    fig.update_layout(title="Standardised logistic coefficients (log-odds)",
                      height=350, margin=dict(t=40, l=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Limitations — please read before quoting any number")
    st.markdown(
        """
- **ET₀ is calculated, not measured.** Every "ground truth" target comes from the FAO-56
  equation applied to NASA POWER reanalysis weather; models learn to imitate that
  equation from the same inputs, which is why R² is high. This demonstrates ML workflow,
  not independent physical discovery.
- **The irrigation label is derived**, not a record of what farmers did: "irrigation
  required" means the calculated net need exceeded a threshold.
- **The simulator is a single-layer pedagogical water balance** — no groundwater, no
  capillary rise, no crop growth stage variation (single Kc), no salinity.
- **Pune POWER data are reanalysis estimates** at ~0.5° resolution, not a local weather
  station; extremes may be smoothed.
- **Semi-supervised gains are not guaranteed** — with this strong feature–target link it
  often underperforms slightly; the experiment is reported honestly either way.
- **No field validation** of irrigation recommendations was performed.
"""
    )

# ---------------------------------------------------------------------------
# Shared footer
# ---------------------------------------------------------------------------

st.sidebar.divider()
st.sidebar.caption(
    "Targets are FAO-56 calculations from NASA POWER weather. "
    "Not a substitute for agronomic advice."
)
