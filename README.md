# AquaCrop — Smart Irrigation Planner

An end-to-end Machine Learning and agronomic modeling system that recommends **when to irrigate and how much water to apply**, using NASA POWER daily meteorological data and the FAO-56 Penman–Monteith method for crop water demand.

> **Note on Data & Targets:** Weather variables (temperature, humidity, precipitation, solar radiation, wind speed) are real measured reanalysis data from NASA POWER. Reference evapotranspiration ($\text{ET}_0$), crop evapotranspiration ($\text{ET}_c$), the derived irrigation trigger, and soil-water simulator states are calculated and simulated using standard FAO-56 equations.

---

## 1. Objectives

1. **Predict daily reference evapotranspiration ($\text{ET}_0$)** accurately from measured weather variables.
2. **Cluster weather days** into interpretable agro-climatic patterns (hot/dry, wet, humid, moderate).
3. **Conduct a semi-supervised learning experiment** to evaluate whether self-training improves prediction under label scarcity.
4. **Simulate a single-layer daily soil-water balance** to model root-zone depletion and crop stress.
5. **Evaluate irrigation policies** under explicit tradeoffs between applied water cost and crop-stress penalties.

---

## 2. Key Features

- **FAO-56 Physical Modeling**: Daily Penman–Monteith reference evapotranspiration calculation, USDA-SCS effective rainfall estimation, and net/gross crop water demand.
- **Supervised ML Models**: Multiple Linear, Polynomial, and KNN Regressors for $\text{ET}_0$ prediction, plus Logistic Regression for classification of irrigation need.
- **Unsupervised Weather Pattern Clustering**: K-Means clustering with feature standardization and plain-language cluster profiling.
- **Semi-Supervised Self-Training**: Self-training with model confidence filtering (max class probability for classification; KNN target dispersion for regression).
- **Soil-Water Balance Simulator**: Compares rainfed, fixed-threshold, and demand-aware irrigation policies with customizable soil and economic parameters.
- **Interactive Streamlit Dashboard**: 7 interactive pages with Plotly charts, day-level recommendation cards, and test-set diagnostics.
- **Automated Test Suite**: 58 unit tests covering physics validation, data leakage prevention, water balance conservation, and model reproducibility.

---

## 3. Machine Learning Algorithms Used

| Algorithm | Module | Role |
|---|---|---|
| **Multiple Linear Regression** | `src/models.py` | Linear baseline predictor for $\text{ET}_0$ |
| **Polynomial Regression (degree 2)** | `src/models.py` | Non-linear regression capturing weather interactions |
| **K-Nearest Neighbors (KNN) Regression** | `src/models.py` | Non-parametric $\text{ET}_0$ estimation from historical analog days |
| **K-Means Clustering** | `src/models.py` | Agro-climatic regime and weather pattern segmentation |
| **Logistic Regression** | `src/models.py` | Binary classifier for irrigation requirement |
| **Self-Training (LogReg / KNN)** | `src/semi_supervised.py` | Semi-supervised pseudo-labeling with confidence filtering |

**Evaluation Protocol:** Chronological train/test split (first 80% of days for training, last 20% for testing) to prevent future-to-past data leakage.

---

## 4. Dataset Information

The project uses meteorological data from the **[NASA POWER Daily Point API](https://power.larc.nasa.gov/)**:

| Parameter | Feature Name | Description | Units |
|---|---|---|---|
| `T2M_MAX` | `tmax_c` | Daily maximum temperature at 2 m | °C |
| `T2M_MIN` | `tmin_c` | Daily minimum temperature at 2 m | °C |
| `RH2M` | `rh_pct` | Relative humidity at 2 m | % |
| `PRECTOTCORR` | `precip_mm` | Corrected daily precipitation | mm/day |
| `ALLSKY_SFC_SW_DWN` | `solar_mj` | All-sky surface shortwave downward irradiance | MJ/m²/day |
| `WS2M` | `wind_ms` | Wind speed at 2 m | m/s |

- **Default Location**: Pune, Maharashtra, India ($18.52^\circ\text{N}, 73.86^\circ\text{E}$, elevation $560\text{ m}$).
- **Date Range**: 2022-01-01 to 2024-12-31 (1,096 gap-free days).
- **Offline Fallback**: Includes a bundled pre-downloaded dataset (`data/sample_weather_pune.csv`) ensuring the application runs offline or during API downtime.

---

## 5. Technologies & Libraries Used

- **Language**: Python (3.10+)
- **Core Scientific Libraries**: `numpy`, `pandas`, `scipy`
- **Machine Learning**: `scikit-learn`
- **Visualization**: `plotly`
- **Web Dashboard**: `streamlit`
- **API & Networking**: `requests`
- **Testing**: `pytest`

---

## 6. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 7. How to Run the Project

- **Run the Streamlit Dashboard:**
  ```bash
  streamlit run app.py
  ```
  The app will be available locally at `http://localhost:8501`.

- **Run the Automated Test Suite:**
  ```bash
  pytest
  ```

- **(Optional) Refresh Weather Data from NASA POWER API:**
  ```bash
  python -m src.data_loader 2022-01-01 2024-12-31
  ```

---

## 8. Project Structure

```text
aquacrop/
├── app.py                      # Streamlit interactive web dashboard (7 pages)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── data/
│   └── sample_weather_pune.csv # Bundled NASA POWER weather data (Pune 2022-2024)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # NASA POWER API ingestion, validation & fallback
│   ├── water_demand.py         # FAO-56 Penman-Monteith ET0, ETc & effective rain
│   ├── preprocessing.py        # Feature engineering, scaling & chronological splits
│   ├── models.py               # Supervised regression, classification & K-Means
│   ├── semi_supervised.py      # Self-training with confidence filtering
│   ├── irrigation_simulator.py # Soil-water balance model & policy comparison
│   ├── evaluation.py           # Evaluation metrics (MAE, RMSE, R², F1, etc.)
│   └── pipeline.py             # End-to-end dataset preparation pipeline
└── tests/
    ├── __init__.py
    ├── test_water_demand.py    # Physics and calculation checks
    ├── test_preprocessing.py   # Leakage prevention and split validation
    ├── test_simulator.py       # Mass conservation and policy tests
    └── test_models_and_ssl.py  # Model sanity and self-training protocol tests
```

---

## 9. Sample Results

Evaluated on the held-out chronological test period (seed = 42, Pune dataset):

| Model | MAE (mm/day) | RMSE (mm/day) | $R^2$ |
|---|---|---|---|
| **Multiple Linear Regression** | 0.184 | 0.237 | 0.952 |
| **Polynomial Regression (deg=2)** | 0.034 | 0.046 | 0.998 |
| **KNN Regression (k=15)** | 0.170 | 0.255 | 0.944 |

- **Classification (Irrigation Required)**: Logistic Regression achieves an accuracy of ~0.94 and F1-score of ~0.95 on held-out test days.
- **Irrigation Simulator**: The demand-aware policy eliminates water stress while reducing water consumption relative to simple calendar/fixed replenishment policies.

---

## 10. Author

Sanskriti Gilda

AquaCrop combines machine learning, meteorological data, and agronomic modeling to support smarter irrigation decisions and efficient water management.

## 11. License

This project is licensed under the MIT License. See the LICENSE file for details.
