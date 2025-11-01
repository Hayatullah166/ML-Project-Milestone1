## Energy Consumption Prediction with Automated Retraining (MLOps)

### 1) Problem Statement
Energy demand fluctuates daily and seasonally due to factors like time of day, temperature, and human activity. Utilities, campuses, and buildings need accurate short-term load forecasts to plan generation, reduce costs, and avoid overloads. Traditional models degrade as patterns change (new appliances, occupancy shifts, efficiency measures). This project builds a machine learning system that predicts future energy consumption and automatically retrains when performance drifts, ensuring sustained accuracy over time.

### 2) Objectives
- Build a supervised learning model to forecast short-term energy consumption (e.g., next hour/day) from historical load, time features, and weather.
- Track model performance and data drift; trigger automated retraining when drift or performance degradation is detected.
- Version datasets, models, and experiments for reproducibility.
- Serve predictions via a simple API endpoint for later demo use.
- Provide monitoring dashboards and clear evaluation reports.

### 3) Methodology (High-Level)
1. Data collection: acquire historical load and aligned weather data; clean and merge.
2. Feature engineering: time features (hour, day-of-week, holidays), lagged loads, rolling stats, weather (temperature, humidity), and categorical encodings.
3. Baselines and models: start with linear models (Ridge/Lasso), then tree-based (RandomForest, XGBoost); select the best via cross-validation.
4. Evaluation: hold-out or rolling-origin validation; measure MAE, RMSE, R²; compare to naive baseline (e.g., last-week-same-hour).
5. MLOps automation: track experiments (MLflow/DVC), schedule retraining (Airflow), detect drift (EvidentlyAI), and log artifacts and metrics.
6. Serving and monitoring: expose a FastAPI endpoint; monitor predictions, data drift, and performance over time; retrain automatically when thresholds are breached.

### 4) Why MLOps Improves This Project
- Continuous accuracy: drift detection alerts when relationships change; automated retraining restores performance.
- Reproducibility: versioned datasets, code, and models ensure experiments can be repeated and audited.
- Faster iteration: tracked experiments shorten the feedback loop for model improvements.
- Maintainability: clear pipelines (ingest → train → evaluate → register → serve) make the system easier to operate.
- Observability: dashboards and reports surface issues early (data gaps, degraded metrics).

### 5) Datasets (Public Sources)
Primary options (choose one for core build; others for extension/ablation):
- UCI: Individual household electric power consumption dataset (`https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`)
  - Minute-level measurements of active power; can aggregate to hourly/daily.
- Kaggle: Global Energy Forecasting Competition (GEFCom) or household electricity datasets (search "electricity load forecasting", "household power consumption").
- Open Power System Data (OPSD): time series for electricity load and weather (`https://open-power-system-data.org/`).
- PJM Hourly Energy Consumption Data (`https://github.com/PJMETRICS/pjm_load_data/`) or on Kaggle mirrors.

Selection criteria:
- Sufficient historical length (≥ 1–2 years) for seasonal patterns.
- Includes or can be joined with weather data.
- Hourly granularity or convertible to hourly.
- Clear licensing for academic use.

### 6) Tools and Tech Stack (Local, no Docker required)
- Data & Modeling: Python, pandas, NumPy, scikit-learn, XGBoost.
- API Serving: FastAPI + Uvicorn for REST predictions.
- Experiment Tracking & Versioning: MLflow (runs, metrics, artifacts) and/or DVC (data/model versioning); Git for code.
- Orchestration & Scheduling: Apache Airflow for periodic retraining and batch evaluation.
- Drift Detection & Reports: EvidentlyAI to detect data drift, target drift, and performance degradation; generate HTML reports.
- Monitoring UI: Streamlit app to display metrics, drift reports, and simple controls.
- Optional Later: LLM (OpenAI/Llama) for natural-language insights about consumption trends.

### 7) Implementation Plan (Milestones)

Milestone 1 – Model, API, and Drift Monitoring (to be done later)
- Data pipeline: load raw dataset, clean, resample to hourly, join weather.
- Feature engineering: calendar/time features, lags, rolling windows, weather.
- Baseline and model training: Linear baseline → RandomForest/XGBoost; cross-validate.
- Evaluation: MAE, RMSE, R² vs naive baseline; save best model and scaler.
- FastAPI service: `/predict` endpoint consuming recent history and weather; return next-step forecast.
- EvidentlyAI: generate drift dashboard comparing reference window vs. current window; persist reports.
- MLflow/DVC: log runs, metrics, params, and artifacts (model.pkl, plots, reports).

Milestone 2 – LLM Assistant (to be done later)
- Add Q&A assistant that explains drivers of load changes and recommends actions using natural language; ground with model outputs and drift reports.

Milestone 3 – Final Demo & Dashboard (to be done later)
- Streamlit dashboard showing recent load, predictions, residuals, and drift status.
- Trigger retraining from UI (for demo) and show latest model version and metrics.

### 8) Automated Retraining Strategy
- Detection: schedule batch jobs that compute performance on a rolling window (e.g., last 7 days) and run Evidently drift checks on features and predictions.
- Thresholds: if MAE or RMSE worsens by X% vs. reference, or Evidently flags high drift, mark model as stale.
- Action: Airflow DAG triggers retraining: re-ingest recent data, re-fit pipelines, re-evaluate, and, if better, promote the new model.
- Governance: record runs and artifacts in MLflow; keep lineage with DVC; maintain changelog.

### 9) Evaluation Metrics and Expected Outcomes
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination (R²).
- Baselines: naive last-week-same-hour and/or persistence (last hour).
- Expected outcomes: tree-based models should outperform baselines; with automated retraining, performance remains stable across semesters/seasons.

### 10) Risks and Mitigations
- Data gaps or misalignment: implement resampling, interpolation, and robust joins; validate with checks.
- Concept drift: handled via Evidently monitoring and scheduled retraining.
- Overfitting: use cross-validation, early stopping (XGBoost), and hold-out time windows.
- Reproducibility: enforce MLflow/DVC and Git discipline.

### 11) Simple Slide Outline (4–5 Slides)
1. Problem & Motivation
   - Why forecasting energy matters; challenges with changing patterns.
2. Approach & Tools
   - ML models (SKLearn/XGBoost), FastAPI, MLflow/DVC, Airflow, Evidently, Streamlit.
3. MLOps Automation
   - Drift detection → retraining → promotion; experiment tracking and versioning.
4. Results & Metrics (expected)
   - MAE/RMSE/R²; baseline vs. model; stability via retraining.
5. Demo & Next Steps
   - API + dashboard, LLM assistant; future work.

### 12) References
- UCI Power Consumption Dataset: `https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption`
- Open Power System Data: `https://open-power-system-data.org/`
- PJM Hourly Load: `https://github.com/PJMETRICS/pjm_load_data/`
- EvidentlyAI: `https://www.evidentlyai.com/`
- MLflow: `https://mlflow.org/`
- Apache Airflow: `https://airflow.apache.org/`
- FastAPI: `https://fastapi.tiangolo.com/`
- XGBoost: `https://xgboost.readthedocs.io/`


