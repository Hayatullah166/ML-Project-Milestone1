# Energy Consumption Prediction with MLOps
# Project structure for Milestone 1

data/
├── raw/                    # Original datasets
├── processed/              # Cleaned and engineered features
└── reference/              # Reference datasets for drift detection

models/
├── trained/                # Saved model artifacts
└── scalers/                # Feature scalers

src/
├── data/                   # Data pipeline modules
│   ├── __init__.py
│   ├── loader.py          # Data loading utilities
│   ├── preprocessor.py    # Data cleaning and preprocessing
│   └── feature_engineering.py
├── models/                 # Model development
│   ├── __init__.py
│   ├── baseline.py        # Baseline models
│   ├── trainer.py         # Model training pipeline
│   └── evaluator.py       # Model evaluation
├── api/                    # FastAPI application
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   └── schemas.py        # Pydantic models
├── mlops/                  # MLOps components
│   ├── __init__.py
│   ├── drift_detector.py  # EvidentlyAI integration
│   ├── retraining.py      # Automated retraining
│   └── monitoring.py      # Performance monitoring
└── utils/                  # Utility functions
    ├── __init__.py
    ├── config.py          # Configuration management
    └── helpers.py         # Helper functions

notebooks/                  # Jupyter notebooks for exploration
├── data_exploration.ipynb
├── model_development.ipynb
└── drift_analysis.ipynb

tests/                      # Unit tests
├── __init__.py
├── test_data_pipeline.py
├── test_models.py
└── test_api.py

scripts/                    # Standalone scripts
├── download_data.py
├── train_model.py
└── run_drift_check.py

config/                     # Configuration files
├── config.yaml
└── model_config.yaml

mlruns/                     # MLflow tracking (auto-created)
reports/                    # EvidentlyAI reports
logs/                       # Application logs

README.md
requirements.txt
.env.example
