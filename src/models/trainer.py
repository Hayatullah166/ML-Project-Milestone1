"""
Model training pipeline for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from pathlib import Path
import mlflow
import mlflow.sklearn
from src.models.baseline import NaiveBaseline, SeasonalNaiveBaseline, MovingAverageBaseline, PersistenceBaseline, LinearTrendBaseline
from src.utils.config import config

class ModelTrainer:
    """Model training pipeline for energy consumption prediction."""
    
    def __init__(self):
        """Initialize model trainer."""
        self.models = {}
        self.model_configs = config.get('models', {})
        self.mlflow_experiment = config.get('mlflow.experiment_name', 'energy_consumption_prediction')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.get('mlflow.tracking_uri', 'file:./mlruns'))
        
    def get_baseline_models(self) -> Dict[str, Any]:
        """Get baseline models for comparison."""
        return {
            'naive': NaiveBaseline(),
            'seasonal_naive': SeasonalNaiveBaseline(seasonal_period=24),
            'moving_average': MovingAverageBaseline(window_size=24),
            'persistence': PersistenceBaseline(),
            'linear_trend': LinearTrendBaseline()
        }
    
    def get_ml_models(self) -> Dict[str, Any]:
        """Get machine learning models."""
        models = {}
        
        # Linear models
        if 'linear' in self.model_configs:
            linear_config = self.model_configs['linear']
            models['ridge'] = Ridge(alpha=linear_config.get('alpha', 1.0), random_state=42)
            models['lasso'] = Lasso(alpha=linear_config.get('alpha', 1.0), random_state=42)
        
        # Random Forest
        if 'random_forest' in self.model_configs:
            rf_config = self.model_configs['random_forest']
            models['random_forest'] = RandomForestRegressor(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 10),
                min_samples_split=rf_config.get('min_samples_split', 5),
                random_state=42,
                n_jobs=-1
            )
        
        # XGBoost
        if 'xgboost' in self.model_configs:
            xgb_config = self.model_configs['xgboost']
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                learning_rate=xgb_config.get('learning_rate', 0.1),
                subsample=xgb_config.get('subsample', 0.8),
                random_state=42,
                n_jobs=-1
            )
        
        return models
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        print(f"{model_name} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train and evaluate baseline models."""
        print("Training baseline models...")
        
        baseline_models = self.get_baseline_models()
        baseline_results = {}
        
        for name, model in baseline_models.items():
            print(f"Training {name}...")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, X_val, y_val, f"{name} (val)")
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test, f"{name} (test)")
            
            baseline_results[name] = {
                'model': model,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }
        
        return baseline_results
    
    def train_ml_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train and evaluate machine learning models."""
        print("Training machine learning models...")
        
        ml_models = self.get_ml_models()
        ml_results = {}
        
        for name, model in ml_models.items():
            print(f"Training {name}...")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"{name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log model parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_metrics = self.evaluate_model(model, X_val, y_val, f"{name} (val)")
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test, y_test, f"{name} (test)")
                
                # Log metrics
                mlflow.log_metrics({
                    'val_mae': val_metrics['mae'],
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'test_mae': test_metrics['mae'],
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2']
                })
                
                # Log model
                mlflow.sklearn.log_model(model, f"{name}_model")
                
                ml_results[name] = {
                    'model': model,
                    'val_metrics': val_metrics,
                    'test_metrics': test_metrics
                }
        
        return ml_results
    
    def select_best_model(self, baseline_results: Dict[str, Dict[str, float]], 
                         ml_results: Dict[str, Dict[str, float]]) -> Tuple[str, Any, Dict[str, float]]:
        """Select the best model based on validation performance."""
        print("Selecting best model...")
        
        best_score = float('inf')
        best_model_name = None
        best_model = None
        best_metrics = None
        
        # Check baseline models
        for name, results in baseline_results.items():
            val_mae = results['val_metrics']['mae']
            if val_mae < best_score:
                best_score = val_mae
                best_model_name = name
                best_model = results['model']
                best_metrics = results['test_metrics']
        
        # Check ML models
        for name, results in ml_results.items():
            val_mae = results['val_metrics']['mae']
            if val_mae < best_score:
                best_score = val_mae
                best_model_name = name
                best_model = results['model']
                best_metrics = results['test_metrics']
        
        print(f"Best model: {best_model_name} with validation MAE: {best_score:.4f}")
        print(f"Test performance - MAE: {best_metrics['mae']:.4f}, RMSE: {best_metrics['rmse']:.4f}, R²: {best_metrics['r2']:.4f}")
        
        return best_model_name, best_model, best_metrics
    
    def save_model(self, model: Any, model_name: str, metrics: Dict[str, float]) -> Path:
        """Save the trained model and metrics."""
        model_path = Path(config.get('models.trained_path', 'models/trained'))
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / f"{model_name}_model.joblib"
        joblib.dump(model, model_file)
        
        # Save metrics
        metrics_file = model_path / f"{model_name}_metrics.json"
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Model saved to {model_file}")
        print(f"Metrics saved to {metrics_file}")
        
        return model_file
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[str, Any, Dict[str, float]]:
        """Train all models and return the best one."""
        print("Starting model training pipeline...")
        
        # Train baseline models
        baseline_results = self.train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Train ML models
        ml_results = self.train_ml_models(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Select best model
        best_name, best_model, best_metrics = self.select_best_model(baseline_results, ml_results)
        
        # Save best model
        self.save_model(best_model, best_name, best_metrics)
        
        # Store all results
        self.models = {
            'baseline': baseline_results,
            'ml': ml_results,
            'best': {
                'name': best_name,
                'model': best_model,
                'metrics': best_metrics
            }
        }
        
        return best_name, best_model, best_metrics
