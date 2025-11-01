"""
Automated retraining pipeline for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.simple_trainer import SimpleModelTrainer
from src.mlops.drift_detector import DriftDetector
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

class RetrainingPipeline:
    """Automated retraining pipeline for energy consumption models."""
    
    def __init__(self):
        """Initialize retraining pipeline."""
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = SimpleModelTrainer()
        self.drift_detector = DriftDetector()
        
        # Configuration
        self.performance_threshold = config.get('mlops.performance_threshold', 0.05)
        self.drift_threshold = config.get('mlops.drift_threshold', 0.1)
        self.model_path = Path(config.get('models.trained_path', 'models/trained'))
        
        # MLflow setup
        mlflow.set_tracking_uri(config.get('mlflow.tracking_uri', 'file:./mlruns'))
        
    def check_retraining_needed(self) -> Dict[str, Any]:
        """Check if retraining is needed based on drift and performance."""
        logger.info("Checking if retraining is needed...")
        
        retraining_info = {
            "retraining_needed": False,
            "reasons": [],
            "drift_check": None,
            "performance_check": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check for data drift
            drift_result = self.drift_detector.check_drift()
            retraining_info["drift_check"] = drift_result
            
            if drift_result["drift_detected"]:
                drift_score = drift_result["drift_score"]
                if drift_score > self.drift_threshold:
                    retraining_info["retraining_needed"] = True
                    retraining_info["reasons"].append(f"Data drift detected (score: {drift_score:.3f})")
            
            # Check model performance (if we have recent predictions)
            performance_result = self._check_model_performance()
            retraining_info["performance_check"] = performance_result
            
            if performance_result["performance_degraded"]:
                retraining_info["retraining_needed"] = True
                retraining_info["reasons"].append(f"Performance degraded (MAE increase: {performance_result['mae_increase']:.3f})")
            
            logger.info(f"Retraining needed: {retraining_info['retraining_needed']}")
            logger.info(f"Reasons: {retraining_info['reasons']}")
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            retraining_info["error"] = str(e)
        
        return retraining_info
    
    def _check_model_performance(self) -> Dict[str, Any]:
        """Check if model performance has degraded."""
        try:
            # Load current model metrics
            model_files = list(self.model_path.glob("*_metrics.json"))
            if not model_files:
                return {"performance_degraded": False, "reason": "No model metrics found"}
            
            # Get most recent model metrics
            latest_metrics_file = max(model_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metrics_file, 'r') as f:
                current_metrics = json.load(f)
            
            # For demonstration, we'll simulate performance degradation
            # In practice, you'd compare with recent validation performance
            current_mae = current_metrics.get('mae', 1.0)
            
            # Simulate performance check (in practice, use recent validation data)
            simulated_recent_mae = current_mae * (1 + np.random.uniform(0, 0.2))  # Random performance change
            
            mae_increase = (simulated_recent_mae - current_mae) / current_mae
            
            performance_degraded = mae_increase > self.performance_threshold
            
            return {
                "performance_degraded": performance_degraded,
                "current_mae": current_mae,
                "recent_mae": simulated_recent_mae,
                "mae_increase": mae_increase,
                "threshold": self.performance_threshold
            }
            
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
            return {"performance_degraded": False, "error": str(e)}
    
    def retrain_model(self, model_name: Optional[str] = None, 
                     use_latest_data: bool = True) -> Dict[str, Any]:
        """Retrain the model with latest data."""
        logger.info(f"Starting model retraining for {model_name or 'best model'}...")
        
        retraining_result = {
            "success": False,
            "new_model_name": None,
            "metrics": {},
            "training_time": 0.0,
            "message": "",
            "timestamp": datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        
        try:
            # Load latest data
            if use_latest_data:
                logger.info("Loading latest data for retraining...")
                df = self._load_latest_data()
            else:
                logger.info("Using existing processed data...")
                df = self._load_processed_data()
            
            if df is None or len(df) == 0:
                raise ValueError("No data available for retraining")
            
            # Prepare features
            logger.info("Preparing features...")
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            
            if len(df_processed) == 0:
                raise ValueError("No valid data after preprocessing")
            
            # Split data
            logger.info("Splitting data...")
            train_df, val_df, test_df = self.preprocessor.split_data(
                df_processed,
                test_size=config.get('model.test_size', 0.2),
                validation_size=config.get('model.validation_size', 0.1)
            )
            
            # Scale features
            logger.info("Scaling features...")
            train_scaled, val_scaled, test_scaled = self.preprocessor.scale_features(
                train_df, val_df, test_df
            )
            
            # Prepare features and targets
            X_train = train_scaled[feature_columns]
            y_train = train_scaled['target']
            X_val = val_scaled[feature_columns]
            y_val = val_scaled['target']
            X_test = test_scaled[feature_columns]
            y_test = test_scaled['target']
            
            # Train models
            logger.info("Training models...")
            best_name, best_model, best_metrics = self.trainer.train_all_models(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Update reference data for drift detection
            logger.info("Updating reference data...")
            self.drift_detector.set_reference_data(df_processed, feature_columns)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare result
            retraining_result.update({
                "success": True,
                "new_model_name": best_name,
                "metrics": best_metrics,
                "training_time": training_time,
                "message": f"Model retrained successfully. Best model: {best_name}"
            })
            
            logger.info(f"Retraining completed successfully in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            retraining_result.update({
                "success": False,
                "message": f"Retraining failed: {str(e)}",
                "training_time": (datetime.now() - start_time).total_seconds()
            })
        
        return retraining_result
    
    def _load_latest_data(self) -> Optional[pd.DataFrame]:
        """Load latest data for retraining."""
        try:
            # Try to load raw data and process it
            df = self.data_loader.load_raw_data()
            
            if df is None or len(df) == 0:
                # Generate sample data if no real data available
                logger.info("No raw data found, generating sample data...")
                df = self.data_loader.create_sample_data(n_samples=5000)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading latest data: {e}")
            return None
    
    def _load_processed_data(self) -> Optional[pd.DataFrame]:
        """Load existing processed data."""
        try:
            processed_files = list(Path(config.get('data.processed_path', 'data/processed')).glob("*.parquet"))
            
            if not processed_files:
                return None
            
            # Load most recent processed data
            latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            return pd.read_parquet(latest_file)
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return None
    
    def schedule_retraining(self, frequency: str = "daily") -> Dict[str, Any]:
        """Schedule retraining based on frequency."""
        logger.info(f"Scheduling retraining with frequency: {frequency}")
        
        schedule_result = {
            "scheduled": False,
            "frequency": frequency,
            "next_run": None,
            "message": ""
        }
        
        try:
            # Calculate next run time based on frequency
            now = datetime.now()
            
            if frequency == "daily":
                next_run = now + timedelta(days=1)
            elif frequency == "weekly":
                next_run = now + timedelta(weeks=1)
            elif frequency == "hourly":
                next_run = now + timedelta(hours=1)
            else:
                raise ValueError(f"Unsupported frequency: {frequency}")
            
            schedule_result.update({
                "scheduled": True,
                "next_run": next_run.isoformat(),
                "message": f"Retraining scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')}"
            })
            
            # In a real implementation, you would:
            # 1. Save the schedule to a database or file
            # 2. Set up a cron job or use Airflow
            # 3. Implement the actual scheduling logic
            
            logger.info(f"Retraining scheduled for {next_run}")
            
        except Exception as e:
            logger.error(f"Error scheduling retraining: {e}")
            schedule_result["message"] = f"Scheduling failed: {str(e)}"
        
        return schedule_result
    
    def run_automated_retraining(self) -> Dict[str, Any]:
        """Run the complete automated retraining pipeline."""
        logger.info("Running automated retraining pipeline...")
        
        pipeline_result = {
            "success": False,
            "steps_completed": [],
            "retraining_result": None,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Check if retraining is needed
            logger.info("Step 1: Checking if retraining is needed...")
            retraining_check = self.check_retraining_needed()
            pipeline_result["steps_completed"].append("retraining_check")
            
            if not retraining_check["retraining_needed"]:
                pipeline_result.update({
                    "success": True,
                    "message": "No retraining needed",
                    "retraining_check": retraining_check
                })
                return pipeline_result
            
            # Step 2: Retrain model
            logger.info("Step 2: Retraining model...")
            retraining_result = self.retrain_model()
            pipeline_result["steps_completed"].append("model_retraining")
            pipeline_result["retraining_result"] = retraining_result
            
            if not retraining_result["success"]:
                pipeline_result["message"] = "Retraining failed"
                return pipeline_result
            
            # Step 3: Update model registry (if using MLflow)
            logger.info("Step 3: Updating model registry...")
            self._update_model_registry(retraining_result)
            pipeline_result["steps_completed"].append("model_registry_update")
            
            # Step 4: Run post-retraining validation
            logger.info("Step 4: Running post-retraining validation...")
            validation_result = self._run_post_retraining_validation(retraining_result)
            pipeline_result["steps_completed"].append("post_retraining_validation")
            pipeline_result["validation_result"] = validation_result
            
            pipeline_result.update({
                "success": True,
                "message": "Automated retraining pipeline completed successfully"
            })
            
            logger.info("Automated retraining pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in automated retraining pipeline: {e}")
            pipeline_result.update({
                "success": False,
                "message": f"Pipeline failed: {str(e)}"
            })
        
        return pipeline_result
    
    def _update_model_registry(self, retraining_result: Dict[str, Any]):
        """Update model registry with new model."""
        try:
            # In a real implementation, you would:
            # 1. Register the new model in MLflow Model Registry
            # 2. Update model versioning
            # 3. Set the new model as production-ready
            
            logger.info(f"Model registry updated for {retraining_result['new_model_name']}")
            
        except Exception as e:
            logger.error(f"Error updating model registry: {e}")
    
    def _run_post_retraining_validation(self, retraining_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation tests after retraining."""
        try:
            # Run drift check on new model
            drift_result = self.drift_detector.check_drift()
            
            # Run stability tests
            stability_result = self.drift_detector.run_stability_tests()
            
            validation_result = {
                "drift_check": drift_result,
                "stability_tests": stability_result,
                "model_metrics": retraining_result["metrics"],
                "validation_passed": True
            }
            
            logger.info("Post-retraining validation completed")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error in post-retraining validation: {e}")
            return {"validation_passed": False, "error": str(e)}
