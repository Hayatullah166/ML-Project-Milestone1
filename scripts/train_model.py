"""
Training script for Energy Consumption Prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import logging
import argparse


sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.simple_trainer import SimpleModelTrainer
from src.models.evaluator import ModelEvaluator
from src.mlops.drift_detector import DriftDetector
from src.utils.config import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_models(use_sample_data: bool = True, n_samples: int = 5000):
    """Train energy consumption prediction models."""
    logger.info("Starting model training...")
    
    try:
       
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        trainer = SimpleModelTrainer()
        evaluator = ModelEvaluator()
        drift_detector = DriftDetector()
        
        
        logger.info("Loading data...")
        if use_sample_data:
            df = data_loader.create_sample_data(n_samples=n_samples)
            logger.info(f"Using sample data with {len(df)} samples")
        else:
            df = data_loader.load_raw_data()
            logger.info(f"Loaded real data with {len(df)} samples")
        
        
        logger.info("Preparing features...")
        df_processed, feature_columns = preprocessor.prepare_features(df)
        logger.info(f"Prepared {len(feature_columns)} features")
        
       
        logger.info("Splitting data...")
        train_df, val_df, test_df = preprocessor.split_data(df_processed)
        logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        
        logger.info("Scaling features...")
        train_scaled, val_scaled, test_scaled = preprocessor.scale_features(
            train_df, val_df, test_df
        )
       
        X_train = train_scaled[feature_columns]
        y_train = train_scaled['target']
        X_val = val_scaled[feature_columns]
        y_val = val_scaled['target']
        X_test = test_scaled[feature_columns]
        y_test = test_scaled['target']
        
        
        logger.info("Training models...")
        best_name, best_model, best_metrics = trainer.train_all_models(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        
        logger.info("Evaluating best model...")
        y_pred = best_model.predict(X_test)
        evaluation_metrics = evaluator.evaluate_time_series(y_test, y_pred, best_name)
        
      
        logger.info("Setting reference data for drift detection...")
        drift_detector.set_reference_data(df_processed, feature_columns)
        
        
        logger.info("Generating evaluation plots...")
        plots_path = Path("reports") / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        
        evaluator.plot_predictions(
            best_name, y_test, y_pred,
            save_path=plots_path / f"{best_name}_predictions.png"
        )
        
      
        logger.info("Generating evaluation report...")
        model_results = {best_name: {'metrics': evaluation_metrics, 'predictions': y_pred}}
        report = evaluator.generate_report(model_results)
        
        report_path = Path("reports") / f"model_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
       
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best Model: {best_name}")
        print(f"Test MAE: {best_metrics['mae']:.4f}")
        print(f"Test RMSE: {best_metrics['rmse']:.4f}")
        print(f"Test R²: {best_metrics['r2']:.4f}")
        print(f"Features Used: {len(feature_columns)}")
        print(f"Training Samples: {len(X_train)}")
        print(f"Validation Samples: {len(X_val)}")
        print(f"Test Samples: {len(X_test)}")
        print("="*50)
        
        return {
            "success": True,
            "best_model": best_name,
            "metrics": best_metrics,
            "feature_count": len(feature_columns),
            "data_shape": df_processed.shape
        }
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Energy Consumption Prediction models")
    parser.add_argument("--use-sample-data", action="store_true", default=True,
                       help="Use sample data instead of real data")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Number of samples for sample data")
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use real UCI dataset (requires download)")
    
    args = parser.parse_args()
    
    
    use_sample_data = args.use_sample_data and not args.use_real_data
    
    logger.info(f"Starting training with {'sample' if use_sample_data else 'real'} data")
    
    
    result = train_models(use_sample_data=use_sample_data, n_samples=args.n_samples)
    
    if result["success"]:
        logger.info("Training completed successfully!")
    else:
        logger.error(f"Training failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
