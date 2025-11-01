"""
End-to-end testing script for Energy Consumption Prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
from typing import Dict, Any


sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.simple_trainer import SimpleModelTrainer
from src.models.evaluator import ModelEvaluator
from src.mlops.drift_detector import DriftDetector
from src.mlops.retraining import RetrainingPipeline
from src.utils.config import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemTester:
    """End-to-end system tester for Energy Consumption Prediction."""
    
    def __init__(self):
        """Initialize system tester."""
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.trainer = SimpleModelTrainer()
        self.evaluator = ModelEvaluator()
        self.drift_detector = DriftDetector()
        self.retraining_pipeline = RetrainingPipeline()
        
        self.test_results = {}
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests."""
        logger.info("Starting end-to-end system tests...")
        
        test_results = {
            "overall_success": True,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        
        logger.info("Test 1: Data Pipeline")
        data_test = self.test_data_pipeline()
        test_results["test_details"]["data_pipeline"] = data_test
        if data_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
       
        logger.info("Test 2: Model Training")
        training_test = self.test_model_training()
        test_results["test_details"]["model_training"] = training_test
        if training_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
       
        logger.info("Test 3: Model Evaluation")
        evaluation_test = self.test_model_evaluation()
        test_results["test_details"]["model_evaluation"] = evaluation_test
        if evaluation_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
        
        logger.info("Test 4: Drift Detection")
        drift_test = self.test_drift_detection()
        test_results["test_details"]["drift_detection"] = drift_test
        if drift_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
        
        logger.info("Test 5: Retraining Pipeline")
        retraining_test = self.test_retraining_pipeline()
        test_results["test_details"]["retraining_pipeline"] = retraining_test
        if retraining_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
        # Test 6: End-to-End Workflow
        logger.info("Test 6: End-to-End Workflow")
        workflow_test = self.test_end_to_end_workflow()
        test_results["test_details"]["end_to_end_workflow"] = workflow_test
        if workflow_test["success"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["overall_success"] = False
        
        logger.info(f"System tests completed. Passed: {test_results['tests_passed']}, Failed: {test_results['tests_failed']}")
        
        return test_results
    
    def test_data_pipeline(self) -> Dict[str, Any]:
        """Test data loading and preprocessing pipeline."""
        logger.info("Testing data pipeline...")
        
        test_result = {
            "success": False,
            "steps": [],
            "errors": []
        }
        
        try:
           
            logger.info("Step 1: Loading data...")
            df = self.data_loader.create_sample_data(n_samples=1000)
            test_result["steps"].append("data_loading")
            
            if df is None or len(df) == 0:
                raise ValueError("Failed to load data")
            
            
            logger.info("Step 2: Preprocessing data...")
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            test_result["steps"].append("data_preprocessing")
            
            if len(df_processed) == 0:
                raise ValueError("No data after preprocessing")
            
         
            logger.info("Step 3: Splitting data...")
            train_df, val_df, test_df = self.preprocessor.split_data(df_processed)
            test_result["steps"].append("data_splitting")
            
            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                raise ValueError("Invalid data splits")
            
            
            logger.info("Step 4: Scaling features...")
            train_scaled, val_scaled, test_scaled = self.preprocessor.scale_features(
                train_df, val_df, test_df
            )
            test_result["steps"].append("feature_scaling")
            
            test_result.update({
                "success": True,
                "data_shape": df_processed.shape,
                "feature_count": len(feature_columns),
                "train_samples": len(train_scaled),
                "val_samples": len(val_scaled),
                "test_samples": len(test_scaled)
            })
            
            logger.info("Data pipeline test passed")
            
        except Exception as e:
            logger.error(f"Data pipeline test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def test_model_training(self) -> Dict[str, Any]:
        """Test model training pipeline."""
        logger.info("Testing model training...")
        
        test_result = {
            "success": False,
            "models_trained": 0,
            "best_model": None,
            "errors": []
        }
        
        try:
            
            df = self.data_loader.create_sample_data(n_samples=2000)
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            train_df, val_df, test_df = self.preprocessor.split_data(df_processed)
            train_scaled, val_scaled, test_scaled = self.preprocessor.scale_features(
                train_df, val_df, test_df
            )
          
            X_train = train_scaled[feature_columns]
            y_train = train_scaled['target']
            X_val = val_scaled[feature_columns]
            y_val = val_scaled['target']
            X_test = test_scaled[feature_columns]
            y_test = test_scaled['target']
           
            logger.info("Training models...")
            best_name, best_model, best_metrics = self.trainer.train_all_models(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            test_result.update({
                "success": True,
                "models_trained": len(self.trainer.models.get('baseline', {})) + len(self.trainer.models.get('ml', {})),
                "best_model": best_name,
                "best_metrics": best_metrics
            })
            
            logger.info("Model training test passed")
            
        except Exception as e:
            logger.error(f"Model training test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def test_model_evaluation(self) -> Dict[str, Any]:
        """Test model evaluation pipeline."""
        logger.info("Testing model evaluation...")
        
        test_result = {
            "success": False,
            "evaluation_metrics": {},
            "errors": []
        }
        
        try:
           
            if not hasattr(self.trainer, 'models') or not self.trainer.models:
                raise ValueError("No trained models available")
            
            
            df = self.data_loader.create_sample_data(n_samples=500)
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            _, _, test_df = self.preprocessor.split_data(df_processed)
            _, _, test_scaled = self.preprocessor.scale_features(test_df, test_df, test_df)
            
            X_test = test_scaled[feature_columns]
            y_test = test_scaled['target']
            
            
            model_results = {}
            for model_type, models in self.trainer.models.items():
                if model_type == 'best':
                    continue
                    
                for model_name, model_data in models.items():
                    model = model_data['model']
                    y_pred = model.predict(X_test)
                    
                    metrics = self.evaluator.evaluate_time_series(y_test, y_pred, model_name)
                    model_results[model_name] = {
                        'metrics': metrics,
                        'predictions': y_pred
                    }
            
            
            comparison_df = self.evaluator.compare_models(model_results)
            
            test_result.update({
                "success": True,
                "evaluation_metrics": {name: data['metrics'] for name, data in model_results.items()},
                "model_comparison": comparison_df.to_dict('records')
            })
            
            logger.info("Model evaluation test passed")
            
        except Exception as e:
            logger.error(f"Model evaluation test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def test_drift_detection(self) -> Dict[str, Any]:
        """Test drift detection pipeline."""
        logger.info("Testing drift detection...")
        
        test_result = {
            "success": False,
            "drift_results": {},
            "errors": []
        }
        
        try:
            
            df = self.data_loader.create_sample_data(n_samples=1000)
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            self.drift_detector.set_reference_data(df_processed, feature_columns)
            
            
            drift_result = self.drift_detector.check_drift()
            
            
            stability_result = self.drift_detector.run_stability_tests()
            
            test_result.update({
                "success": True,
                "drift_results": drift_result,
                "stability_results": stability_result
            })
            
            logger.info("Drift detection test passed")
            
        except Exception as e:
            logger.error(f"Drift detection test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def test_retraining_pipeline(self) -> Dict[str, Any]:
        """Test retraining pipeline."""
        logger.info("Testing retraining pipeline...")
        
        test_result = {
            "success": False,
            "retraining_results": {},
            "errors": []
        }
        
        try:
            
            retraining_check = self.retraining_pipeline.check_retraining_needed()
            
            
            retraining_result = self.retraining_pipeline.retrain_model()
            
            test_result.update({
                "success": True,
                "retraining_check": retraining_check,
                "retraining_results": retraining_result
            })
            
            logger.info("Retraining pipeline test passed")
            
        except Exception as e:
            logger.error(f"Retraining pipeline test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow."""
        logger.info("Testing end-to-end workflow...")
        
        test_result = {
            "success": False,
            "workflow_steps": [],
            "errors": []
        }
        
        try:
          
            logger.info("Workflow Step 1: Data Pipeline")
            df = self.data_loader.create_sample_data(n_samples=1500)
            df_processed, feature_columns = self.preprocessor.prepare_features(df)
            train_df, val_df, test_df = self.preprocessor.split_data(df_processed)
            train_scaled, val_scaled, test_scaled = self.preprocessor.scale_features(
                train_df, val_df, test_df
            )
            test_result["workflow_steps"].append("data_pipeline")
            
           
            logger.info("Workflow Step 2: Model Training")
            X_train = train_scaled[feature_columns]
            y_train = train_scaled['target']
            X_val = val_scaled[feature_columns]
            y_val = val_scaled['target']
            X_test = test_scaled[feature_columns]
            y_test = test_scaled['target']
            
            best_name, best_model, best_metrics = self.trainer.train_all_models(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            test_result["workflow_steps"].append("model_training")
            
          
            logger.info("Workflow Step 3: Model Evaluation")
            y_pred = best_model.predict(X_test)
            metrics = self.evaluator.evaluate_time_series(y_test, y_pred, best_name)
            test_result["workflow_steps"].append("model_evaluation")
           
            logger.info("Workflow Step 4: Drift Detection Setup")
            self.drift_detector.set_reference_data(df_processed, feature_columns)
            drift_result = self.drift_detector.check_drift()
            test_result["workflow_steps"].append("drift_detection")
            
            
            logger.info("Workflow Step 5: Automated Retraining Check")
            retraining_check = self.retraining_pipeline.check_retraining_needed()
            test_result["workflow_steps"].append("retraining_check")
            
            test_result.update({
                "success": True,
                "best_model": best_name,
                "best_metrics": best_metrics,
                "drift_detected": drift_result["drift_detected"],
                "retraining_needed": retraining_check["retraining_needed"]
            })
            
            logger.info("End-to-end workflow test passed")
            
        except Exception as e:
            logger.error(f"End-to-end workflow test failed: {e}")
            test_result["errors"].append(str(e))
        
        return test_result
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# Energy Consumption Prediction System - Test Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        
        report.append("## Overall Results\n")
        report.append(f"- **Overall Success**: {'PASS' if test_results['overall_success'] else 'FAIL'}")
        report.append(f"- **Tests Passed**: {test_results['tests_passed']}")
        report.append(f"- **Tests Failed**: {test_results['tests_failed']}")
        report.append(f"- **Success Rate**: {test_results['tests_passed'] / (test_results['tests_passed'] + test_results['tests_failed']) * 100:.1f}%\n")
        
       
        report.append("## Detailed Test Results\n")
        
        for test_name, test_detail in test_results['test_details'].items():
            report.append(f"### {test_name.replace('_', ' ').title()}\n")
            
            if test_detail['success']:
                report.append("**Status**: PASS\n")
            else:
                report.append("**Status**: FAIL\n")
            
            
            if 'errors' in test_detail and test_detail['errors']:
                report.append("**Errors**:")
                for error in test_detail['errors']:
                    report.append(f"- {error}")
                report.append("")
            
            
            if test_detail['success']:
                for key, value in test_detail.items():
                    if key not in ['success', 'errors']:
                        report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run system tests."""
    logger.info("Starting Energy Consumption Prediction System Tests")
    
    
    tester = SystemTester()
    
    
    test_results = tester.run_all_tests()
    
    
    report = tester.generate_test_report(test_results)
    
    report_path = Path("reports") / f"system_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Test report saved to {report_path}")
    
   
    print("\n" + "="*50)
    print("SYSTEM TEST SUMMARY")
    print("="*50)
    print(f"Overall Success: {'PASS' if test_results['overall_success'] else 'FAIL'}")
    print(f"Tests Passed: {test_results['tests_passed']}")
    print(f"Tests Failed: {test_results['tests_failed']}")
    print(f"Success Rate: {test_results['tests_passed'] / (test_results['tests_passed'] + test_results['tests_failed']) * 100:.1f}%")
    print("="*50)
    
    return test_results

if __name__ == "__main__":
    main()
