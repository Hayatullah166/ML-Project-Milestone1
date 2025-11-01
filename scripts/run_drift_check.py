"""
Run drift detection check for Energy Consumption Prediction project.
"""

import sys
from pathlib import Path
import logging


sys.path.append(str(Path(__file__).parent.parent))

from src.mlops.drift_detector import DriftDetector
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_drift_check():
    """Run drift detection check."""
    logger.info("Starting drift detection check...")
    
    try:
       
        drift_detector = DriftDetector()
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        
        if drift_detector.reference_data is None:
            logger.info("No reference data found. Setting up reference data...")
            
            
            df = data_loader.create_sample_data(n_samples=2000)
            df_processed, feature_columns = preprocessor.prepare_features(df)
            
           
            drift_detector.set_reference_data(df_processed, feature_columns)
            logger.info("Reference data set successfully")
        
        
        logger.info("Running drift detection...")
        drift_result = drift_detector.check_drift()
        
        
        print("\n" + "="*50)
        print("DRIFT DETECTION RESULTS")
        print("="*50)
        print(f"Drift Detected: {'Yes' if drift_result['drift_detected'] else 'No'}")
        print(f"Drift Score: {drift_result['drift_score']:.4f}")
        print(f"Recommendation: {drift_result['recommendation']}")
        
        if 'report_path' in drift_result:
            print(f"Report saved to: {drift_result['report_path']}")
        
        print("="*50)
        
      
        logger.info("Running stability tests...")
        stability_result = drift_detector.run_stability_tests()
        
        print("\n" + "="*50)
        print("STABILITY TEST RESULTS")
        print("="*50)
        print(f"Overall Status: {stability_result.get('overall_status', 'Unknown')}")
        
        if 'test_results' in stability_result:
            for test_name, test_result in stability_result['test_results'].items():
                status_icon = "✅" if test_result['status'] == "SUCCESS" else "❌"
                print(f"{status_icon} {test_name}: {test_result['status']}")
        
        print("="*50)
        
        
        logger.info("Creating monitoring dashboard...")
        drift_detector.create_dashboard()
        
        return True
        
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("Energy Consumption Prediction - Drift Detection Check")
    
    success = run_drift_check()
    
    if success:
        logger.info("Drift detection check completed successfully!")
    else:
        logger.error("Drift detection check failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
