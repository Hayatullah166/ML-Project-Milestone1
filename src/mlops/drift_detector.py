"""
Simplified drift detection for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error
from scipy import stats

from src.utils.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor

class DriftDetector:
    """Simplified drift detection for energy consumption data."""
    
    def __init__(self):
        """Initialize drift detector."""
        self.reference_data = None
        self.reference_path = Path(config.get('data.reference_path', 'data/reference'))
        self.reports_path = Path(config.get('evidently.report_path', 'reports'))
        self.dashboard_path = Path(config.get('evidently.dashboard_path', 'reports/dashboard'))
        
        # Create directories
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.dashboard_path.mkdir(parents=True, exist_ok=True)
        
        # Drift thresholds
        self.drift_threshold = config.get('mlops.drift_threshold', 0.1)
        self.performance_threshold = config.get('mlops.performance_threshold', 0.05)
        
        # Initialize data loader and preprocessor
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        
    def set_reference_data(self, df: pd.DataFrame, feature_columns: list):
        """Set reference dataset for drift detection."""
        print("Setting reference dataset for drift detection...")
        
        # Store reference data
        self.reference_data = df[feature_columns + ['target']].copy()
        
        # Save reference data
        self.reference_path.mkdir(parents=True, exist_ok=True)
        reference_file = self.reference_path / "reference_data.parquet"
        self.reference_data.to_parquet(reference_file)
        
        print(f"Reference dataset saved with {len(self.reference_data)} samples")
        
    def load_reference_data(self) -> Optional[pd.DataFrame]:
        """Load reference dataset."""
        reference_file = self.reference_path / "reference_data.parquet"
        
        if not reference_file.exists():
            print("No reference dataset found")
            return None
        
        return pd.read_parquet(reference_file)
    
    def check_drift(self, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Check for data drift between reference and current data."""
        print("Checking for data drift...")
        
        # Load reference data
        if self.reference_data is None:
            self.reference_data = self.load_reference_data()
        
        if self.reference_data is None:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drift_details": {"error": "No reference data available"},
                "recommendation": "Set reference data first",
                "timestamp": datetime.now().isoformat()
            }
        
        # Get current data
        if current_data is None:
            current_data = self._get_current_data()
        
        if current_data is None or len(current_data) == 0:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drift_details": {"error": "No current data available"},
                "recommendation": "Collect current data",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Prepare data for drift detection
            reference_df = self.reference_data.copy()
            current_df = current_data.copy()
            
            # Ensure same columns
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            reference_df = reference_df[common_columns]
            current_df = current_df[common_columns]
            
            # Calculate drift metrics
            drift_metrics = self._calculate_drift_metrics(reference_df, current_df)
            
            # Determine if drift is detected
            drift_detected = drift_metrics['overall_drift_score'] > self.drift_threshold
            
            # Generate recommendation
            recommendation = self._generate_recommendation(drift_metrics, drift_detected)
            
            # Save report
            report_file = self.reports_path / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(drift_metrics, f, indent=2)
            
            result = {
                "drift_detected": drift_detected,
                "drift_score": drift_metrics['overall_drift_score'],
                "drift_details": drift_metrics,
                "recommendation": recommendation,
                "timestamp": datetime.now().isoformat(),
                "report_path": str(report_file)
            }
            
            print(f"Drift check completed. Drift detected: {drift_detected}")
            return result
            
        except Exception as e:
            print(f"Error in drift detection: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drift_details": {"error": str(e)},
                "recommendation": "Check data quality",
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_drift_metrics(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate drift metrics using statistical tests."""
        metrics = {}
        
        try:
            # Statistical tests for numerical features
            numerical_features = reference_df.select_dtypes(include=[np.number]).columns
            
            drift_scores = []
            feature_drifts = {}
            
            for feature in numerical_features:
                if feature in current_df.columns:
                    ref_values = reference_df[feature].dropna()
                    curr_values = current_df[feature].dropna()
                    
                    if len(ref_values) > 0 and len(curr_values) > 0:
                        # Kolmogorov-Smirnov test
                        ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                        
                        # Mann-Whitney U test
                        mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
                        
                        # Mean difference
                        ref_std = ref_values.std()
                        if ref_std == 0:
                            mean_diff = 0.0
                        else:
                            mean_diff = abs(curr_values.mean() - ref_values.mean()) / ref_std
                        
                        # Store feature drift
                        drift_detected = ks_pvalue < 0.05 or mean_diff > 0.1
                        feature_drifts[feature] = {
                            'ks_statistic': float(ks_stat),
                            'ks_pvalue': float(ks_pvalue),
                            'mw_statistic': float(mw_stat),
                            'mw_pvalue': float(mw_pvalue),
                            'mean_difference': float(mean_diff),
                            'drift_detected': bool(drift_detected)
                        }
                        
                        # Use KS statistic as drift score
                        drift_scores.append(ks_stat)
            
            # Calculate overall drift score
            overall_drift_score = max(drift_scores) if drift_scores else 0.0
            
            metrics = {
                'overall_drift_score': overall_drift_score,
                'feature_drifts': feature_drifts,
                'drift_detected_features': sum(1 for f in feature_drifts.values() if f['drift_detected']),
                'total_features': len(feature_drifts)
            }
            
        except Exception as e:
            print(f"Error calculating drift metrics: {e}")
            metrics = {
                'overall_drift_score': 0.0,
                'feature_drifts': {},
                'drift_detected_features': 0,
                'total_features': 0,
                'error': str(e)
            }
        
        return metrics
    
    def _generate_recommendation(self, drift_metrics: Dict[str, Any], drift_detected: bool) -> str:
        """Generate recommendation based on drift metrics."""
        if not drift_detected:
            return "No significant drift detected. Model performance is stable."
        
        overall_score = drift_metrics.get('overall_drift_score', 0.0)
        drift_features = drift_metrics.get('drift_detected_features', 0)
        total_features = drift_metrics.get('total_features', 1)
        
        drift_percentage = (drift_features / total_features) * 100
        
        if overall_score > 0.3 or drift_percentage > 50:
            return f"High drift detected ({drift_percentage:.1f}% of features). Immediate model retraining recommended."
        elif overall_score > 0.2 or drift_percentage > 30:
            return f"Moderate drift detected ({drift_percentage:.1f}% of features). Consider retraining model soon."
        elif overall_score > 0.1 or drift_percentage > 10:
            return f"Low drift detected ({drift_percentage:.1f}% of features). Monitor closely and consider retraining."
        else:
            return "Minimal drift detected. Continue monitoring."
    
    def _get_current_data(self) -> Optional[pd.DataFrame]:
        """Get current data for drift detection."""
        try:
            # Try to load latest processed data
            processed_files = list(Path(config.get('data.processed_path', 'data/processed')).glob("*.parquet"))
            
            if not processed_files:
                # Generate sample current data for demonstration
                return self._generate_sample_current_data()
            
            # Load most recent processed data
            latest_file = max(processed_files, key=lambda x: x.stat().st_mtime)
            current_data = pd.read_parquet(latest_file)
            
            # Take recent samples (last 1000 records)
            if len(current_data) > 1000:
                current_data = current_data.tail(1000)
            
            return current_data
            
        except Exception as e:
            print(f"Error getting current data: {e}")
            return self._generate_sample_current_data()
    
    def _generate_sample_current_data(self) -> pd.DataFrame:
        """Generate sample current data for demonstration."""
        print("Generating sample current data for drift detection...")
        
        # Generate synthetic data with slight drift
        np.random.seed(42)
        n_samples = 1000
        
        # Create datetime index
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=n_samples)
        datetime_index = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Generate features with slight drift
        hour_of_day = datetime_index.hour
        day_of_week = datetime_index.dayofweek
        
        # Add slight drift to the pattern
        drift_factor = 1.1  # 10% increase
        daily_pattern = (2 + np.sin(2 * np.pi * hour_of_day / 24) * 0.5) * drift_factor
        weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
        
        noise = np.random.normal(0, 0.1, len(datetime_index))
        base_consumption = daily_pattern * weekly_pattern + noise
        global_active_power = np.maximum(0.1, base_consumption)
        
        # Create sample data
        data = {
            'Global_active_power': global_active_power,
            'Global_reactive_power': np.random.uniform(0, 0.5, len(datetime_index)),
            'Voltage': np.random.normal(240, 5, len(datetime_index)),
            'Global_intensity': global_active_power * np.random.uniform(3, 5, len(datetime_index)),
            'Sub_metering_1': np.random.uniform(0, 1, len(datetime_index)),
            'Sub_metering_2': np.random.uniform(0, 1, len(datetime_index)),
            'Sub_metering_3': np.random.uniform(0, 1, len(datetime_index)),
            'hour': hour_of_day,
            'day_of_week': day_of_week,
            'month': datetime_index.month,
            'quarter': datetime_index.quarter,
            'year': datetime_index.year,
            'hour_sin': np.sin(2 * np.pi * hour_of_day / 24),
            'hour_cos': np.cos(2 * np.pi * hour_of_day / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'is_weekend': (day_of_week >= 5).astype(int),
            'is_night': ((hour_of_day >= 22) | (hour_of_day <= 6)).astype(int),
            'is_work_hours': ((hour_of_day >= 9) & (hour_of_day <= 17)).astype(int),
            'target': global_active_power  # Shift target for demonstration
        }
        
        df = pd.DataFrame(data, index=datetime_index)
        
        # Add some lag features
        for lag in [1, 2, 3, 24]:
            df[f'Global_active_power_lag_{lag}h'] = df['Global_active_power'].shift(lag)
        
        # Add rolling features
        for window in [24, 168]:
            df[f'Global_active_power_rolling_mean_{window}h'] = df['Global_active_power'].rolling(window=window, min_periods=1).mean()
            df[f'Global_active_power_rolling_std_{window}h'] = df['Global_active_power'].rolling(window=window, min_periods=1).std()
        
        # Remove NaN values
        df = df.dropna()
        
        print(f"Generated sample current data with {len(df)} samples")
        return df
    
    def create_dashboard(self, current_data: Optional[pd.DataFrame] = None):
        """Create simple dashboard for monitoring."""
        print("Creating monitoring dashboard...")
        
        if self.reference_data is None:
            self.reference_data = self.load_reference_data()
        
        if self.reference_data is None:
            print("No reference data available for dashboard")
            return
        
        if current_data is None:
            current_data = self._get_current_data()
        
        if current_data is None:
            print("No current data available for dashboard")
            return
        
        try:
            # Run drift check
            drift_result = self.check_drift(current_data)
            
            # Create simple dashboard data
            # Ensure JSON-serializable (cast numpy types to native Python)
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": bool(drift_result["drift_detected"]),
                "drift_score": float(drift_result["drift_score"]),
                "recommendation": str(drift_result["recommendation"]),
                "reference_samples": int(len(self.reference_data)),
                "current_samples": int(len(current_data))
            }
            
            # Save dashboard
            dashboard_file = self.dashboard_path / "dashboard.json"
            with open(dashboard_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
            
            print(f"Dashboard saved to {dashboard_file}")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
    
    def run_stability_tests(self, current_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run data stability tests."""
        print("Running data stability tests...")
        
        if self.reference_data is None:
            self.reference_data = self.load_reference_data()
        
        if self.reference_data is None:
            return {"error": "No reference data available"}
        
        if current_data is None:
            current_data = self._get_current_data()
        
        if current_data is None:
            return {"error": "No current data available"}
        
        try:
            # Simple stability tests
            test_results = {}
            
            # Test 1: Data completeness
            ref_completeness = (1 - self.reference_data.isnull().sum().sum() / (len(self.reference_data) * len(self.reference_data.columns))) * 100
            curr_completeness = (1 - current_data.isnull().sum().sum() / (len(current_data) * len(current_data.columns))) * 100
            
            test_results["data_completeness"] = {
                "status": "SUCCESS" if abs(ref_completeness - curr_completeness) < 5 else "FAIL",
                "reference_completeness": ref_completeness,
                "current_completeness": curr_completeness
            }
            
            # Test 2: Data range stability
            ref_range = self.reference_data.select_dtypes(include=[np.number]).describe()
            curr_range = current_data.select_dtypes(include=[np.number]).describe()
            
            range_stable = True
            for col in ref_range.columns:
                if col in curr_range.columns:
                    ref_mean = ref_range.loc['mean', col]
                    curr_mean = curr_range.loc['mean', col]
                    if ref_mean != 0 and abs(curr_mean - ref_mean) / abs(ref_mean) > 0.2:
                        range_stable = False
                        break
            
            test_results["data_range_stability"] = {
                "status": "SUCCESS" if range_stable else "FAIL",
                "stable": range_stable
            }
            
            # Test 3: Sample size stability
            size_ratio = len(current_data) / len(self.reference_data)
            size_stable = 0.5 <= size_ratio <= 2.0
            
            test_results["sample_size_stability"] = {
                "status": "SUCCESS" if size_stable else "FAIL",
                "size_ratio": size_ratio,
                "stable": size_stable
            }
            
            overall_status = "PASS" if all(test["status"] == "SUCCESS" for test in test_results.values()) else "FAIL"
            
            return {
                "test_results": test_results,
                "overall_status": overall_status
            }
            
        except Exception as e:
            print(f"Error running stability tests: {e}")
            return {"error": str(e)}