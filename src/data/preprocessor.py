"""
Data preprocessing utilities for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.config import config

class DataPreprocessor:
    """Data preprocessing pipeline for electricity consumption data."""
    
    def __init__(self):
        """Initialize preprocessor."""
        self.scaler = None
        self.feature_columns = None
        self.target_column = config.get('model.target_column', 'Global_active_power')
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw dataset."""
        print("Cleaning data...")
        
        # Remove rows with missing values in target column
        initial_rows = len(df)
        df_clean = df.dropna(subset=[self.target_column])
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows with missing target values")
        
        # Handle outliers using IQR method
        Q1 = df_clean[self.target_column].quantile(0.25)
        Q3 = df_clean[self.target_column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_clean[self.target_column] < lower_bound) | \
                       (df_clean[self.target_column] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            print(f"Found {outliers_count} outliers in target column")
            # Cap outliers instead of removing them
            df_clean.loc[df_clean[self.target_column] < lower_bound, self.target_column] = lower_bound
            df_clean.loc[df_clean[self.target_column] > upper_bound, self.target_column] = upper_bound
        
        # Fill remaining missing values with forward fill
        df_clean = df_clean.ffill()
        
        # Ensure target column is positive
        df_clean[self.target_column] = np.maximum(df_clean[self.target_column], 0.1)
        
        print(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def resample_data(self, df: pd.DataFrame, freq: str = 'h') -> pd.DataFrame:
        """Resample data to specified frequency."""
        print(f"Resampling data to {freq} frequency...")
        
        # Resample to hourly data
        df_resampled = df.resample(freq).mean()
        
        # Remove rows with all NaN values
        df_resampled = df_resampled.dropna(how='all')
        
        print(f"Resampled data shape: {df_resampled.shape}")
        return df_resampled
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        print("Creating time features...")
        
        df_features = df.copy()
        
        # Extract time components
        df_features['hour'] = df_features.index.hour
        df_features['day_of_week'] = df_features.index.dayofweek
        df_features['day_of_month'] = df_features.index.day
        df_features['month'] = df_features.index.month
        df_features['quarter'] = df_features.index.quarter
        df_features['year'] = df_features.index.year
        
        # Cyclical encoding for hour and day of week
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        # Binary features
        df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
        df_features['is_night'] = ((df_features['hour'] >= 22) | (df_features['hour'] <= 6)).astype(int)
        df_features['is_work_hours'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
        
        print(f"Added time features. New shape: {df_features.shape}")
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, lag_hours: List[int]) -> pd.DataFrame:
        """Create lagged features for the target variable."""
        print(f"Creating lag features for hours: {lag_hours}")
        
        df_lagged = df.copy()
        
        for lag in lag_hours:
            df_lagged[f'{self.target_column}_lag_{lag}h'] = df_lagged[self.target_column].shift(lag)
        
        print(f"Added lag features. New shape: {df_lagged.shape}")
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Create rolling window features."""
        print(f"Creating rolling features for windows: {windows}")
        
        df_rolling = df.copy()
        
        for window in windows:
            # Rolling mean
            df_rolling[f'{self.target_column}_rolling_mean_{window}h'] = \
                df_rolling[self.target_column].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            df_rolling[f'{self.target_column}_rolling_std_{window}h'] = \
                df_rolling[self.target_column].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max
            df_rolling[f'{self.target_column}_rolling_min_{window}h'] = \
                df_rolling[self.target_column].rolling(window=window, min_periods=1).min()
            df_rolling[f'{self.target_column}_rolling_max_{window}h'] = \
                df_rolling[self.target_column].rolling(window=window, min_periods=1).max()
        
        print(f"Added rolling features. New shape: {df_rolling.shape}")
        return df_rolling
    
    def create_target_variable(self, df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Create target variable for forecasting."""
        print(f"Creating target variable with {horizon}h horizon...")
        
        df_target = df.copy()
        df_target['target'] = df_target[self.target_column].shift(-horizon)
        
        print(f"Added target variable. New shape: {df_target.shape}")
        return df_target
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix for modeling."""
        print("Preparing feature matrix...")
        
        # Get feature configuration
        lag_features = config.get('features.lag_features', [1, 2, 3, 24, 48])
        rolling_windows = config.get('features.rolling_windows', [24, 168])
        
        # Apply all transformations
        df_processed = self.clean_data(df)
        df_processed = self.resample_data(df_processed)
        df_processed = self.create_time_features(df_processed)
        df_processed = self.create_lag_features(df_processed, lag_features)
        df_processed = self.create_rolling_features(df_processed, rolling_windows)
        df_processed = self.create_target_variable(df_processed)
        
        # Remove rows with NaN values (from lag and target creation)
        df_processed = df_processed.dropna()
        
        # Select feature columns (exclude target and original target column)
        exclude_columns = [self.target_column, 'target']
        feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
        
        self.feature_columns = feature_columns
        
        print(f"Feature preparation completed. Features: {len(feature_columns)}")
        print(f"Final dataset shape: {df_processed.shape}")
        
        return df_processed, feature_columns
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        print(f"Splitting data: train={1-test_size-validation_size:.1%}, validation={validation_size:.1%}, test={test_size:.1%}")
        
        n_samples = len(df)
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - validation_size))
        
        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]
        
        print(f"Split sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Scale features using StandardScaler."""
        print("Scaling features...")
        
        self.scaler = StandardScaler()
        
        # Fit scaler on training data
        train_scaled = train_df.copy()
        train_scaled[self.feature_columns] = self.scaler.fit_transform(train_df[self.feature_columns])
        
        # Transform validation and test data
        val_scaled = val_df.copy()
        val_scaled[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])
        
        test_scaled = test_df.copy()
        test_scaled[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])
        
        print("Feature scaling completed")
        return train_scaled, val_scaled, test_scaled
