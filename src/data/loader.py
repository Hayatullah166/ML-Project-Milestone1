"""
Data loading utilities for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import requests
from src.utils.config import config

class DataLoader:
    """Data loader for electricity consumption dataset."""
    
    def __init__(self):
        """Initialize data loader."""
        self.raw_path = config.get_data_path()
        self.processed_path = Path(config.get('data.processed_path', 'data/processed'))
        
    def download_data(self, url: Optional[str] = None) -> Path:
        """Download the UCI household power consumption dataset."""
        if url is None:
            url = config.get('data.dataset_url')
        
        filename = "household_power_consumption.txt"
        filepath = self.raw_path / filename
        
        if filepath.exists():
            print(f"Dataset already exists at {filepath}")
            return filepath
        
        print(f"Downloading dataset from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            self.raw_path.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Dataset downloaded successfully to {filepath}")
            return filepath
            
        except Exception as e:
            raise Exception(f"Failed to download dataset: {e}")
    
    def load_raw_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """Load raw household power consumption data."""
        if filepath is None:
            filename = "household_power_consumption.txt"
            filepath = self.raw_path / filename
        
        if not filepath.exists():
            filepath = self.download_data()
        
        print(f"Loading data from {filepath}...")
        
        # Load data with proper parsing
        df = pd.read_csv(
            filepath,
            sep=';',
            na_values=['?'],
            parse_dates=[['Date', 'Time']],
            date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'),
            low_memory=False
        )
        
        # Rename the combined datetime column
        df.rename(columns={'Date_Time': 'datetime'}, inplace=True)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def create_sample_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create synthetic sample data for testing purposes."""
        print(f"Creating {n_samples} synthetic data samples...")
        
        # Generate datetime index (hourly data)
        start_date = pd.Timestamp('2020-01-01')
        end_date = start_date + pd.Timedelta(hours=n_samples)
        datetime_index = pd.date_range(start=start_date, end=end_date, freq='h')
        
        # Generate synthetic power consumption data
        np.random.seed(42)
        
        # Base consumption with daily and weekly patterns
        hour_of_day = datetime_index.hour
        day_of_week = datetime_index.dayofweek
        
        # Daily pattern (higher consumption during day)
        daily_pattern = 2 + np.sin(2 * np.pi * hour_of_day / 24) * 0.5
        
        # Weekly pattern (lower consumption on weekends)
        weekly_pattern = np.where(day_of_week < 5, 1.0, 0.7)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(datetime_index))
        
        # Generate target variable
        base_consumption = daily_pattern * weekly_pattern + noise
        global_active_power = np.maximum(0.1, base_consumption)  # Ensure positive values
        
        # Generate other features
        data = {
            'Global_active_power': global_active_power,
            'Global_reactive_power': np.random.uniform(0, 0.5, len(datetime_index)),
            'Voltage': np.random.normal(240, 5, len(datetime_index)),
            'Global_intensity': global_active_power * np.random.uniform(3, 5, len(datetime_index)),
            'Sub_metering_1': np.random.uniform(0, 1, len(datetime_index)),
            'Sub_metering_2': np.random.uniform(0, 1, len(datetime_index)),
            'Sub_metering_3': np.random.uniform(0, 1, len(datetime_index))
        }
        
        df = pd.DataFrame(data, index=datetime_index)
        print(f"Created synthetic dataset with shape {df.shape}")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save processed dataframe to file."""
        self.processed_path.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_path / filename
        
        if filename.endswith('.csv'):
            df.to_csv(filepath)
        elif filename.endswith('.parquet'):
            df.to_parquet(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        print(f"Saved processed data to {filepath}")
        return filepath
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed dataframe from file."""
        filepath = self.processed_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .parquet")
        
        print(f"Loaded processed data from {filepath}")
        return df
