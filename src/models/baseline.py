"""
Baseline models for Energy Consumption Prediction project.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Union, Optional

class NaiveBaseline(BaseEstimator, RegressorMixin):
    """Naive baseline that predicts the last observed value."""
    
    def __init__(self):
        self.last_value_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveBaseline':
        """Fit the naive baseline model."""
        self.last_value_ = y.iloc[-1]
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the last observed value."""
        return np.full(len(X), self.last_value_)

class SeasonalNaiveBaseline(BaseEstimator, RegressorMixin):
    """Seasonal naive baseline that predicts the same hour from previous day."""
    
    def __init__(self, seasonal_period: int = 24):
        self.seasonal_period = seasonal_period
        self.seasonal_values_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SeasonalNaiveBaseline':
        """Fit the seasonal naive baseline model."""
        # Store the last seasonal_period values
        self.seasonal_values_ = y.iloc[-self.seasonal_period:].values
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using seasonal pattern."""
        predictions = []
        for i in range(len(X)):
            # Use modulo to cycle through seasonal values
            idx = i % self.seasonal_period
            predictions.append(self.seasonal_values_[idx])
        return np.array(predictions)

class MovingAverageBaseline(BaseEstimator, RegressorMixin):
    """Moving average baseline model."""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.moving_avg_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MovingAverageBaseline':
        """Fit the moving average baseline model."""
        self.moving_avg_ = y.iloc[-self.window_size:].mean()
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using moving average."""
        return np.full(len(X), self.moving_avg_)

class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Persistence baseline that predicts the last value."""
    
    def __init__(self):
        self.last_value_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'PersistenceBaseline':
        """Fit the persistence baseline model."""
        self.last_value_ = y.iloc[-1]
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using persistence."""
        return np.full(len(X), self.last_value_)

class LinearTrendBaseline(BaseEstimator, RegressorMixin):
    """Linear trend baseline model."""
    
    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.last_index_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearTrendBaseline':
        """Fit the linear trend baseline model."""
        # Use the last 100 points to fit the trend
        n_points = min(100, len(y))
        y_recent = y.iloc[-n_points:]
        x_recent = np.arange(len(y_recent))
        
        # Fit linear regression
        self.slope_ = np.polyfit(x_recent, y_recent, 1)[0]
        self.intercept_ = y_recent.iloc[-1] - self.slope_ * (len(y_recent) - 1)
        self.last_index_ = len(y) - 1
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using linear trend."""
        predictions = []
        for i in range(len(X)):
            steps_ahead = i + 1
            pred = self.intercept_ + self.slope_ * (self.last_index_ + steps_ahead)
            predictions.append(pred)
        return np.array(predictions)
