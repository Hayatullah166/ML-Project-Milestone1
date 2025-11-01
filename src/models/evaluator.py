"""
Model evaluation utilities for Energy Consumption Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import json

class ModelEvaluator:
    """Model evaluation utilities for energy consumption prediction."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.results = {}
        
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'smape': np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(y_true - y_pred),
            'std_error': np.std(y_true - y_pred)
        }
        
        return metrics
    
    def evaluate_time_series(self, y_true: pd.Series, y_pred: np.ndarray, 
                           model_name: str) -> Dict[str, Any]:
        """Evaluate model performance on time series data."""
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Time series specific metrics
        ts_metrics = {
            'directional_accuracy': self._calculate_directional_accuracy(y_true, y_pred),
            'persistence_skill': self._calculate_persistence_skill(y_true, y_pred),
            'residual_autocorr': self._calculate_residual_autocorr(residuals)
        }
        
        metrics.update(ts_metrics)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'residuals': residuals,
            'y_true': y_true
        }
        
        return metrics
    
    def _calculate_directional_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        true_direction = np.diff(y_true.values) > 0
        pred_direction = np.diff(y_pred) > 0
        
        if len(true_direction) == 0:
            return 0.0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def _calculate_persistence_skill(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate persistence skill score."""
        # Persistence forecast (last value)
        persistence_pred = np.full_like(y_pred, y_true.iloc[0])
        persistence_mae = mean_absolute_error(y_true, persistence_pred)
        model_mae = mean_absolute_error(y_true, y_pred)
        
        if persistence_mae == 0:
            return 0.0
        
        return (1 - model_mae / persistence_mae) * 100
    
    def _calculate_residual_autocorr(self, residuals: np.ndarray) -> float:
        """Calculate residual autocorrelation at lag 1."""
        if len(residuals) < 2:
            return 0.0
        
        return np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple models and return comparison DataFrame."""
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MAPE (%)': metrics['mape'],
                'SMAPE (%)': metrics['smape'],
                'Directional Accuracy (%)': metrics['directional_accuracy'],
                'Persistence Skill (%)': metrics['persistence_skill']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('MAE')
        
        return comparison_df
    
    def plot_predictions(self, model_name: str, y_true: pd.Series, y_pred: np.ndarray, 
                        save_path: Optional[Path] = None, n_samples: int = 1000):
        """Plot predictions vs actual values."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # Sample data for plotting
        if len(y_true) > n_samples:
            indices = np.random.choice(len(y_true), n_samples, replace=False)
            y_true_sample = y_true.iloc[indices]
            y_pred_sample = y_pred[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
        
        # 1. Time series plot
        axes[0, 0].plot(y_true_sample.index, y_true_sample.values, label='Actual', alpha=0.7)
        axes[0, 0].plot(y_true_sample.index, y_pred_sample, label='Predicted', alpha=0.7)
        axes[0, 0].set_title('Time Series Comparison')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Energy Consumption')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(y_true_sample.values, y_pred_sample, alpha=0.6)
        min_val = min(y_true_sample.min(), y_pred_sample.min())
        max_val = max(y_true_sample.max(), y_pred_sample.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_title('Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = y_true_sample.values - y_pred_sample
        axes[1, 0].scatter(y_pred_sample, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Residuals histogram
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None):
        """Plot model comparison charts."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        # MAE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[0, 0].set_title('Mean Absolute Error')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0, 1].set_title('Root Mean Squared Error')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'])
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'])
        axes[1, 1].set_title('Mean Absolute Percentage Error')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, model_results: Dict[str, Dict[str, Any]], 
                       save_path: Optional[Path] = None) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("# Model Evaluation Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Model comparison table
        comparison_df = self.compare_models(model_results)
        report.append("## Model Comparison\n")
        report.append(comparison_df.to_string(index=False))
        report.append("\n")
        
        # Detailed metrics for each model
        report.append("## Detailed Metrics\n")
        for model_name, results in model_results.items():
            metrics = results['metrics']
            report.append(f"### {model_name}\n")
            report.append(f"- **MAE**: {metrics['mae']:.4f}")
            report.append(f"- **RMSE**: {metrics['rmse']:.4f}")
            report.append(f"- **R²**: {metrics['r2']:.4f}")
            report.append(f"- **MAPE**: {metrics['mape']:.2f}%")
            report.append(f"- **SMAPE**: {metrics['smape']:.2f}%")
            report.append(f"- **Directional Accuracy**: {metrics['directional_accuracy']:.2f}%")
            report.append(f"- **Persistence Skill**: {metrics['persistence_skill']:.2f}%")
            report.append(f"- **Max Error**: {metrics['max_error']:.4f}")
            report.append(f"- **Mean Error**: {metrics['mean_error']:.4f}")
            report.append(f"- **Std Error**: {metrics['std_error']:.4f}")
            report.append(f"- **Residual Autocorr**: {metrics['residual_autocorr']:.4f}")
            report.append("\n")
        
        # Best model recommendation
        best_model = comparison_df.iloc[0]
        report.append("## Recommendation\n")
        report.append(f"**Best Model**: {best_model['Model']}")
        report.append(f"- MAE: {best_model['MAE']:.4f}")
        report.append(f"- RMSE: {best_model['RMSE']:.4f}")
        report.append(f"- R²: {best_model['R²']:.4f}")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
