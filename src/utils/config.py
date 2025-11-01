"""
Configuration management for Energy Consumption Prediction project.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for the project."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_path(self, subpath: str = "") -> Path:
        """Get data directory path."""
        base_path = Path(self.get('data.raw_path', 'data/raw'))
        return base_path / subpath if subpath else base_path
    
    def get_model_path(self, subpath: str = "") -> Path:
        """Get model directory path."""
        base_path = Path(self.get('models.trained_path', 'models/trained'))
        return base_path / subpath if subpath else base_path
    
    def get_reports_path(self, subpath: str = "") -> Path:
        """Get reports directory path."""
        base_path = Path(self.get('evidently.report_path', 'reports'))
        return base_path / subpath if subpath else base_path

# Global configuration instance
config = Config()
