"""
Data download script for Energy Consumption Prediction project.
"""

import requests
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.utils.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_data():
    """Download the UCI household power consumption dataset."""
    logger.info("Starting data download...")
    
    try:
        
        data_loader = DataLoader()
        
    
        filepath = data_loader.download_data()
        
        logger.info(f"Data downloaded successfully to {filepath}")
        
        # Verify download
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Downloaded file size: {file_size:.2f} MB")
            
            # Load a sample to verify
            df = data_loader.load_raw_data()
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return True
        else:
            logger.error("Download failed - file not found")
            return False
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def main():
    """Main function."""
    logger.info("Energy Consumption Prediction - Data Download")
    
    success = download_data()
    
    if success:
        logger.info("Data download completed successfully!")
    else:
        logger.error("Data download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
