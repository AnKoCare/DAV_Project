"""
Data loader module for Gaming Behavior Prediction Project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and basic preprocessing"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            file_path = self.data_path / filename
            self.raw_data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully from {file_path}")
            logger.info(f"Data shape: {self.raw_data.shape}")
            return self.raw_data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'dtypes': self.raw_data.dtypes.to_dict(),
            'missing_values': self.raw_data.isnull().sum().to_dict(),
            'duplicates': self.raw_data.duplicated().sum(),
            'memory_usage': self.raw_data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def basic_cleaning(self, remove_duplicates: bool = True, 
                      fill_missing: bool = True) -> pd.DataFrame:
        """Perform basic data cleaning"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.processed_data = self.raw_data.copy()
        
        # Remove duplicates
        if remove_duplicates:
            before_shape = self.processed_data.shape[0]
            self.processed_data = self.processed_data.drop_duplicates()
            after_shape = self.processed_data.shape[0]
            logger.info(f"Removed {before_shape - after_shape} duplicate rows")
        
        # Handle missing values
        if fill_missing:
            self._handle_missing_values()
        
        logger.info(f"Data cleaning completed. Final shape: {self.processed_data.shape}")
        return self.processed_data
    
    def _handle_missing_values(self):
        """Handle missing values with appropriate strategies"""
        for column in self.processed_data.columns:
            if self.processed_data[column].isnull().any():
                if self.processed_data[column].dtype in ['int64', 'float64']:
                    # Fill numerical columns with median
                    median_value = self.processed_data[column].median()
                    self.processed_data[column].fillna(median_value, inplace=True)
                    logger.info(f"Filled missing values in {column} with median: {median_value}")
                else:
                    # Fill categorical columns with mode
                    mode_value = self.processed_data[column].mode().iloc[0] if not self.processed_data[column].mode().empty else 'Unknown'
                    self.processed_data[column].fillna(mode_value, inplace=True)
                    logger.info(f"Filled missing values in {column} with mode: {mode_value}")
    
    def split_features_target(self, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Split data into features and target"""
        if self.processed_data is None:
            raise ValueError("No processed data available. Call basic_cleaning() first.")
        
        if target_column not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        X = self.processed_data.drop(columns=[target_column])
        y = self.processed_data[target_column]
        
        logger.info(f"Data split into features ({X.shape}) and target ({y.shape})")
        return X, y
    
    def save_processed_data(self, filename: str, data: Optional[pd.DataFrame] = None):
        """Save processed data to CSV"""
        if data is None:
            data = self.processed_data
        
        if data is None:
            raise ValueError("No data to save")
        
        output_path = self.data_path.parent / "processed" / filename
        data.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    
    def get_data_sample(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of the data"""
        if self.processed_data is not None:
            return self.processed_data.head(n)
        elif self.raw_data is not None:
            return self.raw_data.head(n)
        else:
            raise ValueError("No data loaded")
    
    def get_data_statistics(self) -> dict:
        """Get statistical summary of the data"""
        if self.processed_data is None:
            data = self.raw_data
        else:
            data = self.processed_data
            
        if data is None:
            raise ValueError("No data loaded")
        
        stats = {
            'numerical_summary': data.describe(),
            'categorical_summary': data.select_dtypes(include=['object']).describe() if not data.select_dtypes(include=['object']).empty else None,
            'correlation_matrix': data.corr() if len(data.select_dtypes(include=[np.number]).columns) > 1 else None
        }
        
        return stats 