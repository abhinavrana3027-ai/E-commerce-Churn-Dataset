"""Data Loader Module for E-commerce Churn Analysis

This module handles loading and exploring e-commerce customer data
for marketing analysis and churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple

class DataLoader:
    """Load and preprocess e-commerce customer data."""
    
    def __init__(self, filepath: str):
        """Initialize with data file path.
        
        Args:
            filepath: Path to Excel or CSV file containing customer data
        """
        self.filepath = filepath
        self.df = None
        self.original_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from Excel or CSV file.
        
        Returns:
            DataFrame with customer data
        """
        try:
            if self.filepath.endswith('.xlsx'):
                self.df = pd.read_excel(self.filepath)
            else:
                self.df = pd.read_csv(self.filepath)
            self.original_df = self.df.copy()
            print(f"✓ Data loaded successfully: {self.df.shape[0]} customers, {self.df.shape[1]} features")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
            
    def get_data_info(self) -> dict:
        """Get comprehensive data information for marketing analysis.
        
        Returns:
            Dictionary with data statistics and insights
        """
        if self.df is None:
            return None
            
        info = {
            'total_customers': len(self.df),
            'features': self.df.shape[1],
            'missing_values': self.df.isnull().sum().to_dict(),
            'feature_types': self.df.dtypes.to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        return info
        
    def get_churn_distribution(self) -> dict:
        """Analyze churn distribution for marketing insights.
        
        Returns:
            Dictionary with churn metrics
        """
        if self.df is None or 'Churn' not in self.df.columns:
            return None
            
        churn_counts = self.df['Churn'].value_counts()
        churn_rate = (churn_counts.get(1, 0) / len(self.df)) * 100
        
        return {
            'active_customers': int(churn_counts.get(0, 0)),
            'churned_customers': int(churn_counts.get(1, 0)),
            'churn_rate': round(churn_rate, 2)
        }
        
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for numerical features.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.df is None:
            return None
        return self.df.describe()
        
    def get_head(self, n: int = 5) -> pd.DataFrame:
        """Get first n rows of data.
        
        Args:
            n: Number of rows to return
            
        Returns:
            First n rows of the dataset
        """
        return self.df.head(n) if self.df is not None else None
