"""Data Preprocessing Module for E-commerce Churn Analysis

This module handles data cleaning, feature engineering, and preprocessing
for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple


class DataPreprocessor:
    """Handle all data preprocessing tasks for churn analysis."""
    
    def __init__(self):
        """Initialize preprocessing components."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        
        # Fill numerical columns with median
        numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_cleaned[col].isnull().any():
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame without duplicates
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        removed_rows = initial_rows - len(df_cleaned)
        
        if removed_rows > 0:
            print(f"✓ Removed {removed_rows} duplicate rows")
        
        return df_cleaned
    
    def encode_categorical_variables(self, df: pd.DataFrame, 
                                    fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables using Label Encoding.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit encoders (True for training data)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Exclude target variable if present
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features for better model performance.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Check if required columns exist before creating features
        if 'Tenure' in df_features.columns:
            # Customer tenure categories
            df_features['TenureGroup'] = pd.cut(df_features['Tenure'], 
                                                bins=[0, 6, 12, 24, 48, 100],
                                                labels=['0-6M', '6-12M', '1-2Y', '2-4Y', '4Y+'])
        
        if 'CashbackAmount' in df_features.columns and 'OrderCount' in df_features.columns:
            # Cashback per order
            df_features['CashbackPerOrder'] = df_features['CashbackAmount'] / (df_features['OrderCount'] + 1)
        
        if 'OrderAmountHikeFromlastYear' in df_features.columns:
            # High value customer flag
            df_features['IsHighValueCustomer'] = (df_features['OrderAmountHikeFromlastYear'] > 50).astype(int)
        
        if 'DaySinceLastOrder' in df_features.columns:
            # Recency categories
            df_features['RecencyGroup'] = pd.cut(df_features['DaySinceLastOrder'],
                                                bins=[0, 7, 30, 90, 180, 999],
                                                labels=['0-1W', '1W-1M', '1-3M', '3-6M', '6M+'])
        
        if 'OrderCount' in df_features.columns:
            # Purchase frequency categories
            df_features['FrequencyGroup'] = pd.cut(df_features['OrderCount'],
                                                   bins=[0, 2, 5, 10, 999],
                                                   labels=['Low', 'Medium', 'High', 'VeryHigh'])
        
        return df_features
    
    def scale_features(self, X_train: pd.DataFrame, 
                      X_test: pd.DataFrame = None) -> Tuple:
        """Scale numerical features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Scaled training and test features
        """
        # Select only numerical columns
        numerical_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Fit and transform training data
        X_train_scaled = X_train.copy()
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        
        # Transform test data if provided
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data_for_modeling(self, df: pd.DataFrame, 
                                  target_col: str = 'Churn',
                                  test_size: float = 0.2,
                                  random_state: int = 42) -> Tuple:
        """Complete preprocessing pipeline for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Starting data preprocessing pipeline...")
        
        # 1. Handle missing values
        df_clean = self.handle_missing_values(df)
        print("✓ Missing values handled")
        
        # 2. Remove duplicates
        df_clean = self.remove_duplicates(df_clean)
        
        # 3. Create engineered features
        df_features = self.create_features(df_clean)
        print("✓ Features engineered")
        
        # 4. Encode categorical variables
        df_encoded = self.encode_categorical_variables(df_features, fit=True)
        print("✓ Categorical variables encoded")
        
        # 5. Separate features and target
        if target_col in df_encoded.columns:
            X = df_encoded.drop(columns=[target_col])
            y = df_encoded[target_col]
        else:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")
        
        # 7. Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        print("✓ Features scaled")
        
        print("\n✓✓ Preprocessing complete!")
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_preprocessing_summary(self, df: pd.DataFrame) -> dict:
        """Get summary of preprocessing requirements.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with preprocessing statistics
        """
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        return summary    
