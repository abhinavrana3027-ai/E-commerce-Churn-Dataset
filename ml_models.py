"""Machine Learning Models for E-commerce Churn Prediction

This module implements multiple ML algorithms for churn prediction
and provides model training, evaluation, and comparison functionality.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ChurnPredictor:
    """Train and evaluate ML models for churn prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize with default models."""
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.results = {}
        
    def _initialize_models(self) -> Dict:
        """Initialize all ML models with default parameters."""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state, max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf', probability=True, random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB()
        }
        return models
    
    def train_model(self, model_name: str, X_train, y_train):
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        print(f"Training {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"✓ {model_name} trained successfully")
        
    def train_all_models(self, X_train, y_train):
        """Train all models."""
        print("Starting model training...\n")
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
        print("\n✓✓ All models trained successfully!")
        
    def evaluate_model(self, model_name: str, X_test, y_test) -> Dict:
        """Evaluate a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_all_models(self, X_test, y_test) -> pd.DataFrame:
        """Evaluate all trained models."""
        print("Evaluating models...\n")
        results_list = []
        
        for model_name in self.trained_models.keys():
            metrics = self.evaluate_model(model_name, X_test, y_test)
            results_list.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            })
            print(f"{model_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print()
        
        results_df = pd.DataFrame(results_list)
        return results_df.sort_values('f1_score', ascending=False)
    
    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Dict]:
        """Get the best performing model based on a metric."""
        if not self.results:
            raise ValueError("No models have been evaluated yet")
        
        best_model = max(self.results.items(), 
                        key=lambda x: x[1][metric] if x[1][metric] is not None else 0)
        return best_model
    
    def get_feature_importance(self, model_name: str, 
                              feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        
        # Check if model has feature_importances_
        if not hasattr(model, 'feature_importances_'):
            print(f"{model_name} does not support feature importance")
            return None
        
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def get_classification_report(self, model_name: str) -> str:
        """Get detailed classification report for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not evaluated yet")
        
        # Reconstruct y_test and y_pred from results (you'll need to pass these)
        # This is a simplified version
        return "Classification report available after evaluation"
    
    def predict_churn(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X)
        return predictions
    
    def predict_churn_probability(self, model_name: str, 
                                 X: pd.DataFrame) -> np.ndarray:
        """Get churn probability predictions."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.trained_models[model_name]
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"{model_name} does not support probability predictions")
        
        probabilities = model.predict_proba(X)[:, 1]
        return probabilities
    
    def save_results_to_csv(self, results_df: pd.DataFrame, 
                           filepath: str = 'model_results.csv'):
        """Save model comparison results to CSV."""
        results_df.to_csv(filepath, index=False)
        print(f"✓ Results saved to {filepath}")
