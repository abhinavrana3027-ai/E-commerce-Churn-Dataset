"""Visualization Module for E-commerce Churn Analysis

This module creates professional visualizations for data exploration
and model performance evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style for professional-looking plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ChurnVisualizer:
    """Create visualizations for churn analysis."""
    
    def __init__(self, figsize: tuple = (12, 6)):
        """Initialize visualizer with default figure size."""
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    def plot_churn_distribution(self, df: pd.DataFrame, 
                               target_col: str = 'Churn',
                               save_path: Optional[str] = None):
        """Plot churn distribution with counts and percentages."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Count plot
        churn_counts = df[target_col].value_counts()
        ax1.bar(churn_counts.index, churn_counts.values, 
                color=[self.colors[1], self.colors[0]])
        ax1.set_xlabel('Churn Status')
        ax1.set_ylabel('Count')
        ax1.set_title('Customer Churn Distribution')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Retained', 'Churned'])
        
        # Add value labels
        for i, v in enumerate(churn_counts.values):
            ax1.text(i, v, str(v), ha='center', va='bottom')
        
        # Pie chart
        churn_pct = df[target_col].value_counts(normalize=True) * 100
        ax2.pie(churn_pct.values, labels=['Retained', 'Churned'], 
                autopct='%1.1f%%', colors=[self.colors[1], self.colors[0]],
                startangle=90)
        ax2.set_title('Churn Rate Percentage')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_distribution(self, df: pd.DataFrame,
                                  features: List[str],
                                  target_col: str = 'Churn',
                                  save_path: Optional[str] = None):
        """Plot distribution of features by churn status."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature in df.columns:
                if df[feature].dtype in ['int64', 'float64']:
                    # Numerical features - box plot
                    df.boxplot(column=feature, by=target_col, ax=axes[idx])
                    axes[idx].set_title(f'{feature} by Churn Status')
                    axes[idx].set_xlabel('Churn')
                else:
                    # Categorical features - count plot
                    pd.crosstab(df[feature], df[target_col]).plot(
                        kind='bar', ax=axes[idx], color=[self.colors[1], self.colors[0]])
                    axes[idx].set_title(f'{feature} by Churn Status')
                    axes[idx].legend(['Retained', 'Churned'])
        
        # Hide empty subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                save_path: Optional[str] = None):
        """Plot correlation heatmap for numerical features."""
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        correlation = numerical_df.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_churn_by_category(self, df: pd.DataFrame,
                              category_col: str,
                              target_col: str = 'Churn',
                              save_path: Optional[str] = None):
        """Plot churn rate by categorical variable."""
        churn_rate = df.groupby(category_col)[target_col].mean() * 100
        
        plt.figure(figsize=self.figsize)
        churn_rate.plot(kind='bar', color=self.colors[2])
        plt.xlabel(category_col)
        plt.ylabel('Churn Rate (%)')
        plt.title(f'Churn Rate by {category_col}')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(churn_rate.values):
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str],
                              importances: np.ndarray,
                              top_n: int = 15,
                              save_path: Optional[str] = None):
        """Plot feature importance from model."""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=self.figsize)
        plt.bar(range(top_n), importances[indices], color=self.colors[2])
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xticks(range(top_n), [feature_names[i] for i in indices],
                  rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray,
                            save_path: Optional[str] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Retained', 'Churned'],
                   yticklabels=['Retained', 'Churned'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray,
                      auc_score: float,
                      save_path: Optional[str] = None):
        """Plot ROC curve."""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=self.colors[2], lw=2,
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_df: pd.DataFrame,
                            metric: str = 'accuracy',
                            save_path: Optional[str] = None):
        """Compare multiple models performance."""
        plt.figure(figsize=self.figsize)
        
        x = range(len(results_df))
        plt.bar(x, results_df[metric], color=self.colors[:len(results_df)])
        plt.xlabel('Models')
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Comparison - {metric.capitalize()}')
        plt.xticks(x, results_df['model'], rotation=45, ha='right')
        plt.ylim([0, 1])
        
        # Add value labels
        for i, v in enumerate(results_df[metric]):
            plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
