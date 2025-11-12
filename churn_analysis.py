#!/usr/bin/env python3
"""Complete E-commerce Churn Analysis Pipeline

This script demonstrates the complete workflow from data loading
to model training, evaluation, and visualization.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from data_preprocessing import DataPreprocessor
from ml_models import ChurnPredictor
from visualizations import ChurnVisualizer
import pandas as pd
import numpy as np


def main():
    """Execute complete churn analysis pipeline."""
    
    print("="*70)
    print(" E-COMMERCE CUSTOMER CHURN ANALYSIS")
    print("="*70)
    print()
    
    # ========== STEP 1: LOAD DATA ==========
    print("[STEP 1] Loading Data...")
    print("-" * 70)
    
    loader = DataLoader('E Commerce Dataset.xlsx')
    df = loader.load_data()
    
    if df is None:
        print("Error: Could not load data. Exiting...")
        sys.exit(1)
    
    # Display basic info
    info = loader.get_data_info()
    print(f"\nDataset Shape: {info['total_customers']} rows x {info['features']} columns")
    print(f"Duplicate rows: {info['duplicates']}")
    
    # Check churn distribution
    churn_metrics = loader.get_churn_distribution()
    if churn_metrics:
        print(f"\nChurn Distribution:")
        print(f"  Active Customers: {churn_metrics['active_customers']}")
        print(f"  Churned Customers: {churn_metrics['churned_customers']}")
        print(f"  Churn Rate: {churn_metrics['churn_rate']}%")
    
    print("\n" + "✓"*70)
    
    # ========== STEP 2: EXPLORATORY DATA ANALYSIS ==========
    print("\n[STEP 2] Exploratory Data Analysis...")
    print("-" * 70)
    
    visualizer = ChurnVisualizer()
    
    # Plot churn distribution
    print("Creating churn distribution visualization...")
    visualizer.plot_churn_distribution(df)
    
    # Plot correlation heatmap
    print("Creating correlation heatmap...")
    visualizer.plot_correlation_heatmap(df)
    
    print("✓ EDA visualizations created")
    print("\n" + "✓"*70)
    
    # ========== STEP 3: DATA PREPROCESSING ==========
    print("\n[STEP 3] Data Preprocessing...")
    print("-" * 70)
    
    preprocessor = DataPreprocessor()
    
    # Get preprocessing summary
    prep_summary = preprocessor.get_preprocessing_summary(df)
    print(f"\nData Memory Usage: {prep_summary['memory_usage_mb']:.2f} MB")
    print(f"Numerical Features: {len(prep_summary['numerical_features'])}")
    print(f"Categorical Features: {len(prep_summary['categorical_features'])}")
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df)
    
    print("\n" + "✓"*70)
    
    # ========== STEP 4: MODEL TRAINING ==========
    print("\n[STEP 4] Training Machine Learning Models...")
    print("-" * 70)
    
    predictor = ChurnPredictor(random_state=42)
    
    # Train all models
    predictor.train_all_models(X_train, y_train)
    
    print("\n" + "✓"*70)
    
    # ========== STEP 5: MODEL EVALUATION ==========
    print("\n[STEP 5] Evaluating Models...")
    print("-" * 70)
    
    # Evaluate all models
    results_df = predictor.evaluate_all_models(X_test, y_test)
    
    print("\n" + "="*70)
    print(" MODEL COMPARISON RESULTS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Get best model
    best_model_name, best_model_metrics = predictor.get_best_model('f1_score')
    print(f"\n✨ BEST MODEL: {best_model_name}")
    print(f"   F1-Score: {best_model_metrics['f1_score']:.4f}")
    print(f"   Accuracy: {best_model_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC: {best_model_metrics['roc_auc']:.4f}")
    
    print("\n" + "✓"*70)
    
    # ========== STEP 6: VISUALIZE RESULTS ==========
    print("\n[STEP 6] Creating Model Evaluation Visualizations...")
    print("-" * 70)
    
    # Model comparison chart
    print("Creating model comparison chart...")
    visualizer.plot_model_comparison(results_df, metric='f1_score')
    
    # Confusion matrix for best model
    print(f"Creating confusion matrix for {best_model_name}...")
    visualizer.plot_confusion_matrix(best_model_metrics['confusion_matrix'])
    
    # ROC curve for best model
    if best_model_metrics['probabilities'] is not None:
        print(f"Creating ROC curve for {best_model_name}...")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, best_model_metrics['probabilities'])
        visualizer.plot_roc_curve(fpr, tpr, best_model_metrics['roc_auc'])
    
    # Feature importance for tree-based models
    if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
        print(f"Creating feature importance plot for {best_model_name}...")
        feature_importance_df = predictor.get_feature_importance(
            best_model_name, 
            preprocessor.feature_names
        )
        if feature_importance_df is not None:
            print("\nTop 10 Important Features:")
            print(feature_importance_df.head(10).to_string(index=False))
            visualizer.plot_feature_importance(
                preprocessor.feature_names,
                predictor.trained_models[best_model_name].feature_importances_
            )
    
    print("✓ All visualizations created")
    print("\n" + "✓"*70)
    
    # ========== STEP 7: SAVE RESULTS ==========
    print("\n[STEP 7] Saving Results...")
    print("-" * 70)
    
    # Save model comparison results
    predictor.save_results_to_csv(results_df, 'model_comparison_results.csv')
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE!")
    print("="*70)
    print("\nSummary:")
    print(f"  ✓ Dataset loaded: {len(df)} records")
    print(f"  ✓ Models trained: {len(predictor.trained_models)}")
    print(f"  ✓ Best model: {best_model_name}")
    print(f"  ✓ Best F1-Score: {best_model_metrics['f1_score']:.4f}")
    print(f"  ✓ Results saved: model_comparison_results.csv")
    print("\nThis comprehensive analysis demonstrates:")
    print("  • Data loading and exploration")
    print("  • Feature engineering and preprocessing")
    print("  • Multiple ML algorithm implementation")
    print("  • Model evaluation and comparison")
    print("  • Professional visualization")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
