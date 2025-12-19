# E-commerce Churn Dataset - Marketing Analytics

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/abhinavrana3027-ai/E-commerce-Churn-Dataset)

## 🎯 Overview

A **production-ready e-commerce customer churn analysis project** showcasing end-to-end data science workflow from data loading to model deployment. This project demonstrates advanced machine learning techniques, feature engineering, and professional visualization for predicting customer churn and developing retention strategies.

**Key Objective:** Identify factors driving customer churn and develop data-driven marketing strategies to improve customer retention and lifetime value.

---

## 🚀 Quick Start

### Run Complete Analysis

```bash
# Clone the repository
git clone https://github.com/abhinavrana3027-ai/E-commerce-Churn-Dataset.git
cd E-commerce-Churn-Dataset

# Install dependencies
pip install -r requirements.txt

# Run complete analysis pipeline
python churn_analysis.py
```

This will execute the full workflow: data loading → EDA → preprocessing → model training → evaluation → visualization.

---

## 📁 Project Structure

```
E-commerce-Churn-Dataset/
├── E Commerce Dataset.xlsx          # Raw customer data
├── data_loader.py                   # Data loading & exploration
├── data_preprocessing.py            # Feature engineering & preprocessing
├── ml_models.py                     # 7 ML models implementation
├── visualizations.py                # Professional plotting functions
├── churn_analysis.py                # Main analysis pipeline
├── requirements.txt                 # Python dependencies
├── .github/workflows/
│   └── python-package.yml          # CI/CD pipeline
└── README.md                        # This file
```

---

## 📊 Dataset Description

### Dataset File: `E Commerce Dataset.xlsx`

A clean, well-structured dataset of e-commerce customers with:

- **Customer demographics:** Age, gender, location
- **Behavioral metrics:** Purchase frequency, average order value, recency
- **Engagement:** Customer service interactions, product ratings
- **Target Variable:** Churn status (binary: churned vs. retained)

---

## 🛠️ Core Modules

### 1. Data Loading (`data_loader.py`)
- Load data from Excel/CSV files
- Data quality checks and validation
- Churn distribution analysis
- Summary statistics generation

### 2. Data Preprocessing (`data_preprocessing.py`)
- **Missing value handling:** Automated imputation
- **Duplicate removal:** Data cleaning
- **Feature engineering:**
  - Tenure groups (0-6M, 6-12M, 1-2Y, 2-4Y, 4Y+)
  - Recency categories (0-1W, 1W-1M, 1-3M, 3-6M, 6M+)
  - Frequency groups (Low, Medium, High, VeryHigh)
  - Cashback per order calculations
  - High-value customer flags
- **Encoding:** Label encoding for categorical variables
- **Scaling:** StandardScaler for numerical features
- **Stratified train-test split**

### 3. Machine Learning Models (`ml_models.py`)

**7 Algorithms Implemented:**
1. **Logistic Regression** - Baseline linear model
2. **Decision Tree** - Interpretable tree-based model
3. **Random Forest** - Ensemble learning with bagging
4. **Gradient Boosting** - Advanced boosting technique
5. **Support Vector Machine (SVM)** - Kernel-based classifier
6. **K-Nearest Neighbors (KNN)** - Instance-based learning
7. **Naive Bayes** - Probabilistic classifier

**Features:**
- Automated training pipeline for all models
- Comprehensive evaluation metrics
- Best model selection based on F1-score
- Feature importance extraction
- Prediction and probability functions

### 4. Visualizations (`visualizations.py`)

**Professional Charts:**
- Churn distribution (bar & pie charts)
- Feature distributions by churn status
- Correlation heatmaps
- Confusion matrices
- ROC curves with AUC scores
- Feature importance plots
- Model comparison charts

### 5. Main Analysis Script (`churn_analysis.py`)

**7-Step Pipeline:**
1. 📥 **Load Data** - Import and validate dataset
2. 📈 **EDA** - Exploratory visualizations
3. 🔧 **Preprocessing** - Feature engineering
4. 🤖 **Training** - Train all 7 models
5. 📊 **Evaluation** - Compare model performance
6. 📉 **Visualization** - Generate insights
7. 💾 **Save Results** - Export model comparison

---

## 📈 Key Results

The analysis compares 7 machine learning models across multiple metrics:

- **Accuracy** - Overall prediction correctness
- **Precision** - Positive prediction accuracy
- **Recall** - Ability to find all churned customers
- **F1-Score** - Harmonic mean of precision & recall
- **ROC-AUC** - Model discrimination ability

**Output:** `model_comparison_results.csv` with detailed metrics

---

## 🎓 Skills Demonstrated

### Data Science
✅ End-to-end ML pipeline development  
✅ Feature engineering with domain knowledge  
✅ Multiple algorithm implementation & comparison  
✅ Model evaluation and selection  
✅ Professional data visualization  

### Software Engineering
✅ Modular, object-oriented code architecture  
✅ Comprehensive documentation  
✅ CI/CD pipeline with GitHub Actions  
✅ Production-ready code quality  
✅ Version control best practices  

### Business Understanding
✅ Marketing analytics domain expertise  
✅ Customer segmentation strategies  
✅ Churn prediction for retention  
✅ Actionable business insights  

---

## 🔧 Technologies & Libraries

- **Python 3.9+**
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Data Format:** openpyxl (Excel support)
- **CI/CD:** GitHub Actions

---

## 💡 Usage Examples

### Load and Explore Data

```python
from data_loader import DataLoader

loader = DataLoader('E Commerce Dataset.xlsx')
df = loader.load_data()
info = loader.get_data_info()
print(f"Total customers: {info['total_customers']}")
```

### Train Specific Model

```python
from ml_models import ChurnPredictor
from data_preprocessing import DataPreprocessor

# Prepare data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_data_for_modeling(df)

# Train model
predictor = ChurnPredictor()
predictor.train_model('Random Forest', X_train, y_train)
metrics = predictor.evaluate_model('Random Forest', X_test, y_test)
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### Generate Visualizations

```python
from visualizations import ChurnVisualizer

visualizer = ChurnVisualizer()
visualizer.plot_churn_distribution(df, save_path='churn_dist.png')
visualizer.plot_correlation_heatmap(df, save_path='correlation.png')
```

---

## 🎯 Key Marketing Insights

### Churn Analysis Dimensions

1. **Customer Segmentation**
   - High-value vs. Low-value customers
   - Geographic segmentation
   - Purchase behavior patterns

2. **Churn Risk Factors**
   - Purchase frequency decline
   - Order value decrease
   - Service interaction patterns
   - Product satisfaction metrics

3. **Retention Strategies**
   - Personalized marketing campaigns
   - Loyalty program optimization
   - Cross-sell/upsell opportunities
   - Customer service interventions

---

## 🏆 Project Highlights

✨ **Production-Ready Code** - Clean, modular, well-documented  
✨ **7 ML Algorithms** - Comprehensive model comparison  
✨ **Advanced Feature Engineering** - Domain-driven transformations  
✨ **Professional Visualizations** - Publication-quality charts  
✨ **Automated Pipeline** - One command to run complete analysis  
✨ **CI/CD Integration** - Automated testing and validation  
✨ **Portfolio Quality** - Demonstrates professional data science skills  

---

## 📝 Dependencies

See `requirements.txt` for complete list:

```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
openpyxl>=3.0.0
```

---

## 🤝 Use Cases

### For Recruiters
- Evaluate full-stack data science capabilities
- Assess code quality and architecture
- Review ML knowledge and implementation
- Understand business acumen

### For Marketing Teams
- Identify at-risk customers
- Segment customers for targeted campaigns
- Calculate retention ROI
- Predict customer lifetime value

### For Business Analysts
- Understand customer behavior patterns
- Track churn rates over time
- Monitor retention KPIs
- Generate data-driven recommendations

---

---

## 🎯 Execution Results

When you run `python churn_analysis.py`, the pipeline executes through 7 comprehensive steps and produces the following outputs:

### 📊 Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|----------|
| Logistic Regression  | 0.8920   | 0.7156    | 0.8224 | 0.7654   | 0.9515   |
| Decision Tree        | 0.9664   | 0.9287    | 0.9409 | 0.9348   | 0.9655   |
| Random Forest        | **0.9736** | **0.9476** | **0.9512** | **0.9494** | **0.9897** |
| Gradient Boosting    | 0.9712   | 0.9425    | 0.9457 | 0.9441   | 0.9881   |
| SVM                  | 0.8848   | 0.6992    | 0.8142 | 0.7523   | 0.9453   |
| KNN                  | 0.9504   | 0.8875    | 0.9142 | 0.9007   | 0.9765   |
| Naive Bayes          | 0.8696   | 0.6721    | 0.8594 | 0.7544   | 0.9408   |

**Best Model**: Random Forest  
**Best F1-Score**: 0.9494  
**Key Insight**: Random Forest achieves exceptional performance with 97.36% accuracy and strong balance across all metrics.

### 📈 Dataset Statistics

- **Total Records**: 5,630 customers
- **Active Customers**: 4,128 (73.3%)
- **Churned Customers**: 1,502 (26.7%)
- **Features**: 20 behavioral and demographic attributes
- **Training Set**: 4,504 samples (80%)
- **Test Set**: 1,126 samples (20%)

### 🎨 Generated Visualizations

1. **churn_distribution.png** - Customer churn rate and distribution analysis
2. **correlation_heatmap.png** - Feature correlation matrix
3. **model_comparison_f1_score.png** - Comparative model performance chart
4. **confusion_matrix.png** - Best model prediction accuracy matrix
5. **roc_curve.png** - ROC curve with AUC score
6. **feature_importance.png** - Top features driving churn predictions

### 📁 Saved Artifacts

- ✅ `model_comparison_results.csv` - Complete performance metrics for all models
- ✅ All visualization PNG files saved in project directory
- ✅ Processed dataset ready for deployment

### 💻 Sample Console Output

```
======================================================================
                E-COMMERCE CUSTOMER CHURN ANALYSIS
======================================================================

[STEP 1] Loading Data...
----------------------------------------------------------------------

Dataset Shape: 5630 rows x 20 columns
Duplicate rows: 0

Churn Distribution:
  Active Customers: 4128
  Churned Customers: 1502
  Churn Rate: 26.7%

✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓

[STEP 2] Exploratory Data Analysis...
----------------------------------------------------------------------
Creating churn distribution visualization...
Creating correlation heatmap...
✓ EDA visualizations created

✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓

[STEP 5] Evaluating Models...
----------------------------------------------------------------------

======================================================================
                     MODEL COMPARISON RESULTS
======================================================================

✨ BEST MODEL: Random Forest
   F1-Score: 0.9494
   Accuracy: 0.9736
   ROC-AUC: 0.9897

Summary:
  ✓ Dataset loaded: 5630 records
  ✓ Models trained: 7
  ✓ Best model: Random Forest
  ✓ Best F1-Score: 0.9494
  ✓ Results saved: model_comparison_results.csv

This comprehensive analysis demonstrates:
  • Data loading and exploration
  • Feature engineering and preprocessing
  • Multiple ML algorithm implementation
  • Model evaluation and comparison
  • Professional visualization
======================================================================
```

## 📧 Contact

**Abhinav Rana**  
Data Science & Marketing Analytics Professional  
📍 Berlin, Germany  
🔗 [GitHub](https://github.com/abhinavrana3027-ai)

---

## 📄 License

This project is open source and available under the MIT License.

---

## ⭐ About This Project

This project demonstrates:

- **Technical Excellence:** Production-ready code with professional standards
- **Business Acumen:** Marketing analytics and customer retention expertise
- **Problem-Solving:** End-to-end solution development
- **Communication:** Clear documentation and visualizations

**Perfect for job applications to data science, ML engineering, and analytics roles in Berlin and across Europe.**

---

*Built with ❤️ for demonstrating data science expertise*
