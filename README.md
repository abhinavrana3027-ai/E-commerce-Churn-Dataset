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
