# E-commerce Churn Dataset - Marketing Analytics

## Overview

This is a comprehensive **e-commerce customer churn analysis project** designed for marketing teams to understand customer behavior, predict churn, and develop retention strategies. The dataset contains customer information and behavioral metrics needed to identify at-risk customers and optimize marketing campaigns.

**Key Objective:** Identify factors driving customer churn and develop data-driven marketing strategies to improve customer retention and lifetime value.

---

## Dataset Description

### Dataset File: `E Commerce Dataset.xlsx`

A clean, well-structured dataset of e-commerce customers with:
- **Customer demographics:** Age, gender, location
- **Behavioral metrics:** Purchase frequency, average order value, recency
- **Engagement:** Customer service interactions, product ratings
- **Target Variable:** Churn status (binary: churned vs. retained)

---

## Project Structure

```
E-commerce-Churn-Dataset/
├── E Commerce Dataset.xlsx      # Raw customer data
├── data_loader.py              # Data loading & exploration module
├── requirements.txt             # Python dependencies
├── .github/workflows/           # GitHub Actions CI/CD pipeline
│   └── python-package.yml      # Automated testing workflow
└── README.md                    # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhinavrana3027-ai/E-commerce-Churn-Dataset.git
   cd E-commerce-Churn-Dataset
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Load and explore the data:**
   ```python
   from data_loader import DataLoader
   
   # Initialize data loader
   loader = DataLoader('E Commerce Dataset.xlsx')
   
   # Load data
   df = loader.load_data()
   
   # Get data insights
   info = loader.get_data_info()
   print(f"Total customers: {info['total_customers']}")
   
   # Get churn metrics
   churn_metrics = loader.get_churn_distribution()
   print(f"Churn rate: {churn_metrics['churn_rate']}%")
   ```

---

## Key Marketing Insights

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
   - Personalized marketing campaigns for at-risk segments
   - Loyalty program optimization
   - Cross-sell and upsell opportunities
   - Customer service intervention triggers

---

## Technologies & Libraries

- **Data Processing:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn
- **CI/CD:** GitHub Actions
- **Python Version:** 3.9, 3.10, 3.11

---

## Project Features

✅ **Automated Data Loading:** Clean data import and preprocessing  
✅ **Churn Analysis:** Comprehensive customer retention metrics  
✅ **Marketing Insights:** Actionable segmentation and targeting  
✅ **CI/CD Pipeline:** Automated testing with GitHub Actions  
✅ **Production-Ready:** Modular, well-documented Python code  
✅ **Portfolio-Ready:** Demonstrates marketing analytics expertise  

---

## GitHub Actions Workflow

This project includes automated CI/CD pipeline that:
- Runs on Python 3.9, 3.10, and 3.11
- Installs all dependencies from requirements.txt
- Performs linting checks with flake8
- Validates code quality automatically
- **Status:** ✅ All tests passing

---

## Use Cases

### For Marketing Teams
- **Churn Prevention:** Identify at-risk customers before they leave
- **Segment Marketing:** Target campaigns to high-value segments
- **Retention ROI:** Calculate marketing spend for retention programs
- **Customer Lifetime Value:** Predict CLV for resource allocation

### For Business Analysts
- **Customer Analytics:** Understand behavior patterns
- **Trend Analysis:** Track churn rates over time
- **Performance Metrics:** Monitor retention KPIs
- **Decision Support:** Data-driven business recommendations

---

## Dependencies

All required packages are listed in `requirements.txt`:

```
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## How to Run the Analysis

### Step 1: Load the Data
```python
from data_loader import DataLoader
loader = DataLoader('E Commerce Dataset.xlsx')
df = loader.load_data()
```

### Step 2: Explore Data Statistics
```python
stats = loader.get_summary_statistics()
print(stats)
```

### Step 3: Analyze Churn Distribution
```python
churn_data = loader.get_churn_distribution()
print(f"Active: {churn_data['active_customers']}")
print(f"Churned: {churn_data['churned_customers']}")
print(f"Churn Rate: {churn_data['churn_rate']}%")
```

---

## Project Status

- ✅ Data loader module created
- ✅ Requirements file configured
- ✅ GitHub Actions CI/CD pipeline active
- ✅ All tests passing on Python 3.9, 3.10, 3.11
- 🔄 Marketing analysis modules in development

---

## Contributing

This is a portfolio project. For questions or suggestions about the marketing analytics approach, feel free to open an issue.

---

## Author

**Abhinav Rana** | Data Science & Marketing Analytics Professional  
📍 Berlin, Germany | 🔗 [GitHub](https://github.com/abhinavrana3027-ai)

---

## License

This project is open source and available under the MIT License.

---

## About This Project

This e-commerce churn analysis project demonstrates:
- **Marketing Analytics:** Customer segmentation and retention strategies
- **Data Science Skills:** Python, data processing, and analysis
- **Engineering Excellence:** Modular code, CI/CD automation, professional documentation
- **Portfolio Quality:** Production-ready code suitable for employer evaluation

**Perfect for:** Job applications to marketing analytics, data science, and business intelligence roles in Berlin and across Europe.
