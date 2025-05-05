# Predictive Customer Churn Analysis with SHAP Explainability

## Project Overview

This end-to-end machine learning project predicts customer churn (when customers stop using a service) while providing transparent explanations for each prediction using SHAP (SHapley Additive exPlanations) values. The solution helps businesses:

- Identify at-risk customers before they churn
- Understand key factors driving churn decisions
- Develop targeted retention strategies
- Improve customer experience based on data-driven insights

## Technical Implementation

### Data Processing Pipeline

1. **Data Loading & Cleaning**:
   - Handles missing values in TotalCharges
   - Converts data types appropriately
   - Drops irrelevant columns (customerID)

2. **Feature Engineering**:
   - Creates meaningful ratios (TenureToMonthlyChargeRatio)
   - Combines features (HasDependentsAndPartner)
   - Proper encoding of categorical variables

3. **Preprocessing**:
   - Numeric features: Median imputation + Standard scaling
   - Categorical features: One-hot encoding
   - Feature selection using ANOVA F-value

### Machine Learning Modeling

1. **Model Comparison**:
   - Evaluates Logistic Regression, Random Forest, and Gradient Boosting
   - Uses stratified k-fold cross-validation
   - Tracks multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

2. **Hyperparameter Optimization**:
   - Implements RandomizedSearchCV for efficient tuning
   - Focuses on Gradient Boosting with early stopping
   - Tracks training time vs. performance tradeoffs

3. **Model Interpretation**:
   - SHAP values for global and local explanations
   - Multiple visualization techniques:
     - Summary plots
     - Force plots
     - Dependence plots
     - Waterfall plots
     - Decision plots

## Key Features

1. **Business-Ready Insights**:
   - Clear identification of top churn drivers
   - Customer-specific risk profiles
   - Actionable retention recommendations

2. **Technical Sophistication**:
   - Robust handling of different SHAP output formats
   - Comprehensive error handling
   - Memory-efficient implementation
   - Parallel processing support

3. **Visualization Suite**:
   - Interactive JavaScript plots
   - Static matplotlib visualizations
   - Automatic fallback to simpler plots when needed

## Getting Started

### Prerequisites
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, shap
- (Optional) Jupyter Notebook for interactive exploration

### Installation
```bash
pip install -r requirements.txt
