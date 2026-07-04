# Employee Promotion Eligibility Prediction

Supervised machine learning project predicting employee promotion eligibility from HR performance data.

## Overview
Built a classification pipeline on 5,000 employee records (11 features: performance, KPI, attendance, peer rating, manager feedback, department, job role, etc.) to predict promotion eligibility, comparing 5 algorithms.

## Approach
- Exploratory data analysis on numeric and categorical features
- One-hot encoding for categorical variables, standardized scaling for numeric features
- Stratified train/test split to preserve class balance (86% No / 14% Yes)
- Compared Logistic Regression, Decision Tree, Random Forest, KNN, and XGBoost

## Key Finding
Three tree-based models achieved 100% accuracy. Rather than reporting this at face value, I investigated using decision-tree rule extraction and feature ablation testing, confirming the target was governed by a deterministic threshold rule (Performance Score and Manager Feedback exceeding fixed thresholds). Removing these two features collapsed model performance to near-random (ROC-AUC 0.48), confirming no other feature carried predictive signal.

## Final Model
Selected **Logistic Regression** as the production model over the higher-scoring tree-based alternatives, since it reflects a smooth generalizable decision boundary rather than exact rule-memorization:
- Accuracy: 95.7%
- F1 Score: 84.5%
- ROC-AUC: 0.987
- 5-fold cross-validation F1: 0.839 ± 0.026

## Features
- Full EDA with distribution, outlier, and correlation analysis
- 5-model comparison with precision/recall/F1/ROC-AUC
- Feature importance analysis
- Interactive prediction widget (ipywidgets) for instant scoring on new employee inputs

## Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn, ipywidgets

## Files
- `promotion_prediction.ipynb` — full notebook (EDA, modeling, evaluation, widget)
- `summer_project_data_set.csv` — dataset
