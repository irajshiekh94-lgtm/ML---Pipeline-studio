ML Pipeline Studio

A Streamlit-based Machine Learning Pipeline Builder that lets you upload your dataset, explore it visually, and build end-to-end ML pipelines — all without writing code.

This app follows production-grade best practices for data preprocessing, model selection, and evaluation, powered by Scikit-learn, Pandas, and Seaborn.

Features
Automated EDA

Detects data types (numeric, categorical, datetime)

Generates descriptive statistics

Visualizes missing values, outliers, skewness, and correlations

Interactive histogram and distribution plots

Preprocessing Pipeline

Missing value imputation (mean, median, mode)

Feature scaling (Standard, MinMax, Robust)

Skewness correction with PowerTransformer

One-Hot Encoding for categorical features

Outlier analysis and multicollinearity detection

Model Training

Supports both Classification and Regression

Built-in models:

Random Forest

Logistic Regression

SVM

Ridge Regression

Optional GridSearchCV for hyperparameter tuning

Cross-validation support

Model Evaluation

Auto-generated performance reports:

Accuracy, F1-Score, RMSE, R²

Confusion matrix (for classification)

Highlighted insights and warnings for skewed data or overfitting

Exportable trained pipelines via joblib

Tech Stack
Component	Technology
Frontend UI	Streamlit
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn
Model Storage	Joblib
