# MachineLearning

Allstate Claims Severity Prediction - Machine Learning Project
Authors: Pauline Comvopoulos, Léa Dantec, Eva Carrard
Program: M1 Engineering School - Actuarial Science
Date: December 2025



1. Project Overview
2. Dataset Description
3. Code Structure
4. Installation and Dependencies
5. Code Execution
6. Methodology
7. Implementation Details
8. Results
9. Reproducibility
10. References

# Project Overview
This project implements a complete machine learning pipeline to predict insurance claim severity using the Allstate Claims Severity dataset from Kaggle. The objective is to develop and compare multiple regression models to accurately forecast claim costs based on anonymized features.
Main objectives:

Implement a complete data preprocessing pipeline
Compare baseline and advanced machine learning models
Perform systematic hyperparameter tuning
Ensure reproducibility and code documentation
Analyze model performance and business implications

# Dataset Description
Source: Kaggle - Allstate Claims Severity Competition
Characteristics:

Initial size: 188,318 insurance claims
Features: 130 anonymized variables

116 categorical features
14 numerical features


Target variable: loss (continuous, representing claim cost in USD)
Final dataset: 184,887 observations (after outlier removal)

Statistical properties of target variable (before preprocessing):

StatisticValueCount 188,318
Mean 3,037.34
Standard Deviation 2,904.09
Minimum 0.67
Median 2,115.57
Maximum 121,012.25
Skewness 3.79 
Kurtosis 48.08
The target variable exhibits strong right-skewness, which motivated our use of logarithmic transformation.

# Code Structure
The project is implemented in a single Jupyter notebook
Notebook organization:

Setup and Imports 

Library imports
Random seed configuration
Visualization settings


Data Loading and Exploration 

Dataset loading
Exploratory data analysis
Statistical analysis
Visualization of distributions


Data Preprocessing 

Outlier detection and removal
Feature encoding
Feature scaling
Train-test split


Model Training - Baseline 

Linear Regression
Ridge Regression
Random Forest


Model Training - Log Transformed 

Ridge Regression with log transformation
Random Forest with log transformation
XGBoost with log transformation


Hyperparameter Tuning 

RandomizedSearchCV for Random Forest
RandomizedSearchCV for XGBoost
Cross-validation analysis


Model Evaluation 

Performance metrics calculation
Model comparison
Feature importance analysis
Residual analysis


Visualization and Reporting 

Performance comparison plots
Feature importance plots
Learning curves
