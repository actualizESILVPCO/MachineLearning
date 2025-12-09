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

# Installation 
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
# Code Execution

Upload the notebook
Execute all cells sequentially

Important: Ensure the dataset files (train.csv) are available in the expected directory.
Expected Execution Time

Full notebook execution: approximately 15-20 minutes (depending on hardware)
Hyperparameter tuning: 8-10 minutes
Model training: 5-7 minutes
Data preprocessing: 2-3 minutes

# Methodology 
1. Data Preprocessing
Outlier Removal:

Method: 3 × IQR (Interquartile Range)
Threshold: Q3 + 3 × IQR
Outliers removed: 3,431 observations (1.8%)
Justification: Extreme values can distort model training

Feature Encoding:

Categorical features: Label Encoding
Rationale: High cardinality makes one-hot encoding impractical

Feature Scaling:

Method: StandardScaler (mean=0, std=1)
Applied to: All numerical features
Rationale: Ensures equal contribution to distance-based algorithms

Train-Test Split:

Ratio: 80% training, 20% testing
Method: train_test_split with stratification
Random state: 42 (for reproducibility)

Target Transformation:

Method: Logarithmic transformation log(1 + y)
Rationale: Addresses right-skewness (skewness=3.79)
Result: More symmetric distribution, improved model performance

2. Models Implemented
Baseline Models:

Linear Regression (no regularization)
Ridge Regression (L2 regularization, alpha=1.0)
Random Forest (default parameters)

Advanced Models:
4. Ridge Regression with log-transformed target
5. Random Forest with log-transformed target
6. XGBoost with log-transformed target
7. XGBoost (Tuned) with log-transformed target
3. Hyperparameter Tuning
Method: RandomizedSearchCV with 5-fold cross-validation
Random Forest parameter grid:

n_estimators: [100, 200]
max_depth: [4, 8]
min_samples_split: [2, 5]
max_features: ['auto', 'sqrt']

XGBoost parameter grid:

n_estimators: [100, 200]
max_depth: [3, 5]
learning_rate: [0.05, 0.1]

Search strategy:

Number of iterations: 10
Scoring metric: Mean Absolute Error (MAE)
Cross-validation folds: 5

4. Evaluation Metrics
Primary metric: Mean Absolute Error (MAE)

Measures average absolute deviation from true values
Easy to interpret (in original units)
Robust to outliers

Secondary metrics:

Root Mean Squared Error (RMSE): Penalizes large errors more heavily
R² Score: Proportion of variance explained by the model

Implementation Details
Key Code Sections

1. Random Seed Configuration :
pythonSEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

Ensures reproducibility across all stochastic operations.

2. Outlier Removal :
pythonQ1 = df['loss'].quantile(0.25)
Q3 = df['loss'].quantile(0.75)
IQR = Q3 - Q1
outlier_threshold = Q3 + 3*IQR
df = df[df['loss'] <= outlier_threshold]
Removes extreme values using the 3×IQR method.

3. Feature Encoding :
pythonlabel_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
Applies label encoding to all categorical features.

4. Log Transformation :
pythony_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)
Applies log(1+x) transformation to target variable.

5. Hyperparameter Tuning :
pythonparam_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1]
}

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=SEED
)
Searches for optimal hyperparameters using cross-validation.

6. Model Evaluation :
pythony_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
Calculates performance metrics for model comparison.
