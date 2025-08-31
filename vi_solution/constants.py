# This file contains all hyperparameter configurations for model training

# Random Forest Hyperparameters
RF_PARAM_DIST = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3,4,5,6, 7],
    'min_samples_split': [5, 10, 11, 12, 13],
    'min_samples_leaf': [2, 3, 4, 5, 6],
    'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0]
}

RF_BASE_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'bootstrap': True,
    'oob_score': True,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
}

# Logistic Regression Hyperparameters
LR_PARAMS = {
    'random_state': 42,
    'max_iter': 1000
}

# Gradient Boosting Hyperparameters
GB_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 3,
    'subsample': 0.8,
    'validation_fraction': 0.2,
    'n_iter_no_change': 25,
    'random_state': 42
}

# XGBoost Hyperparameters
XGB_PARAM_DIST = {
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05, 0.1],
    'max_depth': [5, 6, 7, 8, 9],
    'subsample': [ 0.8, 0.9, 0.95, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_lambda': [0.1, 0.5, 1.0, 2.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'min_child_weight': [1, 3, 5, 7, 8, 9],
    'gamma': [0, 0.1, 0.2, 0.25, 0.3, 0.4]
}

# Baseline AUC for comparison
BASELINE_AUC = 0.501
BASELINE_F1 = 0.29

# Outreach Optimization Parameters
OUTREACH_PARAMS = {
    'R_min': 0.65,      # Minimum recall requirement
    'L_min': 1.50,      # Minimum lift requirement (quality guard)
    'K_cap': 6000       # Maximum capacity (optional)
}

# Business Cost Parameters for ROI Analysis
BUSINESS_PARAMS = {
    'outreach_cost_per_member': 2,  # Cost per outreach ($)
    'churn_cost_per_member': 15     # Cost of churn ($)
}
