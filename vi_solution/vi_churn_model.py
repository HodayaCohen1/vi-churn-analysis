import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import warnings
from constants import *
warnings.filterwarnings('ignore')

print("=== VI Churn Prediction Model ===\n")

# -----------------------------
# Utils
# -----------------------------
def evaluate_model(name, model, Xtr, ytr, Xte, yte):
    try:
        proba_tr = model.predict_proba(Xtr)[:, 1]
        proba_te = model.predict_proba(Xte)[:, 1]
        print(f"\n{name} - Train AUC: {roc_auc_score(ytr, proba_tr):.4f}")
        print(f"{name} - Test AUC:  {roc_auc_score(yte, proba_te):.4f}")
    except Exception:
        pass
    y_pred = model.predict(Xte)
    print(f"\n{name} - Classification Report:")
    print(classification_report(yte, y_pred, target_names=['No Churn', 'Churn']))
    cm = confusion_matrix(yte, y_pred)
    print(f"{name} - Confusion Matrix:")
    print("                 Predicted")
    print("Actual    No Churn    Churn")
    print(f"No Churn    {cm[0,0]:6d}    {cm[0,1]:5d}")
    print(f"Churn       {cm[1,0]:6d}    {cm[1,1]:5d}")

def print_feature_importances(model, name, cols):
    try:
        if model is None:
            return
        importances = getattr(model, 'feature_importances_', None)
        if importances is None:
            return
        df = pd.DataFrame({'model': name, 'feature': cols, 'importance': importances}).sort_values('importance', ascending=False)
        print(f"\n{name} - Top 10 features:")
        for i, (_, r) in enumerate(df.head(10).iterrows(), 1):
            print(f"{i:2d}. {r['feature']}: {r['importance']:.4f}")
        return df
    except Exception:
        # Silently skip if model does not expose importances or other errors occur
        return None

def hyper_tune(model, X, y, params, label):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=20, scoring='roc_auc', random_state=42, n_jobs=-1, verbose=0)
    search.fit(X, y)
    print(f"{label} best params: {search.best_params_}")
    return search.best_estimator_, search

def cv_auc(model, X, y, label):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
    'roc_auc': 'roc_auc',
    'f1': 'f1'
    }

    cv_res = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,   
        return_estimator=True     
    )

    test_auc_per_fold = cv_res['test_roc_auc']  
    test_f1_per_fold  = cv_res['test_f1']

    train_auc_per_fold = cv_res['train_roc_auc']
    train_f1_per_fold  = cv_res['train_f1']
        
    print(f"{label} CV AUC: {test_auc_per_fold.mean():.4f} (+/- {test_auc_per_fold.std() * 2:.4f}, CV F1: {test_f1_per_fold.mean():.4f} (+/- {test_f1_per_fold.std() * 2:.4f})")
    
    return test_auc_per_fold, test_f1_per_fold, train_auc_per_fold, train_f1_per_fold

# 1. Load processed features
print("1. Loading processed features...")
features = pd.read_csv('vi_churn_analysis_output/processed_features.csv')
print(f"Features shape: {features.shape}")

# 2. Prepare data for modeling
print("\n2. Preparing data for modeling...")

# Select features for modeling (exclude metadata and non-numeric columns)
drop_cols = ['member_id', 'signup_date', 'churn', 'outreach']
X_all = features.drop(columns=[c for c in drop_cols if c in features.columns])
X = X_all.select_dtypes(include=[np.number])
y = features['churn']
feature_columns = list(X.columns)

print(f"Feature columns: {len(feature_columns)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 3. Split data
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training churn rate: {y_train.mean():.3f}")
print(f"Test churn rate: {y_test.mean():.3f}")

# 4. Train & evaluate models
print("\n4. Training models - getting the best configuration...")

# 4.1 Random Forest (regularized + quick hyperparameter search)
print("4.1 Training Random Forest...")
models = {}
rf_base = RandomForestClassifier(**RF_BASE_PARAMS)
param_dist = RF_PARAM_DIST

rf_model, rf_search = hyper_tune(rf_base, X_train, y_train, param_dist, 'Random Forest')

rf_train_pred = rf_model.predict_proba(X_train)[:, 1]
rf_test_pred = rf_model.predict_proba(X_test)[:, 1]

rf_train_auc = roc_auc_score(y_train, rf_train_pred)
rf_test_auc = roc_auc_score(y_test, rf_test_pred)

# print(f"Random Forest - Train AUC: {rf_train_auc:.4f}")
# print(f"Random Forest - Test AUC: {rf_test_auc:.4f}")
models['Random Forest'] = rf_model

# 4.2 Logistic Regression (with scaling)
print("4.2 Training Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(**LR_PARAMS)
lr_model.fit(X_train_scaled, y_train)

lr_train_pred = lr_model.predict_proba(X_train_scaled)[:, 1]
lr_test_pred = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_train_auc = roc_auc_score(y_train, lr_train_pred)
lr_test_auc = roc_auc_score(y_test, lr_test_pred)

# print(f"Logistic Regression - Train AUC: {lr_train_auc:.4f}")
# print(f"Logistic Regression - Test AUC: {lr_test_auc:.4f}")
models['Logistic Regression'] = lr_model

# 4.3 Gradient Boosting (sklearn) with early stopping
print("\n4.3 Gradient Boosting (sklearn) with early stopping...")
gb_model = GradientBoostingClassifier(**GB_PARAMS)
gb_model.fit(X_train, y_train)

gb_train_pred = gb_model.predict_proba(X_train)[:, 1]
gb_test_pred = gb_model.predict_proba(X_test)[:, 1]

gb_train_auc = roc_auc_score(y_train, gb_train_pred)
gb_test_auc = roc_auc_score(y_test, gb_test_pred)

# print(f"Gradient Boosting - Train AUC: {gb_train_auc:.4f}")
# print(f"Gradient Boosting - Test AUC: {gb_test_auc:.4f}")
models['Gradient Boosting'] = gb_model

# 4.4 XGBoost with hyperparameter tuning (if available)
if xgb is not None:
    print("\n4.4 XGBoost with hyperparameter tuning...")
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0
    
    xgb_base = xgb.XGBClassifier(
        n_estimators=1000,
        tree_method='hist',
        eval_metric='auc',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
    )
    
    param_dist = XGB_PARAM_DIST

    xgb_model, xgb_search = hyper_tune(xgb_base, X_train, y_train, param_dist, 'XGBoost')
    
    xgb_train_pred = xgb_model.predict_proba(X_train)[:, 1]
    xgb_test_pred = xgb_model.predict_proba(X_test)[:, 1]
    xgb_train_auc = roc_auc_score(y_train, xgb_train_pred)
    xgb_test_auc = roc_auc_score(y_test, xgb_test_pred)
    # print(f"XGBoost - Train AUC: {xgb_train_auc:.4f}")
    # print(f"XGBoost - Test AUC: {xgb_test_auc:.4f}")
    models['XGBoost'] = xgb_model
    
else:
    print("\n4.4 XGBoost not installed - skipping (pip install xgboost)")

# 5. Cross-validation
print("\n5. Cross-validation AUC and F1 scores...")

# Store CV results for later use
cv_results = {}

# Random Forest CV
rf_cv_auc, rf_cv_f1, rf_cv_train_auc, rf_cv_train_f1 = cv_auc(rf_model, X, y, 'Random Forest')
cv_results['Random Forest'] = {
    'test_auc_mean': rf_cv_auc.mean(),
    'test_auc_std': rf_cv_auc.std(),
    'test_f1_mean': rf_cv_f1.mean(),
    'test_f1_std': rf_cv_f1.std(),
    'train_auc_mean': rf_cv_train_auc.mean(),
    'train_auc_std': rf_cv_train_auc.std(),
    'train_f1_mean': rf_cv_train_f1.mean(),
    'train_f1_std': rf_cv_train_f1.std()
}

# Logistic Regression CV
lr_cv_auc, lr_cv_f1, lr_cv_train_auc, lr_cv_train_f1 = cv_auc(make_pipeline(StandardScaler(), LogisticRegression(**LR_PARAMS)), X, y, 'Logistic Regression')
cv_results['Logistic Regression'] = {
    'test_auc_mean': lr_cv_auc.mean(),
    'test_auc_std': lr_cv_auc.std(),
    'test_f1_mean': lr_cv_f1.mean(),
    'test_f1_std': lr_cv_f1.std(),
    'train_auc_mean': lr_cv_train_auc.mean(),
    'train_auc_std': lr_cv_train_auc.std(),
    'train_f1_mean': lr_cv_train_f1.mean(),
    'train_f1_std': lr_cv_train_f1.std()
}

# Gradient Boosting CV
gb_cv_auc, gb_cv_f1, gb_cv_train_auc, gb_cv_train_f1 = cv_auc(GradientBoostingClassifier(**GB_PARAMS), X, y, 'Gradient Boosting')
cv_results['Gradient Boosting'] = {
    'test_auc_mean': gb_cv_auc.mean(),
    'test_auc_std': gb_cv_auc.std(),
    'test_f1_mean': gb_cv_f1.mean(),
    'test_f1_std': gb_cv_f1.std(),
    'train_auc_mean': gb_cv_train_auc.mean(),
    'train_auc_std': gb_cv_train_auc.std(),
    'train_f1_mean': gb_cv_train_f1.mean(),
    'train_f1_std': gb_cv_train_f1.std()
}

# XGBoost CV (if available)
if 'xgb' in globals():
    xgb_cv_auc, xgb_cv_f1, xgb_cv_train_auc, xgb_cv_train_f1 = cv_auc(xgb_model, X, y, 'XGBoost')
    cv_results['XGBoost'] = {
        'test_auc_mean': xgb_cv_auc.mean(),
        'test_auc_std': xgb_cv_auc.std(),
        'test_f1_mean': xgb_cv_f1.mean(),
        'test_f1_std': xgb_cv_f1.std(),
        'train_auc_mean': xgb_cv_train_auc.mean(),
        'train_auc_std': xgb_cv_train_auc.std(),
        'train_f1_mean': xgb_cv_train_f1.mean(),
        'train_f1_std': xgb_cv_train_f1.std()
    }

# 6. Evaluate models
print("\n6. Evaluating models...")
evaluate_model('Random Forest', rf_model, X_train, y_train, X_test, y_test)
evaluate_model('Logistic Regression', lr_model, X_train_scaled, y_train, X_test_scaled, y_test)
evaluate_model('Gradient Boosting', gb_model, X_train, y_train, X_test, y_test)
evaluate_model('XGBoost', 'xgb_model' in globals() and xgb_model or None, X_train, y_train, X_test, y_test)

# 7. Feature importance (all models)
print("\n7. Feature Importance (all models):")

# Collect all feature importances
all_importances = []

# Random Forest
rf_importance = pd.DataFrame({
    'model': 'Random Forest',
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
all_importances.append(rf_importance)

print("Random Forest - Top 10 features:")
for i, (_, row) in enumerate(rf_importance.head(10).iterrows(), 1):
    print(f"{i:2d}. {row['feature']}: {row['importance']:.4f}")

# Gradient Boosting
print_feature_importances(gb_model, 'Gradient Boosting', feature_columns)
if hasattr(gb_model, 'feature_importances_'):
    gb_importance = pd.DataFrame({
        'model': 'Gradient Boosting',
        'feature': feature_columns,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    all_importances.append(gb_importance)

# XGBoost
print_feature_importances('xgb_model' in globals() and xgb_model or None, 'XGBoost', feature_columns)
if 'xgb_model' in globals() and hasattr(xgb_model, 'feature_importances_'):
    xgb_importance = pd.DataFrame({
        'model': 'XGBoost',
        'feature': feature_columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    all_importances.append(xgb_importance)

# 8. Model comparison with baseline
print("\n8. Comparison with Baseline:")

# Load baseline confusion matrix from classification report
def load_baseline_confusion_matrix():
    """Calculate baseline confusion matrix from classification report"""
    try:
        with open('data/classification_report_baseline.txt', 'r') as f:
            lines = f.readlines()
        
        # Extract values from classification report
        no_churn_line = lines[2].strip().split()
        churn_line = lines[3].strip().split()
        
        no_churn_support = int(no_churn_line[4])  # 2653 (5th column)
        churn_support = int(churn_line[4])        # 680 (5th column)
        no_churn_recall = float(no_churn_line[2])  # 0.50 (3rd column)
        churn_recall = float(churn_line[2])        # 0.51 (3rd column)
        
        # Calculate confusion matrix values
        # True Negatives (no_churn correctly predicted)
        tn = int(no_churn_support * no_churn_recall)  # 2653 * 0.50 = 1326
        # False Positives (no_churn predicted as churn)
        fp = no_churn_support - tn  # 2653 - 1326 = 1327
        # False Negatives (churn predicted as no_churn)
        fn = int(churn_support * (1 - churn_recall))  # 680 * 0.49 = 333
        # True Positives (churn correctly predicted)
        tp = churn_support - fn  # 680 - 333 = 347
        
        return tp, tn, fp, fn
        
    except Exception as e:
        print(f"Warning: Could not load baseline confusion matrix: {e}")
        return 347, 1326, 1327, 333  # Default values

# Load baseline confusion matrix
baseline_tp, baseline_tn, baseline_fp, baseline_fn = load_baseline_confusion_matrix()

baseline_auc = BASELINE_AUC
baseline_f1 = BASELINE_F1

print(f"Baseline AUC: {baseline_auc:.6f}")
print(f"Baseline F1: {baseline_f1:.3f}")

print(f"\nBaseline Confusion Matrix:")
print(f"                 Predicted")
print(f"Actual    No Churn    Churn")
print(f"No Churn      {baseline_tn:4d}     {baseline_fp:4d}")
print(f"Churn         {baseline_fn:4d}     {baseline_tp:4d}")

# Find best model
best_model = max(cv_results, key=lambda x: cv_results[x]['test_auc_mean']*cv_results[x]['test_f1_mean'])
best_model_auc = cv_results[best_model]['test_auc_mean']
best_model_f1 = cv_results[best_model]['test_f1_mean']

print(f"\n🏆 BEST MODEL: {best_model}")
print(f"   CV AUC: {best_model_auc:.4f}")
print(f"   CV F1:  {best_model_f1:.4f}")

print(f"\nModel Improvements over Baseline:")
print(f"Best model AUC improvement: {best_model_auc - baseline_auc:.6f}")
print(f"Best model F1 improvement:  {best_model_f1 - baseline_f1:.3f}")
print(f"AUC improvement percentage: {((best_model_auc - baseline_auc) / baseline_auc * 100):.1f}%")
print(f"F1 improvement percentage:  {((best_model_f1 - baseline_f1) / baseline_f1 * 100):.1f}%")

# 9. Create results directory and save all outputs
print("\n9. Creating results directory and saving all outputs...")
import joblib
import os

# Create results directory structure
os.makedirs('vi_model_output', exist_ok=True)
os.makedirs('vi_model_output/models', exist_ok=True)

# Save predictions
predictions = models[best_model].predict(X)
predict_proba = models[best_model].predict_proba(X).round(3)
predictions_df = pd.DataFrame({'member_id': features['member_id'], 'churn_prediction': predictions, 'churn_prediction_proba': predict_proba[:, 1]})
predictions_df.to_csv('vi_model_output/predictions.csv', index=False)
print("Predictions saved to 'vi_model_output/predictions.csv'")

# Save models
for model_name, model in models.items():
    joblib.dump(model, f'vi_model_output/models/{model_name}.pkl')
    print(f"Model {model_name} saved to 'vi_model_output/models/{model_name}.pkl'")

# Save feature importances
if all_importances:
    combined_importances = pd.concat(all_importances, ignore_index=True)
    combined_importances.to_csv('vi_model_output/feature_importance.csv', index=False)
    print("Feature importances saved to 'vi_model_output/feature_importance.csv'")
else:
    print("No feature importances available to save")

# Save model performance summary
performance_summary = {
    'best_model': best_model,
    'cv_auc': best_model_auc,
    'cv_f1': best_model_f1,
    'baseline_auc': baseline_auc,
    'baseline_f1': baseline_f1,
    'auc_improvement': best_model_auc - baseline_auc,
    'f1_improvement': best_model_f1 - baseline_f1,
    'auc_improvement_pct': ((best_model_auc - baseline_auc) / baseline_auc * 100),
    'f1_improvement_pct': ((best_model_f1 - baseline_f1) / baseline_f1 * 100)
}
performance_df = pd.DataFrame([performance_summary])
performance_df.to_csv('vi_model_output/model_performance_summary.csv', index=False)
print("Model performance summary saved to 'vi_model_output/model_performance_summary.csv'")

# Save CV results
cv_results_df = pd.DataFrame(cv_results).T.reset_index()
cv_results_df.columns = ['model'] + list(cv_results_df.columns[1:])
cv_results_df.to_csv('vi_model_output/cv_results.csv', index=False)
print("CV results saved to 'vi_model_output/cv_results.csv'")

print(f"\n✅ All results saved to 'vi_model_output/' folder!")
print(f"📁 Results folder structure:")
print(f"   ├── predictions.csv")
print(f"   ├── feature_importance.csv")
print(f"   ├── model_performance_summary.csv")
print(f"   ├── cv_results.csv")
print(f"   └── models/")
print(f"       ├── Random Forest.pkl")
print(f"       ├── Logistic Regression.pkl")
print(f"       ├── Gradient Boosting.pkl")
print(f"       └── XGBoost.pkl")

print("\n=== Model Training Complete ===")
print("See per-model AUCs, reports, and confusion matrices above.")
