# Vi Churn Analysis Solution

This folder contains the complete solution for the Vi churn analysis project.

## Approach Overview

### Problem Statement
Predict customer churn and optimize outreach campaigns to reduce churn while maximizing ROI.

### Methodology
**1. Feature Engineering**
- **App Usage Features:** Session patterns, frequency, time-of-day analysis, activity gaps
- **Claims Features:** Medical condition flags (diabetes, hypertension, dietary counseling)
- **Web Visit Features:** Health taxonomy (diabetes, cardio, nutrition, exercise, etc.), engagement ratios
- **Temporal Features:** Recency, frequency, days since signup, activity patterns

**2. Model Development**
- **Ensemble Approach:** Random Forest, Gradient Boosting, XGBoost, Logistic Regression
- **Cross-Validation:** Stratified 5-fold CV for robust evaluation
- **Hyperparameter Tuning:** Randomized search for optimal model performance
- **Baseline Comparison:** Evaluate against provided baseline scores

**3. Outreach Optimization**
- **Two-Strategy Approach:**
  - **F2 Optimization:** Maximize recall-focused metric (F2 score)
  - **Structured Selection:** Balance recall requirements (≥65%) with quality guards (lift ≥1.5x)
- **ROI Analysis:** Calculate net savings considering outreach costs vs churn prevention
- **Business Recommendations:** Optimal number of customers to contact with clear business impact

## Project Structure

```
vi_solution/
├── vi_churn_analysis.py          # Feature engineering and data processing
├── vi_churn_model.py             # Model optimization, training and evaluation
├── vi_outreach_optimization.py   # Outreach campaign optimization
├── constants.py                  # Configuration parameters
├── vi_churn_analysis_output/     # Output from feature engineering
├── vi_model_output/              # Output from model training
├── vi_outreach_output/           # Output from outreach optimization
└── README.md                     # This file
```

## How to Run the Analysis

### Step 1: Feature Engineering
```bash
cd vi_solution
python vi_churn_analysis.py
```
**Output:** 
- `vi_churn_analysis_output/processed_features.csv`
- `vi_churn_analysis_output/kpi_panel.png`
- `vi_churn_analysis_output/churn_comparison.png`

### Step 2: Model Training
```bash
python vi_churn_model.py
```
**Output:** 
- `vi_model_output/predictions.csv`
- `vi_model_output/models/` (trained models)
- `vi_model_output/feature_importance.csv`
- `vi_model_output/model_performance_summary.csv`
- `vi_model_output/cv_results.csv`

### Step 3: Outreach Optimization
```bash
python vi_outreach_optimization.py
```
**Output:**
- `vi_outreach_output/outreach_optimization_analysis.png`
- `vi_outreach_output/f2_optimization_results.csv`
- `vi_outreach_output/quality_threshold_results.csv`
- `vi_outreach_output/final_recommended_outreach_list.csv` (includes rank for assignment)
- `vi_outreach_output/quality_lift_*x_outreach_list.csv`

## Key Results

### Model Performance
- **Best model:** Random Forest
- **CV AUC:** 0.6488
- **CV F1:** 0.3843
- **AUC improvement:** 29.5% over baseline
- **F1 improvement:** 32.5% over baseline

### Outreach Optimization
- **Recommended Approach:** Structured Selection
- **Optimal customers to contact:** 4,200
- **ROI:** 144%
- **Recall:** 67.6%
- **Net savings:** $12,105
- **Cost per churner prevented:** $15

## Configuration

Edit `constants.py` to adjust:
- Model hyperparameters
- Outreach optimization parameters
- Business cost assumptions
- Baseline scores and thresholds

## Visualization Outputs

The analysis generates presentation-ready visualizations:
- **KPI Panel:** Population, churn rate, and outreach rate metrics
- **Churn Comparison:** Effectiveness of outreach campaigns
- **Outreach Optimization:** ROI analysis and recommended contact lists

## Dependencies

### Option 1: Install using requirements.txt (Recommended)
```bash
cd vi_solution
pip install -r requirements.txt
```

### Option 2: Install packages individually
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

### Note for macOS users
If you encounter issues with XGBoost, you may need to install OpenMP:
```bash
brew install libomp
```
