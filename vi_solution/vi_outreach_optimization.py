import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from constants import OUTREACH_PARAMS, BUSINESS_PARAMS
import warnings
warnings.filterwarnings('ignore')

print("=== Outreach Optimization Analysis ===\n")

# Load predictions and actual data
print("1. Loading data...")
predictions_df = pd.read_csv('vi_model_output/predictions.csv')
features_df = pd.read_csv('vi_churn_analysis_output/processed_features.csv')

# Merge predictions with actual churn labels
data = features_df[['member_id', 'churn']].merge(predictions_df, on='member_id')
print(f"Data shape: {data.shape}")
print(f"Actual churn rate: {data['churn'].mean():.3f}")

# Use actual prediction probabilities from the model
print("\n2. Using actual prediction probabilities...")

# The predictions file contains the actual probabilities from predict_proba()
data['prediction_score'] = data['churn_prediction_proba']

# Sort by prediction probability (highest risk first)
data = data.sort_values('prediction_score', ascending=False).reset_index(drop=True)

print(f"Prediction probability range: {data['prediction_score'].min():.3f} - {data['prediction_score'].max():.3f}")
print(f"Members predicted as churn (prob > 0.5): {(data['churn_prediction'] == 1).sum()}")
print(f"Average probability for predicted churners: {data[data['churn_prediction'] == 1]['prediction_score'].mean():.3f}")
print(f"Average probability for predicted non-churners: {data[data['churn_prediction'] == 0]['prediction_score'].mean():.3f}")

# Calculate base rate
base_rate = data['churn'].mean()
print(f"Base churn rate: {base_rate:.3f}")

# Approach 1: Maximum F2@k (Recall-Focused Approach)
print("\n3. Approach 1: Maximum F2@k Analysis")
print("=" * 50)

def calculate_metrics_at_k(data, k):
    """Calculate precision, recall, F2 at top-k predictions"""
    if k == 0:
        return 0, 0, 0
    
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Label exactly the top-k as positive (1), rest as negative (0)
    # This is the correct @k evaluation approach
    data_copy['top_k_prediction'] = 0  # Initialize all as negative
    data_copy.loc[:k-1, 'top_k_prediction'] = 1  # Label top-k as positive
    
    # Calculate metrics against the entire dataset
    precision = precision_score(data_copy['churn'], data_copy['top_k_prediction'], zero_division=0)
    recall = recall_score(data_copy['churn'], data_copy['top_k_prediction'], zero_division=0)
    
    # Calculate F2 manually: F2 = (1 + 2^2) * (precision * recall) / (2^2 * precision + recall)
    # F2 = 5 * (precision * recall) / (4 * precision + recall)
    if precision + recall > 0:
        f2 = 5 * (precision * recall) / (4 * precision + recall)
    else:
        f2 = 0
    
    return precision, recall, f2

# Test different k values
k_values = list(range(100, min(10000, len(data)), 100))  # Test k from 100 to 5000
results_f2 = []

for k in k_values:
    precision, recall, f2 = calculate_metrics_at_k(data, k)
    lift = precision / base_rate if base_rate > 0 else 0
    
    results_f2.append({
        'k': k,
        'precision': precision,
        'recall': recall,
        'f2': f2,
        'lift': lift,
        'coverage': k / len(data)  # Percentage of total population
    })

results_df = pd.DataFrame(results_f2)

# Find optimal k for maximum F2 (recall-focused)
optimal_k_f2 = results_df.loc[results_df['f2'].idxmax()]

print(f"Optimal k for maximum F2 (recall-focused): {optimal_k_f2['k']}")
print(f"F2@k: {optimal_k_f2['f2']:.3f}")
print(f"Precision@k: {optimal_k_f2['precision']:.3f}")
print(f"Recall@k: {optimal_k_f2['recall']:.3f}")
print(f"Lift@k: {optimal_k_f2['lift']:.2f}x")
print(f"Coverage: {optimal_k_f2['coverage']:.1%}")
print("Note: F2 prioritizes recall over precision - important when churn cost > outreach cost")

# Approach 2: Precision-Floor / Lift-Floor (Quality Threshold)
print("\n4. Approach 2: Precision-Floor Analysis")
print("=" * 50)

# Define different lift thresholds (including lower ones to see recall trade-offs)
lift_thresholds = [1.0, 1.2, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]
precision_thresholds = [base_rate * lift for lift in lift_thresholds]

print(f"Base rate: {base_rate:.3f}")
for i, lift in enumerate(lift_thresholds):
    print(f"Lift {lift}x → Precision threshold: {precision_thresholds[i]:.3f}")

results_quality = []

for lift_threshold in lift_thresholds:
    precision_threshold = base_rate * lift_threshold
    
    # Find the largest k that still meets the precision threshold
    valid_results = results_df[results_df['precision'] >= precision_threshold]
    
    if len(valid_results) > 0:
        optimal_k_quality = valid_results.iloc[-1]  # Largest k that meets threshold
        results_quality.append({
            'lift_threshold': lift_threshold,
            'precision_threshold': precision_threshold,
            'optimal_k': optimal_k_quality['k'],
            'precision_at_k': optimal_k_quality['precision'],
            'recall_at_k': optimal_k_quality['recall'],
            'f2_at_k': optimal_k_quality['f2'],
            'coverage': optimal_k_quality['coverage'],
            'lift_at_k': optimal_k_quality['lift']
        })
    else:
        results_quality.append({
            'lift_threshold': lift_threshold,
            'precision_threshold': precision_threshold,
            'optimal_k': 0,
            'precision_at_k': 0,
            'recall_at_k': 0,
            'f2_at_k': 0,
            'coverage': 0,
            'lift_at_k': 0
        })

quality_df = pd.DataFrame(results_quality)
print("\nQuality Threshold Results:")
print(quality_df.round(3))

for _, row in quality_df.iterrows():
    if row['optimal_k'] > 0:
        print(f"Lift {row['lift_threshold']}x: Recall = {row['recall_at_k']:.3f} (k = {row['optimal_k']})")
    else:
        print(f"Lift {row['lift_threshold']}x: No feasible k found")

# Enhanced Approach 2: Quality Threshold with Structured Selection
print("\n5. Enhanced Approach 2: Quality Threshold with Structured Selection")
print("=" * 50)

# Get parameters from constants
R_min = OUTREACH_PARAMS['R_min']
L_min = OUTREACH_PARAMS['L_min']
K_cap = OUTREACH_PARAMS['K_cap']

print(f"Selection Parameters:")
print(f"  R_min (Minimum Recall): {R_min}")
print(f"  L_min (Minimum Lift): {L_min}")
print(f"  K_cap (Maximum Capacity): {K_cap}")

# Find k that satisfies all constraints
feasible_options = []

for _, row in quality_df.iterrows():
    if row['optimal_k'] > 0:  # Only consider feasible options
        k = row['optimal_k']
        recall = row['recall_at_k']
        lift = row['lift_at_k']
        precision = row['precision_at_k']
        
        # Check all constraints
        recall_ok = recall >= R_min
        lift_ok = lift >= L_min
        capacity_ok = k <= K_cap
        
        if recall_ok and lift_ok and capacity_ok:
            feasible_options.append({
                'lift_threshold': row['lift_threshold'],
                'k': k,
                'recall': recall,
                'lift': lift,
                'precision': precision,
                'f2': row['f2_at_k'],
                'coverage': row['coverage']
            })

if feasible_options:
    # Sort by k (smallest first) as per objective
    feasible_options.sort(key=lambda x: x['k'])
    best_structured = feasible_options[0]
    
    print(f"\n Feasible options found: {len(feasible_options)}")
    print(f" Best structured selection:")
    print(f"   Lift threshold: {best_structured['lift_threshold']}x")
    print(f"   Recommended k: {best_structured['k']}")
    print(f"   Recall: {best_structured['recall']:.3f} (≥ {R_min})")
    print(f"   Lift: {best_structured['lift']:.3f} (≥ {L_min})")
    print(f"   Precision: {best_structured['precision']:.3f}")
    print(f"   F2: {best_structured['f2']:.3f}")
    print(f"   Coverage: {best_structured['coverage']:.1%}")
else:
    print(f"\n No feasible options found with current constraints")
    print(f"   Recall requirement: ≥ {R_min}")
    print(f"   Lift requirement: ≥ {L_min}")
    print(f"   Capacity limit: ≤ {K_cap}")

# Calculate business metrics for ROI analysis
total_churners = len(data[data['churn'] == 1])
outreach_cost_per_member = BUSINESS_PARAMS['outreach_cost_per_member']
churn_cost_per_member = BUSINESS_PARAMS['churn_cost_per_member']

# Compare approaches and find minimum k
print("\n6. Approach Comparison and Final Recommendation")
print("=" * 50)

approach_1_k = optimal_k_f2['k']  # F2 optimal
approach_2_k = best_structured['k'] if feasible_options else None  # Structured selection

print(f"Approach 1 (F2 Optimal): k = {approach_1_k}")
print(f"Approach 2 (Structured): k = {approach_2_k if approach_2_k else 'No feasible option'}")

if approach_2_k:
    final_k = min(approach_1_k, approach_2_k)
    final_approach = "F2 Optimal" if approach_1_k <= approach_2_k else "Structured Selection"
    
    print(f"\n🏆 Final Recommendation:")
    print(f"   Selected approach: {final_approach}")
    print(f"   Final k: {final_k}")
    
    # Get the corresponding metrics for the final k
    final_row = results_df[results_df['k'] >= final_k].iloc[0]
    print(f"   Final metrics:")
    print(f"     Precision: {final_row['precision']:.3f}")
    print(f"     Recall: {final_row['recall']:.3f}")
    print(f"     F2: {final_row['f2']:.3f}")
    print(f"     Lift: {final_row['lift']:.3f}")
    print(f"     Coverage: {final_row['coverage']:.1%}")
    
    # Calculate ROI metrics for final recommendation
    final_churners_caught = final_row['recall'] * total_churners
    final_outreach_cost = final_k * outreach_cost_per_member
    final_churn_savings = final_churners_caught * churn_cost_per_member
    final_net_savings = final_churn_savings - final_outreach_cost
    final_roi = (final_churn_savings - final_outreach_cost) / final_outreach_cost * 100
    
    print(f"   Business Impact:")
    print(f"     Churners prevented: {final_churners_caught:.0f}")
    print(f"     Outreach cost: ${final_outreach_cost:,.0f}")
    print(f"     Churn savings: ${final_churn_savings:,.0f}")
    print(f"     Net savings: ${final_net_savings:,.0f}")
    print(f"     ROI: {final_roi:.1f}%")
    print(f"     Cost per churner prevented: ${final_outreach_cost/final_churners_caught:.0f}")
else:
    final_k = approach_1_k
    final_approach = "F2 Optimal (only feasible option)"
    
    print(f"\n🏆 Final Recommendation:")
    print(f"   Selected approach: {final_approach}")
    print(f"   Final k: {final_k}")
    
    # Calculate ROI metrics for final recommendation
    final_row = results_df[results_df['k'] >= final_k].iloc[0]
    final_churners_caught = final_row['recall'] * total_churners
    final_outreach_cost = final_k * outreach_cost_per_member
    final_churn_savings = final_churners_caught * churn_cost_per_member
    final_net_savings = final_churn_savings - final_outreach_cost
    final_roi = (final_churn_savings - final_outreach_cost) / final_outreach_cost * 100
    
    print(f"   Business Impact:")
    print(f"     Churners prevented: {final_churners_caught:.0f}")
    print(f"     Outreach cost: ${final_outreach_cost:,.0f}")
    print(f"     Churn savings: ${final_churn_savings:,.0f}")
    print(f"     Net savings: ${final_net_savings:,.0f}")
    print(f"     ROI: {final_roi:.1f}%")
    print(f"     Cost per churner prevented: ${final_outreach_cost/final_churners_caught:.0f}")

# Visualization
print("\n6. Creating visualizations...")

# Create a single, simple plot for non-technical stakeholders
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate ROI for the plot
total_churners = len(data[data['churn'] == 1])
churners_caught = results_df['recall'] * total_churners
total_outreach_cost = results_df['k'] * outreach_cost_per_member
churn_savings = churners_caught * churn_cost_per_member
roi = (churn_savings - total_outreach_cost) / total_outreach_cost * 100

# Create the main plot with ROI
ax.plot(results_df['k'], roi, 'b-', linewidth=3, label='Return on Investment')

# Create secondary y-axis for recall
ax2 = ax.twinx()
ax2.plot(results_df['k'], results_df['recall'] * 100, 'orange', linewidth=3, label='Recall (%)')
ax2.axhline(y=R_min * 100, color='red', linestyle='dashed', alpha=0.7, linewidth=1, label=f'Recall Threshold ({R_min*100:.0f}%)')
ax2.set_ylabel('Recall (%)', fontsize=14)
ax2.tick_params(axis='y', labelsize=12)

# Highlight the selected k point (Approach 2 - Structured Selection)
ax.axvline(x=final_k, color='green', linestyle='dashed', linewidth=1, 
           label=f'Recommended (Approach 2): {int(final_k)} outreaches')

# Highlight the F2 optimal k point (Approach 1)
ax.axvline(x=optimal_k_f2['k'], color='purple', linestyle='dashed', linewidth=1, 
           label=f'F2 Optimal (Approach 1): {int(optimal_k_f2["k"])} outreaches')

# Add point markers at both selected k points
selected_roi = roi[results_df['k'] >= final_k].iloc[0]
selected_recall = results_df[results_df['k'] >= final_k]['recall'].iloc[0] * 100

f2_roi = roi[results_df['k'] >= optimal_k_f2['k']].iloc[0]
f2_recall = results_df[results_df['k'] >= optimal_k_f2['k']]['recall'].iloc[0] * 100


# Formatting
ax.set_xlabel('Number of Customer Outreaches', fontsize=14)
ax.set_ylabel('Return on Investment (%)', fontsize=14)
ax.set_title('Outreach Campaign Analysis\n(ROI vs Recall - Two Approaches)', 
             fontsize=16, fontweight='bold')
# Combine legends from both axes
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', which='major', labelsize=12)

# # Add business context
# ax.text(0.02, 0.98, f'Approach 2 (Recommended):\n• {int(final_k)} customers\n• {selected_roi:.1f}% ROI, {selected_recall:.1f}% Recall\n• ${int(final_k * outreach_cost_per_member):,} cost\n\nApproach 1 (F2 Optimal):\n• {int(optimal_k_f2["k"])} customers\n• {f2_roi:.1f}% ROI, {f2_recall:.1f}% Recall\n• ${int(optimal_k_f2["k"] * outreach_cost_per_member):,} cost', 
#         transform=ax.transAxes, fontsize=11, verticalalignment='top',
#         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('vi_outreach_output/outreach_optimization_analysis.png', dpi=300, bbox_inches='tight', transparent=True)
print("Visualization saved to 'vi_outreach_output/outreach_optimization_analysis.png'")

# Print the key information from the graph
print("\n GRAPH SUMMARY:")
print("=" * 50)
print(" ROI vs Recall Analysis:")
print(f"   • ROI ranges from {roi.min():.1f}% to {roi.max():.1f}%")
print(f"   • Recall ranges from {results_df['recall'].min()*100:.1f}% to {results_df['recall'].max()*100:.1f}%")
print(f"   • Recall threshold: {R_min*100:.0f}% (from constants)")

print(f"\n APPROACH 2 (Recommended - Green dashed line):")
print(f"   • Optimal k: {int(final_k)} customers")
print(f"   • ROI: {selected_roi:.1f}%")
print(f"   • Recall: {selected_recall:.1f}%")
print(f"   • Cost: ${int(final_k * outreach_cost_per_member):,}")
print(f"   • Net savings: ${int(final_k * outreach_cost_per_member * selected_roi/100):,}")

print(f"\n APPROACH 1 (F2 Optimal - Purple dashed line):")
print(f"   • Optimal k: {int(optimal_k_f2['k'])} customers")
print(f"   • ROI: {f2_roi:.1f}%")
print(f"   • Recall: {f2_recall:.1f}%")
print(f"   • Cost: ${int(optimal_k_f2['k'] * outreach_cost_per_member):,}")
print(f"   • Net savings: ${int(optimal_k_f2['k'] * outreach_cost_per_member * f2_roi/100):,}")

print(f"\n GRAPH ELEMENTS:")
print(f"   • Blue line: Return on Investment (%)")
print(f"   • Orange line: Recall (%)")
print(f"   • Red dashed line: Recall threshold ({R_min*100:.0f}%)")
print(f"   • Green dashed line: Approach 2 threshold ({int(final_k)} customers)")
print(f"   • Purple dashed line: Approach 1 threshold ({int(optimal_k_f2['k'])} customers)")
print(f"   • Business context box: Shows both approaches with metrics")

# Save detailed results
print("\n7. Saving results...")

# Save F2 optimization results
results_df.to_csv('vi_outreach_output/f2_optimization_results.csv', index=False)
print("F2 optimization results saved to 'vi_outreach_output/f2_optimization_results.csv'")

# Save quality threshold results
quality_df.to_csv('vi_outreach_output/quality_threshold_results.csv', index=False)
print("Quality threshold results saved to 'vi_outreach_output/quality_threshold_results.csv'")

# Save recommended outreach lists
# For F2 optimization (recall-focused)
optimal_k_f2_int = int(optimal_k_f2['k'])
f2_recommendations = data.head(optimal_k_f2_int)[['member_id', 'prediction_score', 'churn_prediction']]
f2_recommendations.to_csv('vi_outreach_output/f2_optimal_outreach_list.csv', index=False)
print(f"F2 optimal outreach list ({optimal_k_f2_int} members) saved to 'vi_outreach_output/f2_optimal_outreach_list.csv'")

# Save final recommendation outreach list with rank
final_k_int = int(final_k)
final_recommendations = data.head(final_k_int)[['member_id', 'prediction_score', 'churn_prediction']]
final_recommendations = final_recommendations.sort_values('prediction_score', ascending=False).reset_index(drop=True)
final_recommendations['rank'] = final_recommendations.index + 1
final_recommendations = final_recommendations[['member_id', 'prediction_score', 'rank', 'churn_prediction']]
final_recommendations.to_csv('vi_outreach_output/final_recommended_outreach_list.csv', index=False)
print(f"Final recommended outreach list ({final_k_int} members) saved to 'vi_outreach_output/final_recommended_outreach_list.csv'")
print("Format: member_id, prediction_score, rank, churn_prediction")

# For quality thresholds
for _, row in quality_df.iterrows():
    if row['optimal_k'] > 0:
        optimal_k_int = int(row['optimal_k'])
        quality_recommendations = data.head(optimal_k_int)[['member_id', 'prediction_score', 'churn_prediction']]
        filename = f'vi_outreach_output/quality_lift_{row["lift_threshold"]}x_outreach_list.csv'
        quality_recommendations.to_csv(filename, index=False)
        print(f"Quality {row['lift_threshold']}x lift outreach list ({optimal_k_int} members) saved to '{filename}'")

print("\n Outreach optimization analysis complete!")
print("\n Key Files Generated:")
print("   ├── outreach_optimization_analysis.png")
print("   ├── f2_optimization_results.csv")
print("   ├── quality_threshold_results.csv")
print("   ├── f2_optimal_outreach_list.csv")
print("   ├── final_recommended_outreach_list.csv (includes rank for assignment)")
print("   └── quality_lift_*x_outreach_list.csv")
print("   All files saved to 'vi_outreach_output/' folder")
