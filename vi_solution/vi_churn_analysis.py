import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

print("=== VI Data Science Assignment: Churn Prediction Analysis ===\n")

# 1. Load the data
print("1. Loading data...")
churn_labels = pd.read_csv('data/churn_labels.csv')
app_usage = pd.read_csv('data/app_usage.csv')
claims = pd.read_csv('data/claims.csv')
web_visits = pd.read_csv('data/web_visits.csv')

print(f"Churn labels: {churn_labels.shape}")
print(f"App usage: {app_usage.shape}")
print(f"Claims: {claims.shape}")
print(f"Web visits: {web_visits.shape}\n")

# 2. Basic data exploration
print("2. Basic data exploration...")

# Churn labels overview
print("Churn Labels Overview:")
print(f"Total members: {len(churn_labels)}")
print(f"Churn rate: {churn_labels['churn'].mean():.3f} ({churn_labels['churn'].sum()} churned)")
print(f"Outreach rate: {churn_labels['outreach'].mean():.3f} ({churn_labels['outreach'].sum()} received outreach)")
print(f"Date range: {churn_labels['signup_date'].min()} to {churn_labels['signup_date'].max()}")

# Check for missing values
print("\nMissing values:")
print(churn_labels.isnull().sum())

# Convert dates
churn_labels['signup_date'] = pd.to_datetime(churn_labels['signup_date'])
app_usage['timestamp'] = pd.to_datetime(app_usage['timestamp'])
claims['diagnosis_date'] = pd.to_datetime(claims['diagnosis_date'])
web_visits['timestamp'] = pd.to_datetime(web_visits['timestamp'])

print("\n" + "="*50 + "\n")

# 3. Feature Engineering
print("3. Feature Engineering...")

# Start with churn labels as base
features = churn_labels.copy()

# 3.1 App Usage Features
print("3.1 Creating app usage features...")
app_features = app_usage.groupby('member_id').agg({
    'timestamp': ['count', 'min', 'max']
}).reset_index()

app_features.columns = ['member_id', 'session_count', 'first_session', 'last_session']
app_features['session_days'] = (app_features['last_session'] - app_features['first_session']).dt.days
app_features['avg_sessions_per_day'] = app_features['session_count'] / (app_features['session_days'] + 1)

# Add session frequency in last 7, 14, 30 days
def get_recent_sessions(df, days):
    cutoff_date = df['timestamp'].max() - timedelta(days=days)
    recent = df[df['timestamp'] >= cutoff_date]
    return recent.groupby('member_id').size()

recent_7 = get_recent_sessions(app_usage, 7)
recent_14 = get_recent_sessions(app_usage, 14)
recent_30 = get_recent_sessions(app_usage, 30)

app_features = app_features.merge(recent_7.rename('sessions_7d'), on='member_id', how='left')
app_features = app_features.merge(recent_14.rename('sessions_14d'), on='member_id', how='left')
app_features = app_features.merge(recent_30.rename('sessions_30d'), on='member_id', how='left')

# Fill only numeric counts; keep datetime columns as NaT
for col in ['session_count', 'sessions_7d', 'sessions_14d', 'sessions_30d']:
    if col in app_features.columns:
        app_features[col] = app_features[col].fillna(0)

# Weekly activity patterns and time-of-day buckets
print("3.1.1 Deriving weekly patterns, dayparts, and inactivity gaps...")
app_tmp = app_usage.copy()
app_tmp['date'] = app_tmp['timestamp'].dt.date
app_tmp['dow'] = app_tmp['timestamp'].dt.dayofweek  # 0=Mon
app_tmp['is_weekend'] = app_tmp['dow'].isin([5, 6]).astype(int)
app_tmp['hour'] = app_tmp['timestamp'].dt.hour

def daypart(h):
    if 5 <= h < 12:
        return 'morning'
    if 12 <= h < 17:
        return 'afternoon'
    if 17 <= h < 22:
        return 'evening'
    return 'night'

app_tmp['daypart'] = app_tmp['hour'].apply(daypart)

# Active days
active_days = app_tmp.groupby('member_id')['date'].nunique().rename('active_days')
weekend_sessions = app_tmp.groupby('member_id')['is_weekend'].mean().rename('weekend_session_share')

# Daypart shares
daypart_counts = app_tmp.pivot_table(index='member_id', columns='daypart', values='timestamp', aggfunc='count', fill_value=0)
for c in ['morning','afternoon','evening','night']:
    if c not in daypart_counts.columns:
        daypart_counts[c] = 0
daypart_tot = daypart_counts.sum(axis=1).replace(0, 1)
daypart_share = daypart_counts.div(daypart_tot, axis=0)
daypart_share = daypart_share.add_prefix('share_')

# Fix column names to avoid numeric column names and ensure proper naming
daypart_share.columns = [f'share_{col}' if col.isdigit() else col for col in daypart_share.columns]

# Gaps between sessions
def gap_stats(group):
    ts = group.sort_values().values
    if len(ts) <= 1:
        return pd.Series({'avg_gap_days': 999.0, 'max_gap_days': 999.0})
    diffs = np.diff(ts).astype('timedelta64[s]').astype(float) / 86400.0
    return pd.Series({'avg_gap_days': float(np.mean(diffs)), 'max_gap_days': float(np.max(diffs))})

gap_df = (
    app_tmp.groupby('member_id')['timestamp']
           .apply(gap_stats)        # returns a Series with ('member_id', inner_key)
           .unstack()               # -> columns: ['avg_gap_days','max_gap_days']
           .reset_index()           # -> ['member_id','avg_gap_days','max_gap_days']
)

# Active days per week (normalize by observed span)
obs_span_days = (app_features['last_session'] - app_features['first_session']).dt.days.replace(0, 1)
app_features['active_days_per_week'] = 0.0
tmp_active = active_days.reindex(app_features['member_id']).values
app_features.loc[:, 'active_days_per_week'] = (tmp_active / (obs_span_days.values / 7.0)).astype(float)
app_features['active_days_per_week'] = app_features['active_days_per_week'].replace([np.inf, -np.inf], 0).fillna(0)

# Merge additional app usage features
app_features = app_features.merge(active_days, on='member_id', how='left')
app_features = app_features.merge(weekend_sessions, on='member_id', how='left')
app_features = app_features.merge(daypart_share, on='member_id', how='left')
app_features = app_features.merge(gap_df, on='member_id', how='left')
for col in ['weekend_session_share','share_morning','share_afternoon','share_evening','share_night','avg_gap_days','max_gap_days']:
    if col in app_features.columns:
        app_features[col] = app_features[col].fillna(0)

# 3.2 Claims Features
print("3.2 Creating claims features...")
claims_features = claims.groupby('member_id').agg({
    'icd_code': 'count',
    'diagnosis_date': ['min', 'max']
}).reset_index()

claims_features.columns = ['member_id', 'total_claims', 'first_claim', 'last_claim']

# Check for specific ICD codes mentioned in the brief
key_icd_codes = ['E11.9', 'I10', 'Z71.3']  # Diabetes, Hypertension, Dietary counseling
for code in key_icd_codes:
    has_code = claims[claims['icd_code'] == code]['member_id'].unique()
    claims_features[f'has_{code}'] = claims_features['member_id'].isin(has_code).astype(int)

# 3.3 Web Visits Features
print("3.3 Creating web visits features...")
web_features = web_visits.groupby('member_id').agg({
    'url': 'count',
    'timestamp': ['min', 'max']
}).reset_index()

web_features.columns = ['member_id', 'total_visits', 'first_visit', 'last_visit']

# Categorize visits by content type
health_keywords = ['health', 'diabetes', 'hypertension', 'heart', 'nutrition', 'exercise', 'sleep', 'stress']
tech_keywords = ['tech', 'gaming', 'gadget', 'smartphone', 'laptop']

def categorize_visits(df):
    df['is_health'] = df['url'].str.contains('|'.join(health_keywords), case=False, na=False)
    df['is_tech'] = df['url'].str.contains('|'.join(tech_keywords), case=False, na=False)
    return df

web_visits_categorized = categorize_visits(web_visits.copy())

health_visits = web_visits_categorized[web_visits_categorized['is_health']].groupby('member_id').size().rename('health_visits')
tech_visits = web_visits_categorized[web_visits_categorized['is_tech']].groupby('member_id').size().rename('tech_visits')

web_features = web_features.merge(health_visits, on='member_id', how='left')
web_features = web_features.merge(tech_visits, on='member_id', how='left')
# Fill numeric counts; keep datetime columns as NaT
for col in ['total_visits', 'health_visits', 'tech_visits']:
    if col in web_features.columns:
        web_features[col] = web_features[col].fillna(0)

# Additional web visit enrichments: domain & path depth
def extract_domain(url: pd.Series) -> pd.Series:
    try:
        return (url.str.split('/').str[2]).str.lower()
    except Exception:
        return pd.Series(index=url.index, dtype='object')

def path_depth(url: pd.Series) -> pd.Series:
    # number of segments after domain
    try:
        return url.str.split('/').apply(lambda parts: max(len(parts) - 3, 0))
    except Exception:
        return pd.Series(index=url.index, dtype='int64')

web_visits_categorized['domain'] = extract_domain(web_visits_categorized['url'])
web_visits_categorized['path_depth'] = path_depth(web_visits_categorized['url'])

# Health taxonomy features with recency
print("3.3.1 Deriving health taxonomy features...")

def contains_any(s: pd.Series, kws):
    pat = '|'.join([re.escape(k) for k in kws])
    return s.str.contains(pat, case=False, na=False)

taxonomy = {
    'diabetes': ['diabetes', 'glycemic', 'glucose', 'a1c'],
    'hypertension': ['hypertension', 'blood pressure', 'bp'],
    'cardio': ['cardio', 'heart', 'cardiovascular'],
    'sleep': ['sleep', 'sleep apnea'],
    'nutrition': ['nutrition', 'diet', 'fiber', 'cholesterol', 'lipid'],
    'exercise': ['exercise', 'aerobic', 'strength training', 'cardio'],
    'stress': ['stress', 'mindfulness', 'meditation', 'wellbeing', 'mental health']
}

text_cols = web_visits_categorized[['url','title','description']].fillna('')
combined_text = (text_cols['url'] + ' ' + text_cols['title'] + ' ' + text_cols['description'])

for name, kws in taxonomy.items():
    web_visits_categorized[f'is_{name}'] = contains_any(combined_text, kws)

def category_aggregates(df, flag_col: str, ts_col: str):
    grp = df[df[flag_col]].groupby('member_id')
    counts = grp.size().rename(f'{flag_col}_count')
    last_ts = grp[ts_col].max().rename(f'{flag_col}_last_ts')
    return counts, last_ts

cat_counts = []
cat_last_ts = []
for name in taxonomy.keys():
    c, t = category_aggregates(web_visits_categorized, f'is_{name}', 'timestamp')
    cat_counts.append(c)
    cat_last_ts.append(t)

cat_counts_df = pd.concat(cat_counts, axis=1).reset_index() if cat_counts else pd.DataFrame()
cat_last_df = pd.concat(cat_last_ts, axis=1).reset_index() if cat_last_ts else pd.DataFrame()


unique_domains = web_visits_categorized.groupby('member_id')['domain'].nunique().rename('unique_domains')
avg_path_depth = web_visits_categorized.groupby('member_id')['path_depth'].mean().rename('avg_path_depth')

web_features = web_features.merge(unique_domains, on='member_id', how='left')
web_features = web_features.merge(avg_path_depth, on='member_id', how='left')
for col in ['unique_domains', 'avg_path_depth']:
    if col in web_features.columns:
        web_features[col] = web_features[col].fillna(0)

# Merge taxonomy counts and recency
if not cat_counts_df.empty:
    web_features = web_features.merge(cat_counts_df, on='member_id', how='left')
if not cat_last_df.empty:
    web_features = web_features.merge(cat_last_df, on='member_id', how='left')

# Compute ratios and recency days for taxonomy
for name in ['diabetes','hypertension','cardio','sleep','nutrition','exercise','stress']:
    cnt_col = f'is_{name}_count'
    ts_col = f'is_{name}_last_ts'
    if cnt_col in web_features.columns:
        web_features[cnt_col] = web_features[cnt_col].fillna(0)
        web_features[f'{name}_ratio'] = np.where(web_features['total_visits'] > 0,
                                                web_features[cnt_col] / web_features['total_visits'], 0.0)
    if ts_col in web_features.columns:
        web_features[ts_col] = pd.to_datetime(web_features[ts_col], errors='coerce')

# 3.4 Merge all features
print("3.4 Merging all features...")
features = features.merge(app_features, on='member_id', how='left')
features = features.merge(claims_features, on='member_id', how='left')
features = features.merge(web_features, on='member_id', how='left')

# Do not blanket-fill all; only fill numeric NaNs
numeric_cols = features.select_dtypes(include=[np.number]).columns
features[numeric_cols] = features[numeric_cols].fillna(0)

# 3.5 Create additional features
print("3.5 Creating additional features...")

# Time-based features (simplified)
# Ensure datetime columns are proper datetimes (coerce zeros/strings to NaT)
for dc in ['signup_date', 'first_session', 'last_session', 'first_claim', 'last_claim', 'first_visit', 'last_visit']:
    if dc in features.columns:
        features[dc] = pd.to_datetime(features[dc], errors='coerce')

features['days_since_signup'] = 0  # Placeholder
mask_last_session = features['last_session'].notna()
features.loc[mask_last_session, 'days_since_signup'] = (
    features.loc[mask_last_session, 'last_session'] - features.loc[mask_last_session, 'signup_date']
).dt.days

# Engagement ratios (set 0 when total_visits == 0)
features['health_visit_ratio'] = np.where(
    features['total_visits'] > 0,
    features['health_visits'] / features['total_visits'],
    0.0
)
features['tech_visit_ratio'] = np.where(
    features['total_visits'] > 0,
    features['tech_visits'] / features['total_visits'],
    0.0
)

# Normalize recent session windows to per-day rates
features['sessions_7d_per_day'] = features['sessions_7d'] / 7.0
features['sessions_14d_per_day'] = features['sessions_14d'] / 14.0
features['sessions_30d_per_day'] = features['sessions_30d'] / 30.0

# Activity recency (simplified)
max_session_date = app_usage['timestamp'].max()
max_visit_date = web_visits['timestamp'].max()
max_claim_date = claims['diagnosis_date'].max()

features['days_since_last_session'] = 999  # Default for no activity
mask_last_session = features['last_session'].notna()
features.loc[mask_last_session, 'days_since_last_session'] = (
    max_session_date - features.loc[mask_last_session, 'last_session']
).dt.days

features['days_since_last_visit'] = 999
mask_last_visit = features['last_visit'].notna()
features.loc[mask_last_visit, 'days_since_last_visit'] = (
    max_visit_date - features.loc[mask_last_visit, 'last_visit']
).dt.days

features['days_since_last_claim'] = 999
mask_last_claim = features['last_claim'].notna()
features.loc[mask_last_claim, 'days_since_last_claim'] = (
    max_claim_date - features.loc[mask_last_claim, 'last_claim']
).dt.days

# Taxonomy recency in days
for name in ['diabetes','hypertension','cardio','sleep','nutrition','exercise','stress']:
    ts_col = f'is_{name}_last_ts'
    if ts_col in features.columns:
        tmp = pd.to_datetime(features[ts_col], errors='coerce')
        features[f'{name}_recency_days'] = (max_visit_date - tmp).dt.days.fillna(999)

print(f"Final feature set shape: {features.shape}")
print(f"Features created: {len(features.columns)}")

print("\n" + "="*50 + "\n")

# 4. Data Analysis
print("4. Data Analysis...")

# 4.1 Churn vs No Churn comparison
print("4.1 Churn vs No Churn Analysis:")
churn_stats = features.groupby('churn').agg({
    'session_count': ['mean', 'std'],
    'total_visits': ['mean', 'std'],
    'health_visits': ['mean', 'std'],
    'total_claims': ['mean', 'std'],
    'days_since_last_session': ['mean', 'std']
}).round(2)

print(churn_stats)

# 4.2 Outreach effectiveness
print("\n4.2 Outreach Effectiveness Analysis:")
outreach_stats = features.groupby(['outreach', 'churn']).size().unstack(fill_value=0)
print("Churn by Outreach Status:")
print(outreach_stats)

# Calculate churn rates
churn_rate_no_outreach = outreach_stats.loc[0, 1] / outreach_stats.loc[0].sum()
churn_rate_with_outreach = outreach_stats.loc[1, 1] / outreach_stats.loc[1].sum()

print(f"\nChurn rate without outreach: {churn_rate_no_outreach:.3f}")
print(f"Churn rate with outreach: {churn_rate_with_outreach:.3f}")
print(f"Outreach effectiveness: {churn_rate_no_outreach - churn_rate_with_outreach:.3f}")

# 4.3 Key ICD codes analysis
print("\n4.3 Key ICD Codes Analysis:")
for code in key_icd_codes:
    code_col = f'has_{code}'
    if code_col in features.columns:
        churn_by_code = features.groupby(code_col)['churn'].mean()
        print(f"{code}: Churn rate with code = {churn_by_code[1]:.3f}, without = {churn_by_code[0]:.3f}")

print("\n" + "="*50 + "\n")

# 5. Save processed data
print("5. Saving processed data...")

# Remove duplicates if any exist
print(f"Before deduplication: {len(features)} rows")
features = features.drop_duplicates(subset=['member_id'], keep='first')
print(f"After deduplication: {len(features)} rows")

features.to_csv('vi_churn_analysis_output/processed_features.csv', index=False)
print("Processed features saved to 'vi_churn_analysis_output/processed_features.csv'")

# 6. Summary of key findings
print("\n6. Key Findings Summary:")
print(f"- Total members analyzed: {len(features)}")
print(f"- Overall churn rate: {features['churn'].mean():.3f}")
print(f"- Members with app sessions: {(features['session_count'] > 0).sum()}")
print(f"- Members with web visits: {(features['total_visits'] > 0).sum()}")
print(f"- Members with claims: {(features['total_claims'] > 0).sum()}")
print(f"- Outreach appears to {'reduce' if churn_rate_with_outreach < churn_rate_no_outreach else 'increase'} churn")

# 7. Feature list
print("\n7. Features created:")
feature_columns = [col for col in features.columns if col not in ['member_id', 'signup_date', 'churn', 'outreach']]
for i, col in enumerate(feature_columns, 1):
    print(f"{i:2d}. {col}")

print("\n=== Data Exploration and Feature Engineering Complete ===")

# 8. Create visualizations for presentation
print("\n8. Creating visualizations for presentation...")
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
os.makedirs('vi_churn_analysis_output', exist_ok=True)

# Set style for clean, professional look
plt.style.use('default')
sns.set_palette("husl")

# 8.1 KPI Panel
print("Creating KPI panel...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# KPI Boxes with better spacing
kpi_data = [
    ('Population', f'{len(features):,}', '#2E86AB'),
    ('Churn Rate', f'{features["churn"].mean()*100:.1f}%', '#A23B72'),
    ('Outreach Rate', f'{features["outreach"].mean()*100:.1f}%', '#F18F01')
]

y_positions = [0.75, 0.45, 0.15]
for i, (label, value, color) in enumerate(kpi_data):
    # KPI Box
    rect = plt.Rectangle((0.1, y_positions[i]-0.12), 0.8, 0.2, 
                        facecolor=color, alpha=0.9, edgecolor='white', linewidth=3)
    ax.add_patch(rect)
    
    # Labels with better positioning
    ax.text(0.5, y_positions[i]+0.03, label, ha='center', va='center', 
            fontsize=16, fontweight='bold', color='white')
    ax.text(0.5, y_positions[i]-0.03, value, ha='center', va='center', 
            fontsize=24, fontweight='bold', color='white')

ax.set_title('Key Performance Indicators', fontsize=18, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('vi_churn_analysis_output/kpi_panel.png', dpi=300, bbox_inches='tight', 
            transparent=True)
plt.close()
print("KPI panel saved to 'vi_churn_analysis_output/kpi_panel.png'")

# 8.2 Churn by Outreach Comparison
print("Creating churn comparison chart...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

outreach_stats = features.groupby(['outreach', 'churn']).size().unstack(fill_value=0)
churn_rates = outreach_stats[1] / outreach_stats.sum(axis=1) * 100

bars = ax.bar(['No Outreach', 'With Outreach'], churn_rates, 
              color=['#E74C3C', '#27AE60'], alpha=0.8, edgecolor='white', linewidth=2, width=0.6)

# Add value labels on bars
for bar, rate in zip(bars, churn_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_ylabel('Churn Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Churn Rate by Outreach Status', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, max(churn_rates) * 1.2)
ax.grid(axis='y', alpha=0.3)

# Add effectiveness annotation with better positioning
effectiveness = churn_rates[0] - churn_rates[1]
effectiveness_text = f"Outreach Effectiveness:\n{effectiveness:.1f}% reduction"
ax.text(0.5, 0.92, effectiveness_text, transform=ax.transAxes, 
        ha='center', va='top', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('vi_churn_analysis_output/churn_comparison.png', dpi=300, bbox_inches='tight', 
            transparent=True)
plt.close()
print("Churn comparison chart saved to 'vi_churn_analysis_output/churn_comparison.png'")

# Create a summary slide text
print("\n📊 PRESENTATION SLIDE SUMMARY:")
print("=" * 50)
print("🎯 Key Insights for Slide 1:")
print(f"   • Total Population: {len(features):,} members")
print(f"   • Overall Churn Rate: {features['churn'].mean()*100:.1f}%")
print(f"   • Outreach Coverage: {features['outreach'].mean()*100:.1f}% of population")
print(f"   • Outreach Effectiveness: {effectiveness:.1f}% churn reduction")
print(f"   • Business Impact: Outreach reduces churn from {churn_rates[0]:.1f}% to {churn_rates[1]:.1f}%")
print("\n📈 Visualization Elements:")
print("   • KPI Panel: Population, Churn Rate, Outreach Rate")
print("   • Churn Comparison: No Outreach vs With Outreach")
print("   • Effectiveness Highlight: Clear reduction in churn rate")
