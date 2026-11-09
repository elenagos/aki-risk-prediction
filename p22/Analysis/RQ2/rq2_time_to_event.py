# Main writer:Cahit kok 
# Reviewer: Noah Charpentier
# Contributors: Kimi Knaider

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ###################################################################################
# SETUP
# ###################################################################################

# Add parent directory (Analysis/model/) to path so we can import model_config
SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Go up to p22/
sys.path.insert(0, str(SCRIPT_DIR / "model"))
from model_config import DATASET_PATH, TARGET, engineer_features

#
print("RQ2 TIME-TO-EVENT ANALYSIS: AKI ONSET TIMING")
#

# Load and engineer data
df = pd.read_csv(DATASET_PATH)
df = engineer_features(df)  # Add interaction features
print(f"\n Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f" Engineered features computed")

# ###################################################################################
# CREATE TIME-TO-AKI OUTCOMES (Proper Survival Data)
# ###################################################################################

#
print("DEFINING SURVIVAL DATA: TIME-TO-AKI OUTCOMES")
#

# Create survival data with proper censoring logic
surv_df = df[['subject_id', 'stay_id', 'aki_day7']].copy()

# Handle time-to-AKI
if 'time_to_aki_hours' in df.columns:
    surv_df['time_to_aki_hours'] = df['time_to_aki_hours'].fillna(7 * 24)

    surv_df.loc[surv_df['aki_day7'] == 0, 'time_to_aki_hours'] = 7 * 24
    
    surv_df['T'] = surv_df['time_to_aki_hours'] / 24  # Convert hours to days
    surv_df['E'] = surv_df['aki_day7'].astype(int)  # Event indicator (1=AKI, 0=censored)
else:
    # Fallback: use aki_day7 as binary outcome with default 7-day window
    print(" 'time_to_aki_hours' not found. Using aki_day7 as event indicator.")
    surv_df['T'] = 7  # Default observation window: 7 days
    surv_df['E'] = df['aki_day7'].astype(int)

# Ensure T is bounded [0, 7]
surv_df['T'] = surv_df['T'].clip(0, 7)

# Create AKI onset timing categories
def categorize_aki_timing(row):
    """Categorizes AKI by onset timing."""
    if row['aki_day7'] == 0:
        return 'No AKI (Censored)'
    elif row['time_to_aki_hours'] <= 24:
        return 'Very Early AKI (0-1 day)'
    elif row['time_to_aki_hours'] <= 48:
        return 'Early AKI (1-2 days)'
    elif row['time_to_aki_hours'] <= 96:
        return 'Mid AKI (2-4 days)'
    else:
        return 'Late AKI (4-7 days)'

surv_df['AKI_Category'] = surv_df.apply(categorize_aki_timing, axis=1)

# Print survival summary
print(f"\nSurvival Summary (N={len(surv_df)}):")
print(f"  Events (AKI cases): {surv_df['E'].sum()}")
print(f"  Censored (no AKI): {(1 - surv_df['E']).sum()}")
print(f"  Median follow-up time: {surv_df['T'].median():.1f} days")
print(f"  Median time-to-AKI (AKI cases only): {surv_df.loc[surv_df['E']==1, 'T'].median():.2f} days")

print(f"\nAKI Onset Timing Distribution:")
timing_dist = surv_df['AKI_Category'].value_counts()
for cat in ['No AKI (Censored)', 'Very Early AKI (0-1 day)', 'Early AKI (1-2 days)', 
            'Mid AKI (2-4 days)', 'Late AKI (4-7 days)']:
    if cat in timing_dist.index:
        count = timing_dist[cat]
        pct = count / len(surv_df) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")

# ============================================================================
# MERGE WITH PREDICTOR VARIABLES (LEAKAGE-FREE FEATURES FROM RQ1)
# ============================================================================

#
print("SELECTING COX MODEL FEATURES (Leakage-Free Predictors)")
#

cox_features = [

    "age",
    "baseline_creatinine",
    "charlson_index",
    "sofa_score",

    "bun_max",
    "lactate_max",

    "exposure",

    "vasopressor_use",

    "num_nephrotoxic_drugs"

]
# Select features that exist in the dataframe
cox_features = [f for f in cox_features if f in df.columns]
print(f" Cox model features selected: {len(cox_features)}")

# Prepare Cox model data
cox_data = df[['subject_id', 'stay_id'] + cox_features].copy()
surv_data = surv_df.merge(cox_data, on=['subject_id', 'stay_id'], how='left')

print(f"\n Merged survival + predictor data: {surv_data.shape}")

# Drop fully missing Cox features before imputation
all_missing_features = [f for f in cox_features if surv_data[f].isna().all()]
if all_missing_features:
    print(f" Dropping fully-missing Cox features before imputation: {all_missing_features}")
usable_cox_features = [f for f in cox_features if f not in all_missing_features]
print(f" Usable Cox features after dropping fully-missing columns: {len(usable_cox_features)}")

# ###################################################################################
# HANDLE MISSING VALUES (Median Imputation)
# ###################################################################################

#
print("HANDLING MISSING VALUES")

#

# Check missingness before imputation
missing_before = surv_data[usable_cox_features].isnull().sum().sum()
print(f"Missing values before imputation: {missing_before:,}")

# Use median imputation (don't remove patients)
imputer = SimpleImputer(strategy='median')
surv_data[usable_cox_features] = imputer.fit_transform(surv_data[usable_cox_features])

missing_after = surv_data[usable_cox_features].isnull().sum().sum()
print(f"Missing values after imputation: {missing_after:,}")
print(f"Sample size retained: {len(surv_data):,} (100% vs. dropna approach)")

# ###################################################################################
# BASELINE CHARACTERISTICS TABLE BY AKI ONSET TIMING
# ###################################################################################

#
print("BASELINE CHARACTERISTICS BY AKI ONSET TIMING")
#

baseline_features = [
    'age', 'baseline_creatinine', 'charlson_index', 'sofa_score',
    'heart_rate_mean', 'systolic_bp_mean', 'bun_mean', 'lactate_mean'
]

# Keep only columns that actually exist in surv_data
baseline_features = [col for col in baseline_features if col in surv_data.columns]

if len(baseline_features) > 0:
    char_summary = (
        surv_data
        .groupby('AKI_Category')[baseline_features]
        .agg(['count', 'mean', 'std'])
        .round(2)
    )
    print("\nBaseline Characteristics by AKI Onset Timing:")
    print(char_summary)

    char_output_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_baseline_characteristics.csv"
    char_output_path.parent.mkdir(parents=True, exist_ok=True)
    char_summary.to_csv(char_output_path)
    print(f" Saved: {char_output_path}")
else:
    print(" No baseline features available for baseline characteristics table.")

print("\nBaseline Characteristics by AKI Onset Timing:")
print(char_summary)

# Save baseline characteristics table
char_output_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_baseline_characteristics.csv"
char_output_path.parent.mkdir(parents=True, exist_ok=True)
char_summary.to_csv(char_output_path)
print(f" Saved: {char_output_path}")

# ###################################################################################
# KAPLAN-MEIER CURVES WITH LOG-RANK TESTS
# ###################################################################################

#
print("KAPLAN-MEIER SURVIVAL ANALYSIS WITH LOG-RANK TESTS")
#

kmf = KaplanMeierFitter()

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('RQ2 Kaplan-Meier Curves: AKI-Free Survival by Risk Group',
             fontsize=14, fontweight='bold')

logrank_results = []

# Plot 1: Overall survival
ax1 = axes[0, 0]
kmf.fit(surv_data['T'], surv_data['E'], label='All Patients')
kmf.plot_survival_function(ax=ax1)
ax1.set_xlabel('Days from Index Time')
ax1.set_ylabel('AKI-Free Probability')
ax1.set_title(f'Overall AKI-Free Survival (N={len(surv_data):,})')
ax1.grid(alpha=0.3)
ax1.set_ylim([0, 1.05])

# Plot 2: By exposure group (with log-rank test)
ax2 = axes[0, 1]
colors = ['#2E86AB', '#A23B72']
for idx, (exposure, label) in enumerate([(0, 'Vanco Only'), (1, 'Vanco + PTZ')]):
    mask = surv_data['exposure'] == exposure
    T_group = surv_data.loc[mask, 'T']
    E_group = surv_data.loc[mask, 'E']
    kmf.fit(T_group, E_group, label=f'{label} (N={mask.sum():,})')
    kmf.plot_survival_function(ax=ax2, color=colors[idx])

# Log-rank test: exposure group
mask0 = surv_data['exposure'] == 0
mask1 = surv_data['exposure'] == 1
result_exp = logrank_test(surv_data.loc[mask0, 'T'], surv_data.loc[mask1, 'T'],
                          surv_data.loc[mask0, 'E'], surv_data.loc[mask1, 'E'])
logrank_results.append(('Exposure Group (Vanco Only vs Combo)', result_exp.p_value))
print(f"\nLog-Rank Test: Exposure Group")
print(f"  p-value: {result_exp.p_value:.4f} {'***' if result_exp.p_value < 0.001 else '**' if result_exp.p_value < 0.01 else '*' if result_exp.p_value < 0.05 else 'ns'}")

ax2.set_xlabel('Days from Index Time')
ax2.set_ylabel('AKI-Free Probability')
ax2.set_title(f'Survival by Vancomycin Exposure Group\n(log-rank p={result_exp.p_value:.4f})')
ax2.grid(alpha=0.3)
ax2.set_ylim([0, 1.05])

# Plot 3: By baseline creatinine (HIGH vs LOW)
ax3 = axes[1, 0]
creat_median = surv_data['baseline_creatinine'].median()
for idx, (high, label) in enumerate([(False, f'Low Cr (≤{creat_median:.1f})'), 
                                       (True, f'High Cr (>{creat_median:.1f})')]):
    mask = surv_data['baseline_creatinine'] > creat_median if high else surv_data['baseline_creatinine'] <= creat_median
    T_group = surv_data.loc[mask, 'T']
    E_group = surv_data.loc[mask, 'E']
    kmf.fit(T_group, E_group, label=f'{label} (N={mask.sum():,})')
    kmf.plot_survival_function(ax=ax3, color=colors[idx])

# Log-rank test: baseline creatinine
mask_low = surv_data['baseline_creatinine'] <= creat_median
mask_high = surv_data['baseline_creatinine'] > creat_median
result_cr = logrank_test(surv_data.loc[mask_low, 'T'], surv_data.loc[mask_high, 'T'],
                         surv_data.loc[mask_low, 'E'], surv_data.loc[mask_high, 'E'])
logrank_results.append(('Baseline Creatinine (Low vs High)', result_cr.p_value))
print(f"\nLog-Rank Test: Baseline Creatinine")
print(f"  p-value: {result_cr.p_value:.4f} {'***' if result_cr.p_value < 0.001 else '**' if result_cr.p_value < 0.01 else '*' if result_cr.p_value < 0.05 else 'ns'}")

ax3.set_xlabel('Days from Index Time')
ax3.set_ylabel('AKI-Free Probability')
ax3.set_title(f'Survival by Baseline Creatinine\n(log-rank p={result_cr.p_value:.4f})')
ax3.grid(alpha=0.3)
ax3.set_ylim([0, 1.05])

# Plot 4: By SOFA score (HIGH vs LOW)
ax4 = axes[1, 1]
sofa_median = surv_data['sofa_score'].median()
for idx, (high, label) in enumerate([(False, f'Low SOFA (≤{sofa_median:.0f})'), 
                                      (True, f'High SOFA (>{sofa_median:.0f})')]):
    mask = surv_data['sofa_score'] > sofa_median if high else surv_data['sofa_score'] <= sofa_median
    T_group = surv_data.loc[mask, 'T']
    E_group = surv_data.loc[mask, 'E']
    kmf.fit(T_group, E_group, label=f'{label} (N={mask.sum():,})')
    kmf.plot_survival_function(ax=ax4, color=colors[idx])

# Log-rank test: SOFA score
mask_low_sofa = surv_data['sofa_score'] <= sofa_median
mask_high_sofa = surv_data['sofa_score'] > sofa_median
result_sofa = logrank_test(surv_data.loc[mask_low_sofa, 'T'], surv_data.loc[mask_high_sofa, 'T'],
                           surv_data.loc[mask_low_sofa, 'E'], surv_data.loc[mask_high_sofa, 'E'])
logrank_results.append(('SOFA Score (Low vs High)', result_sofa.p_value))
print(f"\nLog-Rank Test: SOFA Score")
print(f"  p-value: {result_sofa.p_value:.4f} {'***' if result_sofa.p_value < 0.001 else '**' if result_sofa.p_value < 0.01 else '*' if result_sofa.p_value < 0.05 else 'ns'}")

ax4.set_xlabel('Days from Index Time')
ax4.set_ylabel('AKI-Free Probability')
ax4.set_title(f'Survival by SOFA Score\n(log-rank p={result_sofa.p_value:.4f})')
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 1.05])

plt.tight_layout()
km_fig_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_kaplan_meier.png"
km_fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(km_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Saved: {km_fig_path}")

# Save log-rank test results
logrank_df = pd.DataFrame(logrank_results, columns=['Comparison', 'p_value'])
logrank_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_logrank_tests.csv"
logrank_df.to_csv(logrank_path, index=False)
print(f" Saved log-rank test results: {logrank_path}")

# ###################################################################################
# COX PROPORTIONAL HAZARDS MODEL
# ###################################################################################

#
print("COX PROPORTIONAL HAZARDS REGRESSION")
#

# Prepare data for Cox model
cox_df = surv_data[['T', 'E'] + usable_cox_features].copy()

print(f"\nCox model sample size: {len(cox_df):,}")
print(f"Events (AKI): {cox_df['E'].sum():,}")
print(f"Censored (no AKI): {(1 - cox_df['E']).sum():,}")

# Standardize features for interpretability (optional but recommended)
scaler = StandardScaler()
cox_df_scaled = cox_df.copy()
cox_df_scaled[usable_cox_features] = scaler.fit_transform(cox_df[usable_cox_features])

# Fit Cox model on standardized features
cph = CoxPHFitter()
cph.fit(cox_df_scaled, duration_col='T', event_col='E', show_progress=False)

# Extract and display results
cox_summary = cph.summary[
    [
        'coef',
        'exp(coef)',
        'exp(coef) lower 95%',
        'exp(coef) upper 95%',
        'se(coef)',
        'p'
    ]
].copy()

cox_summary = cox_summary.sort_values('p')

print("\nCox Model Results:")
print(
    cox_summary[
        [
            'exp(coef)',
            'exp(coef) lower 95%',
            'exp(coef) upper 95%',
            'p'
        ]
    ].round(3)
)
# Identify significant predictors
sig_threshold = 0.05
sig_features = cox_summary[cox_summary['p'] < sig_threshold]
print(f"\nSignificant predictors (p < {sig_threshold}): {len(sig_features)}")
for feat, row in sig_features.iterrows():
    hr = row['exp(coef)']
    direction = "hazard up" if hr > 1 else "hazard down"
    print(f"  {feat:.<35} HR={hr:.3f} (p={row['p']:.4f}) {direction}")

# Plot hazard ratios for the Cox model
fig, ax = plt.subplots(figsize=(10, max(6, len(sig_features) * 0.35)))
cph.plot(ax=ax)
ax.set_title('RQ2 Cox Model: Hazard Ratios for Time-to-AKI\n(Standardized Features)')
ax.set_xlabel('Hazard Ratio')
plt.tight_layout()
cox_fig_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_cox_hazard_ratios.png"
plt.savefig(cox_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Saved: {cox_fig_path}")

# Save full Cox model results
cox_results_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_cox_full_results.csv"
cox_summary.reset_index().to_csv(cox_results_path, index=False)
print(f" Saved full Cox results: {cox_results_path}")

# Print model concordance (goodness-of-fit)
print(f"\nModel Concordance Index: {cph.concordance_index_:.4f}")
print(f"  (0.5 = random, 1.0 = perfect discrimination)")
print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(
    f"Concordance Index (C-index): "
    f"{cph.concordance_index_:.4f}"
)
performance_df = pd.DataFrame({
    "Metric": ["Concordance Index"],
    "Value": [cph.concordance_index_]
})

performance_path = (
    PROJECT_ROOT /
    "Analysis" /
    "results" /
    "rq2_model_performance.csv"
)

performance_df.to_csv(
    performance_path,
    index=False
)

print(f" Saved: {performance_path}")

# ###################################################################################
# AKI ONSET TIMING SUMMARY & VISUALIZATION
# ###################################################################################

#
print("AKI ONSET TIMING CHARACTERISTICS")
#

# Summary statistics by AKI onset category
timing_agg = {
    'age': ['count', 'mean'],
    'baseline_creatinine': 'mean',
    'charlson_index': 'mean',
    'sofa_score': 'mean',
    'exposure': 'mean',
    'T': 'mean'
}

# Keep only columns that exist
timing_agg = {col: agg for col, agg in timing_agg.items() if col in surv_data.columns}

if len(timing_agg) > 0:
    timing_stats = surv_data.groupby('AKI_Category').agg(timing_agg).round(2)
    print("\nCharacteristics by AKI Onset Timing:")
    print(timing_stats)
else:
    print(" No timing summary features available.")
print("\nCharacteristics by AKI Onset Timing:")
print(timing_stats)

# Create visualization of AKI onset timing distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ2: AKI Onset Timing Analysis', fontsize=14, fontweight='bold')

# Plot 1: Distribution of AKI onset categories
ax1 = axes[0]
category_order = ['No AKI (Censored)', 'Very Early AKI (0-1 day)', 'Early AKI (1-2 days)', 
                  'Mid AKI (2-4 days)', 'Late AKI (4-7 days)']
category_counts = surv_data['AKI_Category'].value_counts().reindex(category_order, fill_value=0)

colors_cat = ['#2ecc71', '#e74c3c', '#f39c12', '#e67e22', '#c0392b']
bars = ax1.bar(range(len(category_counts)), category_counts.values, color=colors_cat, alpha=0.8)
ax1.set_xticks(range(len(category_counts)))
ax1.set_xticklabels(category_counts.index, rotation=20, ha='right', fontsize=9)
ax1.set_ylabel('Number of Patients', fontsize=11)
ax1.set_title('Distribution of AKI Onset Timing')
ax1.grid(axis='y', alpha=0.3)

# Add count labels on bars
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(surv_data)*100:.1f}%)',
             ha='center', va='bottom', fontsize=9)

# Plot 2: Time-to-AKI distribution (AKI cases only)
ax2 = axes[1]
aki_times = surv_data[surv_data['E'] == 1]['T']
ax2.hist(aki_times, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
ax2.axvline(aki_times.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {aki_times.mean():.1f} days')
ax2.axvline(aki_times.median(), color='navy', linestyle='--', linewidth=2, label=f'Median: {aki_times.median():.1f} days')
ax2.set_xlabel('Time to AKI Onset (days)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title(f'Time-to-AKI Distribution (AKI cases, N={aki_times.shape[0]:,})')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
timing_fig_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_aki_onset_distribution.png"
plt.savefig(timing_fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\n Saved: {timing_fig_path}")

# Save timing summary table
timing_summary_path = PROJECT_ROOT / "Analysis" / "results" / "rq2_timing_summary.csv"
timing_stats.to_csv(timing_summary_path)
print(f" Saved: {timing_summary_path}")

# ###################################################################################
# FINAL SUMMARY & INTERPRETATION
# ###################################################################################

#
print("RQ2 TIME-TO-EVENT ANALYSIS COMPLETE")
#

print(f"""
 OUTPUTS GENERATED:
  1. Kaplan-Meier curves: {km_fig_path}
  2. Cox hazard ratios: {cox_fig_path}
  3. AKI onset timing: {timing_fig_path}
  4. Baseline characteristics: {char_output_path}
  5. Cox model results: {cox_results_path}
  6. Log-rank test results: {logrank_path}
  7. Timing summary table: {timing_summary_path}

 KEY FINDINGS:
  - Total patients analyzed: {len(surv_data):,}
  - AKI events: {surv_data['E'].sum():,} ({surv_data['E'].mean()*100:.1f}%)
  - Median time-to-AKI: {aki_times.median():.1f} days
  - Mean time-to-AKI: {aki_times.mean():.1f} days
  
 INTERPRETATION:
  - Use Kaplan-Meier curves to visualize AKI-free survival by risk groups
  - Log-rank tests show statistical significance of group differences
  - Cox hazard ratios identify factors associated with EARLIER AKI onset
  - Standardized coefficients allow direct comparison of feature importance
""")

#

print("\nChecking proportional hazards assumptions...")

cph.check_assumptions(
    cox_df_scaled,
    p_value_threshold=0.05,
    show_plots=False
)
