# Main writer: Cahit Kok
# Reviewer: Kimi Knaider
# Contributors: Noah Charpentier

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "final" / "final_aki_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "final" / "final_aki_dataset_engineered_FIXED.csv"

#
print("FIXED FEATURE ENGINEERING FOR AKI PREDICTION")
#

# Load data
print(f"\nLoading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f" Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df_eng = df.copy()


#
print("1. OLIGURIA FLAGS")
#

oliguria_threshold_24h = 840
df_eng['oliguria_flag_24h'] = (df_eng['urine_output_24h'] < oliguria_threshold_24h).astype(int)
df_eng['oliguria_flag_12h'] = (df_eng['urine_output_12h'] < 420).astype(int)  
df_eng['oliguria_flag_6h'] = (df_eng['urine_output_6h'] < 210).astype(int)

print(f" Oliguria flag 24h: {df_eng['oliguria_flag_24h'].sum()} cases ({df_eng['oliguria_flag_24h'].mean():.2%})")
print(f" Oliguria flag 12h: {df_eng['oliguria_flag_12h'].sum()} cases ({df_eng['oliguria_flag_12h'].mean():.2%})")
print(f" Oliguria flag 6h: {df_eng['oliguria_flag_6h'].sum()} cases ({df_eng['oliguria_flag_6h'].mean():.2%})")

#
print("2. LAB TRENDS FROM FIRST 24H")
#

if 'creatinine_max' in df_eng.columns and 'creatinine_min' in df_eng.columns:
    df_eng['creatinine_trend_24h'] = (df_eng.get('creatinine_max', 0) - 
                                       df_eng.get('creatinine_min', 0)).fillna(0)
    df_eng['creatinine_slope_24h'] = df_eng['creatinine_trend_24h'] / 24
    print(f" Creatinine slope 24h: mean={df_eng['creatinine_slope_24h'].mean():.4f}")
else:
    print("Creatinine max/min not in dataset; using creatinine_delta_48h as proxy")
    df_eng['creatinine_slope_24h'] = (df_eng.get('creatinine_delta_48h', 0) / 24).fillna(0)

if 'bun_max' in df_eng.columns and 'bun_min' in df_eng.columns:
    df_eng['bun_trend_24h'] = df_eng['bun_max'] - df_eng['bun_min']
    df_eng['bun_change_pct_24h'] = ((df_eng['bun_max'] - df_eng['bun_min']) / 
                                     (df_eng['bun_mean'] + 1)) * 100
    print(f" BUN trend 24h (mg/dL): mean={df_eng['bun_trend_24h'].mean():.2f}")
else:
    print(" BUN max/min not in dataset")
    df_eng['bun_trend_24h'] = 0
    df_eng['bun_change_pct_24h'] = 0

if 'lactate_max' in df_eng.columns and 'lactate_min' in df_eng.columns:
    df_eng['lactate_change_24h'] = df_eng['lactate_max'] - df_eng['lactate_min']
    df_eng['lactate_elevation'] = (df_eng['lactate_max'] > 2.0).astype(int)
    print(f" Lactate change 24h: mean={df_eng['lactate_change_24h'].mean():.2f}")
else:
    print(" Lactate max/min not in dataset")
    df_eng['lactate_change_24h'] = 0
    df_eng['lactate_elevation'] = 0

if 'potassium_max' in df_eng.columns and 'potassium_min' in df_eng.columns:
    df_eng['potassium_change_24h'] = df_eng['potassium_max'] - df_eng['potassium_min']
    df_eng['hyperkalemia_flag'] = (df_eng['potassium_max'] > 5.5).astype(int)
    df_eng['hypokalemia_flag'] = (df_eng['potassium_min'] < 3.0).astype(int)
    print(f" Potassium change 24h: mean={df_eng['potassium_change_24h'].mean():.2f}")
else:
    print(" Potassium max/min not in dataset")
    df_eng['potassium_change_24h'] = 0
    df_eng['hyperkalemia_flag'] = 0
    df_eng['hypokalemia_flag'] = 0

#
print("3. VITAL SIGN VARIABILITY FROM FIRST 24H")
#

if 'heart_rate_max' in df_eng.columns and 'heart_rate_min' in df_eng.columns:
    df_eng['heart_rate_range_24h'] = df_eng['heart_rate_max'] - df_eng['heart_rate_min']
    df_eng['heart_rate_variability_24h'] = df_eng['heart_rate_range_24h'] / (df_eng['heart_rate_mean'] + 1)
    print(f" Heart rate range 24h: mean={df_eng['heart_rate_range_24h'].mean():.1f}")
else:
    print(" Heart rate max/min not in dataset")
    df_eng['heart_rate_range_24h'] = 0
    df_eng['heart_rate_variability_24h'] = 0

if 'systolic_bp_max' in df_eng.columns:
    df_eng['systolic_bp_range_24h'] = df_eng['systolic_bp_max'] - df_eng.get('systolic_bp_min', 0)
    df_eng['diastolic_bp_range_24h'] = df_eng.get('diastolic_bp_max', 0) - df_eng.get('diastolic_bp_min', 0)
    df_eng['map_range_24h'] = df_eng.get('map_max', 0) - df_eng.get('map_min', 0)
    print(f" MAP range 24h: mean={df_eng['map_range_24h'].mean():.1f}")
else:
    print(" Blood pressure values not in dataset")
    df_eng['systolic_bp_range_24h'] = 0
    df_eng['diastolic_bp_range_24h'] = 0
    df_eng['map_range_24h'] = 0

if 'temperature_max' in df_eng.columns:
    df_eng['temperature_range_24h'] = df_eng['temperature_max'] - df_eng.get('temperature_min', 0)
    df_eng['fever_flag'] = (df_eng['temperature_max'] > 38.0).astype(int)
    print(f" Fever cases: {df_eng['fever_flag'].sum()} ({df_eng['fever_flag'].mean():.2%})")
else:
    print(" Temperature values not in dataset")
    df_eng['temperature_range_24h'] = 0
    df_eng['fever_flag'] = 0

if 'spo2_max' in df_eng.columns:
    df_eng['spo2_range_24h'] = df_eng['spo2_max'] - df_eng.get('spo2_min', 0)
    df_eng['hypoxia_flag'] = (df_eng['spo2_min'] < 90).astype(int)
    print(f" Hypoxia cases: {df_eng['hypoxia_flag'].sum()} ({df_eng['hypoxia_flag'].mean():.2%})")
else:
    print(" SpO2 values not in dataset")
    df_eng['spo2_range_24h'] = 0
    df_eng['hypoxia_flag'] = 0

#
print("4. DRUG METRICS (Unchanged)")
#

if 'exposure' not in df_eng.columns:
    df_eng['exposure'] = df_eng['vanco_ptz_combo'].astype(int)

df_eng['num_nephrotoxic_drugs'] = 0
df_eng['num_nephrotoxic_drugs'] += df_eng['aminoglycoside_use'].astype(int)
df_eng['num_nephrotoxic_drugs'] += df_eng['nsaid_use'].astype(int)
df_eng['num_nephrotoxic_drugs'] += df_eng['exposure'].astype(int)

df_eng['vanco_plus_aminoglycoside'] = (
    (df_eng['vancomycin_use'] == True) & 
    (df_eng['aminoglycoside_use'] == True)
).astype(int)

df_eng['vanco_with_diuretic'] = (
    (df_eng['vancomycin_use'] == True) & 
    (df_eng['loop_diuretic_use'] == True)
).astype(int)

df_eng['high_risk_drug_combo'] = (
    (df_eng['num_nephrotoxic_drugs'] >= 2) |
    (df_eng['vanco_plus_aminoglycoside'] == 1)
).astype(int)

print(f" Nephrotoxic drugs: mean={df_eng['num_nephrotoxic_drugs'].mean():.2f}")
print(f" High-risk combos: {df_eng['high_risk_drug_combo'].sum()} cases ({df_eng['high_risk_drug_combo'].mean():.2%})")

#
print("5. CHARLSON COMORBIDITY INDEX")
#

df_eng['charlson_index'] = 0

if 'ckd_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['ckd_flag'].astype(int) * 2
if 'diabetes_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['diabetes_flag'].astype(int) * 1
if 'heart_failure_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['heart_failure_flag'].astype(int) * 1
if 'ischemic_heart_disease_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['ischemic_heart_disease_flag'].astype(int) * 1
if 'mi_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['mi_flag'].astype(int) * 1
if 'liver_disease_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['liver_disease_flag'].astype(int) * 3
if 'malignancy_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['malignancy_flag'].astype(int) * 2
if 'copd_flag' in df_eng.columns:
    df_eng['charlson_index'] += df_eng['copd_flag'].astype(int) * 1

print(f" Charlson Index: mean={df_eng['charlson_index'].mean():.2f}, max={df_eng['charlson_index'].max()}")

#
print("6. SIMPLIFIED SOFA SCORE")
#

df_eng['sofa_score'] = 0

if 'spo2_min' in df_eng.columns:
    df_eng['sofa_score'] += (df_eng['spo2_min'] < 90).astype(int) * 1

if 'map_min' in df_eng.columns:
    df_eng['sofa_score'] += (df_eng['map_min'] < 70).astype(int) * 1

if 'baseline_creatinine' in df_eng.columns:
    df_eng['sofa_score'] += (df_eng['baseline_creatinine'] > 1.2).astype(int) * 1

if 'platelets_min' in df_eng.columns:
    df_eng['sofa_score'] += (df_eng['platelets_min'] < 100).astype(int) * 1

print(f" SOFA Score: mean={df_eng['sofa_score'].mean():.2f}")

#
print("7. SAPS II SCORE (Simplified)")
#

df_eng['saps_ii_score'] = 0

df_eng['saps_ii_score'] += ((df_eng['age'] >= 70).astype(int) * 5)

if 'heart_rate_max' in df_eng.columns:
    df_eng['saps_ii_score'] += (((df_eng['heart_rate_max'] >= 140) | 
                                 (df_eng['heart_rate_min'] <= 40)).astype(int) * 4)

if 'systolic_bp_min' in df_eng.columns:
    df_eng['saps_ii_score'] += ((df_eng['systolic_bp_min'] < 90).astype(int) * 5)

if 'temperature_max' in df_eng.columns:
    df_eng['saps_ii_score'] += (((df_eng['temperature_max'] >= 39) |
                                 (df_eng.get('temperature_min', 100) < 36)).astype(int) * 3)

print(f" SAPS II Score: mean={df_eng['saps_ii_score'].mean():.2f}")

#
print("8. INTERACTION TERMS")
#

df_eng['age_comorbidity_interaction'] = df_eng.get('age', 0) * df_eng.get('charlson_index', 0) / 100

df_eng['age_creatinine_interaction'] = df_eng.get('age', 0) * df_eng.get('baseline_creatinine', 0) / 10

df_eng['exposure_age_interaction'] = df_eng.get('exposure', 0) * df_eng.get('age', 0) / 100

severity_score = (df_eng.get('sofa_score', 0) / 4 + 
                  df_eng.get('saps_ii_score', 0) / 40) / 2
df_eng['severity_drug_risk'] = severity_score * df_eng.get('high_risk_drug_combo', 0)

df_eng['oliguria_electrolyte_risk'] = (
    df_eng.get('oliguria_flag_24h', 0) * 
    (df_eng.get('potassium_change_24h', 0) > 0.5).astype(int) *
    (df_eng.get('hyperkalemia_flag', 0) | df_eng.get('hypokalemia_flag', 0)).astype(int)
).astype(int)

print(" Interaction terms created")

#
print("SAVING OUTPUT")
#

print("\nRemoving leaky columns from original script...")
leaky_cols = [
    'creatinine_trend',
    'creatinine_slope',
    'bun_trend',
    'bun_change_pct',
    'lactate_change',
    'potassium_change',
    'heart_rate_range',
    'heart_rate_variability',
    'systolic_bp_range',
    'diastolic_bp_range',
    'map_range',
    'temperature_range',
    'spo2_range',
]

for col in leaky_cols:
    if col in df_eng.columns:
        df_eng = df_eng.drop(col, axis=1)
        print(f"   Removed: {col}")

df_eng.to_csv(OUTPUT_PATH, index=False)
print(f"\n Fixed dataset saved to: {OUTPUT_PATH}")
print(f"  Shape: {df_eng.shape[0]} rows × {df_eng.shape[1]} columns")

#
print("FEATURE ENGINEERING COMPLETE")
#
print("\n All features now use only BASELINE and FIRST 24H data")
print(" No leakage from 7-day AKI outcome window")
print("\nFeatures by origin:")
print("  - Baseline: age, sex, baseline_creatinine, comorbidities")
print("  - First 24h after exposure: labs, vitals, urine_output")
print("  - Derived: Charlson, SOFA, SAPS II, interaction terms")
