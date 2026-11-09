# Main writer: Aadya Charlewar
# Reviewer: Luca Dragutanu
# Contributors: Kimi Knaider

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
import re
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def load_table(name):
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

print("LOADING CLEANED DATA FROM PARQUET")

# Load all cleaned tables
chart_events = load_table("chartevents")
lab_events = load_table("labevents")
output_events = load_table("outputevents")
prescriptions = load_table("prescriptions")
emar = load_table("emar")
patients = load_table("patients")
admissions = load_table("admissions")
diagnoses_icd = load_table("diagnoses_icd")
icu_stays = load_table("icustays")
d_items = load_table("d_items")
d_lab_items = load_table("d_labitems")

print(f"Loaded {sum(1 for x in locals().values() if isinstance(x, pd.DataFrame))} tables")

print("TIME WINDOW PROCESSING")

# Get index time (first vancomycin administration)
vanco_pattern = r'vancomycin|vancocin|vanco'
vanco_times = []

# From emar (medication administration records)
if emar is not None:
    vanco_emar = emar[emar['medication'].str.contains(vanco_pattern, case=False, na=False)].copy()
    if not vanco_emar.empty:
        vanco_emar['admin_time'] = pd.to_datetime(vanco_emar['charttime'])
        vanco_times.append(vanco_emar[['subject_id', 'admin_time']])
        print(f"Found {len(vanco_emar)} vancomycin administrations in emar")

# From prescriptions (fallback)
if prescriptions is not None:
    vanco_rx = prescriptions[prescriptions['drug'].str.contains(vanco_pattern, case=False, na=False)].copy()
    if not vanco_rx.empty:
        vanco_rx['admin_time'] = pd.to_datetime(vanco_rx['starttime'])
        vanco_times.append(vanco_rx[['subject_id', 'admin_time']])
        print(f"Found {len(vanco_rx)} vancomycin prescriptions")

if vanco_times:
    index_times = pd.concat(vanco_times).sort_values('admin_time').groupby('subject_id').first().reset_index()
    index_times.columns = ['subject_id', 'index_time']
    print(f"Index times: {len(index_times)} patients")
else:
    print("No vancomycin found")
    index_times = pd.DataFrame()

# Create cohort with windows
if icu_stays is not None and not index_times.empty:
    cohort = icu_stays.merge(index_times, on='subject_id', how='inner')
    cohort['index_time'] = pd.to_datetime(cohort['index_time'])
    cohort['intime'] = pd.to_datetime(cohort['intime'])
    cohort['outtime'] = pd.to_datetime(cohort['outtime'])
    
    cohort = cohort[cohort['index_time'] >= cohort['intime']]
    cohort = cohort[cohort['index_time'] <= cohort['outtime']]
    
    cohort['pred_end'] = cohort['index_time'] + timedelta(hours=24)
    cohort['outcome_end'] = cohort['index_time'] + timedelta(days=7)
    print(f"Cohort: {len(cohort)} patients")
else:
    cohort = pd.DataFrame()

# Baseline creatinine
if lab_events is not None and not cohort.empty:
    creat_labs = lab_events[lab_events['itemid'] == 50912].copy()
    creat_labs['charttime'] = pd.to_datetime(creat_labs['charttime'])
    
    merged = creat_labs.merge(cohort[['subject_id', 'index_time']], on='subject_id')
    merged['hours_before'] = (merged['index_time'] - merged['charttime']).dt.total_seconds() / 3600
    merged = merged[(merged['hours_before'] >= 0) & (merged['hours_before'] <= 168)]
    
    baseline = merged.sort_values('hours_before').groupby('subject_id').first()
    cohort = cohort.merge(baseline[['valuenum']], left_on='subject_id', right_index=True, how='left')
    cohort.rename(columns={'valuenum': 'baseline_creatinine'}, inplace=True)
    print(f"Baseline creatinine: {cohort['baseline_creatinine'].notna().sum()} patients")

# Exclude patients with missing baseline
before = len(cohort)
cohort = cohort[cohort['baseline_creatinine'].notna()]
print(f"Excluded {before - len(cohort)} patients with missing baseline")

# AKI label
if lab_events is not None and not cohort.empty:
    creat_labs = lab_events[lab_events['itemid'] == 50912].copy()
    creat_labs['charttime'] = pd.to_datetime(creat_labs['charttime'])
    
    followup = creat_labs.merge(cohort[['subject_id', 'index_time', 'baseline_creatinine', 'outcome_end']], on='subject_id')
    followup = followup[followup['charttime'] >= followup['index_time']]
    followup = followup[followup['charttime'] <= followup['outcome_end']]
    followup['hours_after'] = (followup['charttime'] - followup['index_time']).dt.total_seconds() / 3600
    
    followup['criteria1'] = (followup['valuenum'] - followup['baseline_creatinine'] >= 0.3) & (followup['hours_after'] <= 48)
    followup['criteria2'] = (followup['valuenum'] / followup['baseline_creatinine'] >= 1.5) & (followup['hours_after'] <= 168)
    followup['aki'] = (followup['criteria1'] | followup['criteria2']).astype(int)
    
    aki = followup.groupby('subject_id')['aki'].max().reset_index()
    aki.columns = ['subject_id', 'aki_day7']
    cohort = cohort.merge(aki, on='subject_id', how='left')
    cohort['aki_day7'] = cohort['aki_day7'].fillna(0)
    print(f"AKI rate: {cohort['aki_day7'].sum()}/{len(cohort)} ({cohort['aki_day7'].mean()*100:.1f}%)")

# VECTORIZED FIRST 24H FEATURES (FAST)

if chart_events is not None and output_events is not None and lab_events is not None and not cohort.empty:
    print("\nExtracting first 24h features (vectorized)...")
    
    # Prepare cohort with time windows
    cohort_subset = cohort[['subject_id', 'index_time', 'pred_end']].copy()
    
    #  VITALS 
    print("  Processing vitals...")
    chart = chart_events.copy()
    chart['charttime'] = pd.to_datetime(chart['charttime'])
    
    # Merge with cohort to get windows
    chart = chart.merge(cohort_subset, on='subject_id', how='inner')
    chart = chart[(chart['charttime'] >= chart['index_time']) & (chart['charttime'] <= chart['pred_end'])]
    
    # Filter relevant vitals
    vital_itemids = [220045, 220179, 220180, 220181, 223761, 220277, 220210]
    chart = chart[chart['itemid'].isin(vital_itemids)]
    
    # Group by patient and itemid to get min, max, mean
    vitals_summary = chart.groupby(['subject_id', 'itemid'])['valuenum'].agg(['min', 'max', 'mean']).reset_index()
    
    # Pivot to get one row per patient
    vital_map_name = {220045: 'hr', 220179: 'sbp', 220180: 'dbp', 220181: 'map', 223761: 'temp', 220277: 'spo2', 220210: 'rr'}
    vital_summary_pivot = vitals_summary.pivot(index='subject_id', columns='itemid', values=['min', 'max', 'mean'])
    vital_summary_pivot.columns = [f'{vital_map_name[col[1]]}_{col[0]}' for col in vital_summary_pivot.columns]
    vital_summary_pivot = vital_summary_pivot.reset_index()
    
    # Merge back to cohort
    cohort = cohort.merge(vital_summary_pivot, on='subject_id', how='left')
    print(f"    Vitals done")
    
    #  URINE 
    print("  Processing urine output...")
    urine = output_events.copy()
    urine['charttime'] = pd.to_datetime(urine['charttime'])
    
    urine = urine.merge(cohort_subset, on='subject_id', how='inner')
    urine = urine[(urine['charttime'] >= urine['index_time']) & (urine['charttime'] <= urine['pred_end'])]
    
    urine_summary = urine.groupby('subject_id')['value'].sum().reset_index()
    urine_summary.columns = ['subject_id', 'urine_total_24h']
    
    cohort = cohort.merge(urine_summary, on='subject_id', how='left')
    print(f"    Urine done")
    
    #  LABS 
    print("  Processing labs (first value)...")
    labs = lab_events.copy()
    labs['charttime'] = pd.to_datetime(labs['charttime'])
    
    labs = labs.merge(cohort_subset, on='subject_id', how='inner')
    labs = labs[(labs['charttime'] >= labs['index_time']) & (labs['charttime'] <= labs['pred_end'])]
    
    # Get first value per patient per lab
    labs_sorted = labs.sort_values('charttime')
    first_labs = labs_sorted.groupby(['subject_id', 'itemid']).first().reset_index()
    
    # Pivot to one row per patient
    lab_map_name = {50912: 'creatinine', 51006: 'bun', 51301: 'wbc', 51265: 'platelets', 
                    50813: 'lactate', 50971: 'potassium', 50983: 'sodium', 50882: 'bicarbonate', 
                    50931: 'glucose'}
    first_labs = first_labs[first_labs['itemid'].isin(lab_map_name.keys())]
    
    labs_pivot = first_labs.pivot(index='subject_id', columns='itemid', values='valuenum')
    labs_pivot.columns = [f'lab_{lab_map_name[col]}_first' for col in labs_pivot.columns]
    labs_pivot = labs_pivot.reset_index()
    
    cohort = cohort.merge(labs_pivot, on='subject_id', how='left')
    print(f"    Labs done")
    
    print("First 24h features extracted")

# Final missing handling for labs
lab_cols = [c for c in cohort.columns if c.startswith('lab_') and c.endswith('_first')]
for col in lab_cols:
    median_val = cohort[col].median()
    cohort[f'{col}_missing'] = cohort[col].isnull().astype(int)
    cohort[col] = cohort[col].fillna(median_val)
    print(f"  {col}: median imputation")

print(f"\nFinal cohort shape: {cohort.shape}")

# STANDARDISATION FUNCTIONS

def to_snake_case(name: str) -> str:
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
    name = name.lower().replace(' ', '_').replace('-', '_')
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [to_snake_case(c) for c in df.columns]
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols
    return df

RENAME_MAP = {
    'stay_id': 'icu_stay_id',
    'hadm_id': 'hospital_admission_id',
    'subject_id': 'patient_id',
    'anchor_age': 'age_years',
    'gender': 'sex',
    'intime': 'icu_admit_time',
    'outtime': 'icu_discharge_time',
    'index_time': 'vanco_index_time',
    'pred_end': 'prediction_window_end',
    'outcome_end': 'outcome_window_end',
    'baseline_creatinine': 'baseline_creatinine_mgdl',
    'aki_day7': 'aki_label_day7',
    'hr_min': 'heart_rate_min',
    'hr_max': 'heart_rate_max',
    'hr_mean': 'heart_rate_mean',
    'sbp_min': 'systolic_bp_min',
    'sbp_max': 'systolic_bp_max',
    'sbp_mean': 'systolic_bp_mean',
    'dbp_min': 'diastolic_bp_min',
    'dbp_max': 'diastolic_bp_max',
    'dbp_mean': 'diastolic_bp_mean',
    'map_min': 'mean_arterial_pressure_min',
    'map_max': 'mean_arterial_pressure_max',
    'map_mean': 'mean_arterial_pressure_mean',
    'temp_min': 'temperature_c_min',
    'temp_max': 'temperature_c_max',
    'temp_mean': 'temperature_c_mean',
    'spo2_min': 'spo2_pct_min',
    'spo2_max': 'spo2_pct_max',
    'spo2_mean': 'spo2_pct_mean',
    'rr_min': 'resp_rate_min',
    'rr_max': 'resp_rate_max',
    'rr_mean': 'resp_rate_mean',
    'lab_creatinine_first': 'lab_creatinine_first_mgdl',
    'lab_bun_first': 'lab_bun_first_mgdl',
    'lab_wbc_first': 'lab_wbc_first_k_ul',
    'lab_platelets_first': 'lab_platelets_first_k_ul',
    'lab_lactate_first': 'lab_lactate_first_mmol_l',
    'lab_potassium_first': 'lab_potassium_first_meq_l',
    'lab_sodium_first': 'lab_sodium_first_meq_l',
    'lab_bicarbonate_first': 'lab_bicarbonate_first_meq_l',
    'lab_glucose_first': 'lab_glucose_first_mgdl',
    'urine_total_24h': 'urine_output_24h_ml',
}

def convert_temp_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    temp_cols = [c for c in df.columns if c.startswith('temperature_c_')]
    for col in temp_cols:
        if df[col].notna().any():
            df[col] = (df[col] - 32) * 5 / 9
    if temp_cols:
        print(f"  Temperature converted to °C for: {temp_cols}")
    return df

def standardise_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt_cols = df.select_dtypes(include=['datetime64', 'object']).columns
    for col in dt_cols:
        if df[col].dtype == object:
            sample = df[col].dropna().head(10)
            if sample.apply(lambda x: isinstance(x, str) and
                            any(c in x for c in ['-', '/']) and
                            any(c.isdigit() for c in x)).all():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass
        if hasattr(df[col], 'dt') and hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    return df

BINARY_COLS = ['age_ge65', 'hospital_mortality', 'age_missing', 'aki_label_day7']
COMORBIDITY_COLS = ['ckd', 'diabetes', 'hypertension', 'heart_failure', 'ischemic_heart_disease', 'sepsis', 'arrhythmia', 'peripheral_vascular_disease', 'liver_disease', 'copd', 'malignancy']
EXPOSURE_COLS = ['is_vancomycin', 'is_pip_tazo', 'is_aminoglycoside', 'is_loop_diuretic', 'vasopressor_use', 'mechanical_ventilation']
ONEHOT_COLS = ['admission_type', 'insurance', 'marital_status', 'race']
SEX_MAP = {'M': 1, 'F': 0, 'm': 1, 'f': 0, 'Male': 1, 'Female': 0}

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in BINARY_COLS + COMORBIDITY_COLS + EXPOSURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    if 'sex' in df.columns:
        df['sex_binary'] = df['sex'].map(SEX_MAP)
    print(f"  Binary/comorbidity/exposure flags encoded as int")
    return df

def onehot_encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in ONEHOT_COLS if c in df.columns]
    if not present:
        return df
    dummies = pd.get_dummies(df[present], prefix=present, drop_first=False, dtype=int)
    dummies = dummies[[c for c in dummies.columns if 'unknown' not in c.lower()]]
    df = pd.concat([df, dummies], axis=1)
    print(f"  One-hot encoded: {present} → {list(dummies.columns)}")
    return df

DO_NOT_SCALE = (
    BINARY_COLS + COMORBIDITY_COLS + EXPOSURE_COLS +
    ['patient_id', 'hospital_admission_id', 'icu_stay_id',
     'vanco_index_time', 'prediction_window_end', 'outcome_window_end',
     'icu_admit_time', 'icu_discharge_time',
     'aki_label_day7', 'time_to_aki_hours',
     'sex_binary', 'age_ge65', 'age_missing']
)
DO_NOT_SCALE_PATTERNS = ['_missing', '_binary', '_label', '_flag']

def get_scale_cols(df: pd.DataFrame) -> list:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = set(DO_NOT_SCALE)
    result = []
    for col in numeric:
        if col in skip:
            continue
        if any(pat in col for pat in DO_NOT_SCALE_PATTERNS):
            continue
        result.append(col)
    return result

def scale_features(df: pd.DataFrame, scaler=None, fit: bool = True):
    df = df.copy()
    cols_to_scale = get_scale_cols(df)
    cols_to_scale = [c for c in cols_to_scale if c in df.columns and df[c].std() > 0]
    if not cols_to_scale:
        print("  No columns to scale.")
        return df, scaler
    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].values)
        print(f"  StandardScaler fitted and applied to {len(cols_to_scale)} columns")
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        df[cols_to_scale] = scaler.transform(df[cols_to_scale].values)
        print(f"  Pre-fitted StandardScaler applied to {len(cols_to_scale)} columns")
    return df, scaler

def run_standardisation(cohort: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STANDARDISATION PIPELINE")
    print("=" * 60)
    
    print("\n Standardising column names to snake_case...")
    cohort = standardise_column_names(cohort)
    
    print("Applying explicit rename map...")
    rename_present = {k: v for k, v in RENAME_MAP.items() if k in cohort.columns}
    cohort = cohort.rename(columns=rename_present)
    print(f"    Renamed {len(rename_present)} columns")
    
    print("Converting temperature to Celsius...")
    cohort = convert_temp_to_celsius(cohort)
    
    print(" Standardising datetime columns...")
    cohort = standardise_datetimes(cohort)
    
    print("Encoding categorical variables...")
    cohort = encode_categoricals(cohort)
    cohort = onehot_encode(cohort)
    
    cohort_tree = cohort.copy()
    print("\n    cohort_tree (unscaled, for XGBoost/RF) saved.")
    
    print(" Scaling numeric features (StandardScaler)...")
    cohort_std, scaler = scale_features(cohort, fit=True)
    
    print("STANDARDISATION COMPLETE")
    print(f"  cohort_std  shape : {cohort_std.shape}")
    print(f"  cohort_tree shape : {cohort_tree.shape}")
    print(f"  Remaining NaNs (cohort_std)  : {cohort_std.isnull().sum().sum()}")
    print(f"  Remaining NaNs (cohort_tree) : {cohort_tree.isnull().sum().sum()}")
    
    if 'aki_label_day7' in cohort_std.columns:
        print(f"\n  AKI prevalence (cohort_std): {cohort_std['aki_label_day7'].sum()}/{len(cohort_std)} ({cohort_std['aki_label_day7'].mean()*100:.1f}%)")
    else:
        print("\n  WARNING: aki_label_day7 not found — check time_windows.py output")
    
    return cohort_std, cohort_tree, scaler

#ADD THIS TO GIT   sjgkhsdjkhgd comment to add to git
if 'age_years' in cohort.columns:
    cohort['age_ge65'] = (cohort['age_years'] >= 65).astype(int)
    cohort['age_missing'] = cohort['age_years'].isnull().astype(int)

# Baseline creatinine missing flag
cohort['baseline_creatinine_missing'] = cohort['baseline_creatinine'].isnull().astype(int)

cohort_std, cohort_tree, scaler = run_standardisation(cohort)

# Save final cohorts
cohort_std.to_parquet(PROCESSED_DIR / "cohort_std.parquet", index=False)
cohort_tree.to_parquet(PROCESSED_DIR / "cohort_tree.parquet", index=False)
print(f"\nSaved cohorts to {PROCESSED_DIR}")
