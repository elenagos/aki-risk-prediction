# Main writer: Aadya Charlewar
# Reviewer: Luca Dragutanu
# Contributors: Kimi Knaider

import pandas as pd
import numpy as np
from pathlib import Path

#Just for peace of mind 

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "mimic-iv-3.1"

def load_table(name):
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

def load_raw_gz(relative_path):
    path = RAW_DATA_DIR / relative_path
    if path.exists():
        return pd.read_csv(path, compression='gzip', nrows=1000000)  # Sample 1M rows for speed
    return None

#
print("DATA CLEANING VALIDATION REPORT")
#

# ============================================================================
# 1. TABLE RETENTION CHECK
# ============================================================================

print("\n1. TABLE RETENTION")
print("-"*50)

parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
print(f"Tables saved to parquet: {len(parquet_files)}")
for f in sorted(parquet_files):
    size_mb = f.stat().st_size / 1024 / 1024
    df = pd.read_parquet(f)
    print(f"  {f.stem}: {len(df):,} rows, {len(df.columns)} cols, {size_mb:.1f} MB")

# ============================================================================
# 2. CRITICAL TABLE COMPARISON (Cleaned vs Raw)
# ============================================================================

print("\n2. CRITICAL TABLE COMPARISON (Cleaned vs Raw Sample)")
print("-"*50)

tables_to_compare = ['patients', 'admissions', 'labevents', 'chartevents', 'prescriptions']

for table in tables_to_compare:
    cleaned = load_table(table)
    if cleaned is None:
        print(f"  {table}: cleaned not found")
        continue
    
    # Try to load raw (use the correct path)
    if table == 'labevents':
        raw_path = "hosp/labevents.csv.gz"
    elif table == 'chartevents':
        raw_path = "icu/chartevents.csv.gz"
    elif table == 'prescriptions':
        raw_path = "hosp/prescriptions.csv.gz"
    elif table == 'patients':
        raw_path = "hosp/patients.csv.gz"
    elif table == 'admissions':
        raw_path = "hosp/admissions.csv.gz"
    else:
        continue
    
    raw = load_raw_gz(raw_path)
    
    print(f"\n  {table.upper()}:")
    print(f"    Raw (sample): {len(raw):,} rows, {len(raw.columns)} cols" if raw is not None else "    Raw: not loaded")
    print(f"    Cleaned: {len(cleaned):,} rows, {len(cleaned.columns)} cols")
    
    if raw is not None:
        reduction = (1 - len(cleaned) / len(raw)) * 100 if len(raw) > 0 else 0
        print(f"    Row reduction: {reduction:.1f}% (expected from filtering)")

# ============================================================================
# 3. MISSING VALUE COMPARISON
# ============================================================================

print("\n3. MISSING VALUE ANALYSIS")
print("-"*50)

def missing_report(df, name):
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    return missing_pct

print("\n  Missing percentages in cleaned tables:")
for table in parquet_files:
    df = pd.read_parquet(table)
    missing_pct = missing_report(df, table.stem)
    if missing_pct > 0:
        print(f"    {table.stem}: {missing_pct:.1f}% missing")
    else:
        print(f"    {table.stem}: 0.0% missing")

# ============================================================================
# 4. VALUE RANGE VALIDATION (Clinically Plausible)
# ============================================================================

print("\n4. CLINICAL VALUE RANGE VALIDATION")
print("-"*50)

# Check cleaned chart_events
chart = load_table("chartevents")
if chart is not None:
    print("\n  Chart Events (vitals):")
    vital_ranges = {
        'hr': (20, 250),
        'sbp': (40, 250),
        'dbp': (20, 150),
        'map': (30, 200),
        'temp': (86, 107.6),
        'spo2': (50, 100),
        'rr': (5, 60)
    }
    
    for vitals in vital_ranges:
        cols = [c for c in chart.columns if c.startswith(vitals)]
        if cols:
            for col in cols:
                min_val = chart[col].min()
                max_val = chart[col].max()
                expected_min, expected_max = vital_ranges[vitals]
                in_range = (min_val >= expected_min - 5) and (max_val <= expected_max + 5)
                status = "good" if in_range else "not good"
                print(f"    {status} {col}: {min_val:.1f} - {max_val:.1f} (expected {expected_min}-{expected_max})")

# Check cleaned lab_events
labs = load_table("labevents")
if labs is not None:
    print("\n  Lab Events:")
    lab_ranges = {
        'creatinine': (0.2, 15),
        'bun': (1, 200),
        'wbc': (0, 100),
        'platelets': (10, 1000),
        'lactate': (0, 20),
        'potassium': (1, 10),
        'sodium': (100, 180),
        'bicarbonate': (10, 50),
        'glucose': (30, 2000)
    }
    
    for lab, (expected_min, expected_max) in lab_ranges.items():
        cols = [c for c in labs.columns if lab in c.lower()]
        if cols:
            for col in cols:
                min_val = labs[col].min()
                max_val = labs[col].max()
                if pd.notna(min_val) and pd.notna(max_val):
                    status = "✓" if (min_val >= expected_min - 5 and max_val <= expected_max + 50) else "⚠"
                    print(f"    {status} {col}: {min_val:.2f} - {max_val:.2f} (expected {expected_min}-{expected_max})")

# Check urine output
output = load_table("outputevents")
if output is not None:
    print("\n  Urine Output:")
    if 'urine_total_24h' in output.columns:
        min_val = output['urine_total_24h'].min()
        max_val = output['urine_total_24h'].max()
        print(f"     urine_total_24h: {min_val:.0f} - {max_val:.0f} ml (expected 0-5000)")

# ============================================================================
# 5. EXCLUSION CRITERIA VERIFICATION
# ============================================================================

print("\n5. EXCLUSION CRITERIA VERIFICATION")
print("-"*50)

patients = load_table("patients")
if patients is not None:
    print(f"\n  Patients:")
    if 'age_years' in patients.columns:
        min_age = patients['age_years'].min()
        max_age = patients['age_years'].max()
        print(f"    Age range: {min_age:.0f} - {max_age:.0f} years (should be ≥18)")
        if min_age >= 18:
            print("     Age < 18 excluded")
        else:
            print("     Age < 18 still present")

icu = load_table("icustays")
if icu is not None:
    print(f"\n  ICU Stays:")
    if 'icu_discharge_time' in icu.columns and 'icu_admit_time' in icu.columns:
        icu['los_hours'] = (icu['icu_discharge_time'] - icu['icu_admit_time']).dt.total_seconds() / 3600
        min_los = icu['los_hours'].min()
        print(f"    ICU LOS range: {min_los:.1f} - {icu['los_hours'].max():.1f} hours")
        if min_los >= 24:
            print("     ICU stay < 24h excluded")
        else:
            print(f"     ICU stays < 24h still present (min: {min_los:.1f}h)")

# ============================================================================
# 6. COHORT STATISTICS (from more_handling.py output)
# ============================================================================

print("\n6. COHORT STATISTICS")
print("-"*50)

# Load final cohort if exists
cohort_std = load_table("cohort_std")
if cohort_std is not None:
    print(f"\n  Final cohort shape: {cohort_std.shape}")
    
    if 'aki_label_day7' in cohort_std.columns:
        aki_count = cohort_std['aki_label_day7'].sum()
        aki_pct = (aki_count / len(cohort_std)) * 100
        print(f"  AKI prevalence: {aki_count}/{len(cohort_std)} ({aki_pct:.1f}%)")
    
    # Check for any impossible values in final cohort
    numeric_cols = cohort_std.select_dtypes(include=[np.number]).columns
    negative_counts = {}
    for col in numeric_cols:
        if 'patient_id' not in col and 'id' not in col.lower():
            neg_count = (cohort_std[col] < 0).sum()
            if neg_count > 0:
                negative_counts[col] = neg_count
    
    if negative_counts:
        print(f"\n   Negative values found in {len(negative_counts)} columns:")
        for col, count in list(negative_counts.items())[:5]:
            print(f"      {col}: {count} rows")
    else:
        print(f"\n   No negative values in numeric columns")

# ============================================================================
# 7. MISSING FLAGS VERIFICATION
# ============================================================================

print("\n7. MISSING FLAGS VERIFICATION")
print("-"*50)

cohort_tree = load_table("cohort_tree")
if cohort_tree is not None:
    missing_flag_cols = [c for c in cohort_tree.columns if '_missing' in c]
    if missing_flag_cols:
        print(f"\n  Missing flags created: {len(missing_flag_cols)}")
        for col in missing_flag_cols[:10]:
            flag_sum = cohort_tree[col].sum()
            flag_pct = (flag_sum / len(cohort_tree)) * 100
            print(f"    {col}: {flag_sum:.0f} ({flag_pct:.1f}%) flagged as missing")
    else:
        print("   No missing flags found")

#
print("VALIDATION COMPLETE")
#
