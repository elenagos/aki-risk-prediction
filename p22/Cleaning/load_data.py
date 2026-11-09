# Main writer: Aadya Charlewar
# Reviewer:  Bartlomiej Kaczor
# Contributors: Kimi Knaider

import pandas as pd
import gzip
from pathlib import Path
import time
import os

# =========================
# DYNAMIC PATH DETECTION
# =========================

def find_project_root(start_path):
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "Cleaning").exists():
            return parent
        if (parent / "data").exists() or (parent / "Data").exists():
            return parent
    return Path(start_path).parent

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)

possible_data_names = ['data', 'Data', 'DATA']
possible_mimic_names = ['mimic-iv-3.1', 'mimic-iv', 'mimic-iv-full']

DATA_DIR = None
for data_name in possible_data_names:
    for mimic_name in possible_mimic_names:
        test_path = PROJECT_ROOT / data_name / mimic_name
        if test_path.exists():
            DATA_DIR = test_path
            break
    if DATA_DIR:
        break

if DATA_DIR is None:
    for data_name in possible_data_names:
        for mimic_name in possible_mimic_names:
            test_path = SCRIPT_DIR.parent / data_name / mimic_name
            if test_path.exists():
                DATA_DIR = test_path
                break
        if DATA_DIR:
            break

if DATA_DIR is None:
    raise RuntimeError("Could not find MIMIC-IV data directory.")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.time()

# =========================
# FILTER FUNCTIONS
# =========================

def filter_chartevents(chunk):
    relevant_itemids = [220045, 220179, 220180, 220181, 223761, 220277, 220210]
    chunk['valuenum'] = pd.to_numeric(chunk['valuenum'], errors='coerce')
    chunk = chunk[chunk['itemid'].isin(relevant_itemids)]
    chunk = chunk[chunk['valuenum'].notna()]
    return chunk

def filter_labevents(chunk):
    critical_itemids = [50912, 51006, 51301, 51265, 50813, 50971, 50983, 50882, 50931]
    chunk['valuenum'] = pd.to_numeric(chunk['valuenum'], errors='coerce')
    chunk = chunk[chunk['itemid'].isin(critical_itemids)]
    chunk = chunk[chunk['valuenum'].notna()]
    return chunk

def filter_outputevents(chunk):
    urine_itemids = [226559, 226560, 226561, 226584]
    chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
    chunk = chunk[chunk['itemid'].isin(urine_itemids)]
    chunk = chunk[chunk['value'].notna()]
    chunk = chunk[chunk['value'].between(0, 5000)]
    return chunk

def filter_emar(chunk):
    chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
    return chunk

def filter_prescriptions(chunk):
    chunk['starttime'] = pd.to_datetime(chunk['starttime'], errors='coerce')
    return chunk

# =========================
# TABLE DEFINITIONS
# =========================

small_tables = {
    'patients': "hosp/patients.csv.gz",
    'admissions': "hosp/admissions.csv.gz",
    'icustays': "icu/icustays.csv.gz",
    'd_labitems': "hosp/d_labitems.csv.gz",
    'd_items': "icu/d_items.csv.gz",
    'diagnoses_icd': "hosp/diagnoses_icd.csv.gz",
    'd_icd_diagnoses': "hosp/d_icd_diagnoses.csv.gz",
    'd_icd_procedures': "hosp/d_icd_procedures.csv.gz",
}

large_tables = [
    ("icu/chartevents.csv.gz", "chartevents", ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'valuenum'], filter_chartevents),
    ("icu/outputevents.csv.gz", "outputevents", ['subject_id', 'hadm_id', 'stay_id', 'charttime', 'itemid', 'value'], filter_outputevents),
    ("hosp/labevents.csv.gz", "labevents", ['subject_id', 'hadm_id', 'charttime', 'itemid', 'valuenum'], filter_labevents),
    ("hosp/emar.csv.gz", "emar", ['subject_id', 'hadm_id', 'charttime', 'medication'], filter_emar),
    ("hosp/prescriptions.csv.gz", "prescriptions", ['subject_id', 'hadm_id', 'starttime', 'drug'], filter_prescriptions),
]

def process_large_table(relative_path, table_name, keep_cols, filter_func, chunksize=500_000):
    full_path = DATA_DIR / relative_path
    if not full_path.exists():
        print(f"   {table_name}: not found")
        return 0

    print(f"\n  Processing {table_name}...")
    chunks_list = []
    chunk_count = 0
    total_rows = 0

    icustays = None
    for chunk in pd.read_csv(full_path, compression='gzip', chunksize=chunksize):
        chunk_count += 1
        if chunk_count % 10 == 0:
            print(f"    Chunk {chunk_count}: processed {total_rows:,} rows so far...")

        chunk = chunk[[c for c in keep_cols if c in chunk.columns]]

        if filter_func:
            chunk = filter_func(chunk)

        # For labevents, merge stay_id from icustays
        if table_name == "labevents":
            if icustays is None:
                icustays = pd.read_csv(DATA_DIR / "icu/icustays.csv.gz", compression='gzip', usecols=['subject_id', 'hadm_id', 'stay_id'])
            chunk = chunk.merge(icustays, on=['subject_id', 'hadm_id'], how='left')

        if len(chunk) == 0:
            continue

        chunks_list.append(chunk)
        total_rows += len(chunk)

    if chunks_list:
        combined = pd.concat(chunks_list, ignore_index=True)
        combined.to_parquet(PROCESSED_DIR / f"{table_name}.parquet", index=False)
        print(f"     {table_name}: {total_rows:,} rows saved")
    else:
        print(f"     {table_name}: no data after filtering")

    return total_rows

print("\n" + "="*60)
print("PROCESSING LARGE TABLES")
print("="*60)

for path, name, cols, filter_func in large_tables:
    process_large_table(path, name, cols, filter_func)

def load_small_table(relative_path, table_name):
    full_path = DATA_DIR / relative_path
    if not full_path.exists():
        print(f"  ✗ {table_name}: not found")
        return None

    df = pd.read_csv(full_path, compression='gzip')
    print(f"  ✓ {table_name}: {len(df):,} rows, {len(df.columns)} cols")
    return df

print("\n" + "="*60)
print("LOADING SMALL TABLES")
print("="*60)

loaded_small = {}
for name, path in small_tables.items():
    loaded_small[name] = load_small_table(path, name)

print("\n" + "="*60)
print("SAVING SMALL TABLES TO PARQUET")
print("="*60)

for name, df in loaded_small.items():
    if df is not None:
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
        print(f"   Saved {name}.parquet")

# =========================
# SUMMARY
# =========================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary = []

large_names = ['chartevents', 'labevents', 'outputevents', 'emar', 'prescriptions']
for name in large_names:
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        summary.append({
            'Table': name,
            'Rows': f"{len(df):,}",
            'Columns': len(df.columns),
            'Missing %': f"{missing_pct:.1f}%"
        })

for name, df in loaded_small.items():
    if df is not None:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        summary.append({
            'Table': name,
            'Rows': f"{df.shape[0]:,}",
            'Columns': df.shape[1],
            'Missing %': f"{missing_pct:.1f}%"
        })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# =========================
# COMPLETE
# =========================

elapsed = time.time() - start_time
print("\n" + "="*60)
print("LOADING COMPLETE")
print("="*60)
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"All data saved to: {PROCESSED_DIR}")
