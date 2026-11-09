# Main writer: Aadya Charlewar
# Reviewer: Luca Dragutanu
# Contributors: Kimi Knaider

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def load_table(name):
    """Loads a parquet file from processed directory"""
    path = PROCESSED_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None

# Load all tables (same variable names as before)
icu_stays = load_table("icustays")
patients = load_table("patients")
diagnoses_icd = load_table("diagnoses_icd")
microbiologyevents = load_table("microbiologyevents")
pharmacy = load_table("pharmacy")
d_items = load_table("d_items")
lab_events = load_table("labevents")
chart_events = load_table("chartevents")
admissions = load_table("admissions")
prescriptions = load_table("prescriptions")
emar = load_table("emar")
output_events = load_table("outputevents")
d_lab_items = load_table("d_labitems")
d_icd_diagnoses = load_table("d_icd_diagnoses")
d_icd_procedures = load_table("d_icd_procedures")
drug_codes = load_table("drgcodes")
poe = load_table("poe")
poe_detail = load_table("poe_detail")
services = load_table("services")
transfers = load_table("transfers")
procedures_icd = load_table("procedures_icd")
hcpcs_events = load_table("hcpcsevents")
omr = load_table("omr")
ingredient_events = load_table("ingredientevents")
input_events = load_table("inputevents")
procedure_events = load_table("procedureevents")
date_time_events = load_table("datetimeevents")
caregivers = load_table("caregivers")

print(f"Loaded {sum(1 for x in locals().values() if isinstance(x, pd.DataFrame))} tables")


# EXCLUSION CRITERIA

print("APPLYING EXCLUSION CRITERIA")

# 1. ICU stay < 24h
if icu_stays is not None:
    icu = icu_stays.copy()
    icu['intime'] = pd.to_datetime(icu['intime'])
    icu['outtime'] = pd.to_datetime(icu['outtime'])
    icu['los_hours'] = (icu['outtime'] - icu['intime']).dt.total_seconds() / 3600
    before = len(icu)
    icu = icu[icu['los_hours'] >= 24]
    icu_stays = icu
    print(f"Excluded {before - len(icu)} ICU stays < 24h")

# 2. Age < 18
if patients is not None:
    pat = patients.copy()
    pat['anchor_age'] = pd.to_numeric(pat['anchor_age'], errors='coerce')
    before = len(pat)
    pat = pat[pat['anchor_age'] >= 18]
    patients = pat
    print(f"Excluded {before - len(pat)} patients under 18")

# 3. Renal disease (from diagnoses_icd)
if diagnoses_icd is not None and patients is not None:
    diag = diagnoses_icd.copy()
    renal_codes = ['585', '586', 'N18', 'N19']
    renal_patients = diag[diag['icd_code'].str.startswith(tuple(renal_codes), na=False)]['subject_id'].unique()
    pat = patients.copy()
    before = len(pat)
    pat = pat[~pat['subject_id'].isin(renal_patients)]
    patients = pat
    print(f"Excluded {before - len(pat)} patients with renal disease")

# 4. First ICU stay only
if icu_stays is not None:
    icu = icu_stays.copy()
    before = len(icu)
    icu = icu.sort_values(['subject_id', 'intime']).groupby('subject_id').first().reset_index()
    icu_stays = icu
    print(f"Kept only first ICU stay per patient (was {before}, now {len(icu)})")

print("REMOVING HIGH MISSING TABLES")

print("  emar_detail: excluded (71.5% missing, not loaded)")
print("  d_hcpcs: excluded (25% missing, not needed)")

if microbiologyevents is not None:
    micro = microbiologyevents.copy()
    micro['chartdate'] = pd.to_datetime(micro['chartdate'], errors='coerce')
    micro_keep_cols = ['subject_id', 'hadm_id', 'chartdate', 'spec_type_desc',
                       'org_name', 'ab_name', 'interpretation']
    micro_cols = [col for col in micro_keep_cols if col in micro.columns]
    micro = micro[micro_cols]
    microbiologyevents = micro
    print(f"microbiologyevents: kept {len(micro)} rows, {len(micro.columns)} columns")

if pharmacy is not None:
    pharm = pharmacy.copy()
    pharm['starttime'] = pd.to_datetime(pharm['starttime'], errors='coerce')
    pharm['stoptime'] = pd.to_datetime(pharm['stoptime'], errors='coerce')
    pharm_keep_cols = ['subject_id', 'hadm_id', 'starttime', 'stoptime',
                       'medication', 'route', 'status']
    pharm_cols = [col for col in pharm_keep_cols if col in pharm.columns]
    pharm = pharm[pharm_cols]
    if 'status' in pharm.columns:
        pharm = pharm[pharm['status'].str.contains('Finished|Active', na=False)]
    pharmacy = pharm
    if len(pharm) == 0:
        print(f"pharmacy: no active orders found")
    else:
        print(f"pharmacy: kept {len(pharm)} rows, {len(pharm.columns)} columns")

# D_ITEMS

print("PROCESSING D_ITEMS")

if d_items is not None:
    print(f"d_items: kept {len(d_items)} rows, {len(d_items.columns)} columns")

# LAB EVENTS

print("PROCESSING LAB EVENTS")

if lab_events is not None:
    labs = lab_events.copy()
    labs['valuenum'] = pd.to_numeric(labs['valuenum'], errors='coerce')
    labs['charttime'] = pd.to_datetime(labs['charttime'], errors='coerce')

    critical_lab_itemids = [
        50912,  # Creatinine
        51006,  # BUN
        51301,  # WBC
        51265,  # Platelets
        50813,  # Lactate
        50971,  # Potassium
        50983,  # Sodium
        50882,  # Bicarbonate
        50931   # Glucose
    ]

    labs = labs[labs['itemid'].isin(critical_lab_itemids)]
    labs = labs[labs['valuenum'].notna()]

    labs = labs[
        ((labs['itemid'] == 50912) & (labs['valuenum'].between(0.2, 15)))   |
        ((labs['itemid'] == 51006) & (labs['valuenum'].between(1, 200)))    |
        ((labs['itemid'] == 51301) & (labs['valuenum'].between(0, 100)))    |
        ((labs['itemid'] == 51265) & (labs['valuenum'].between(10, 1000)))  |
        ((labs['itemid'] == 50813) & (labs['valuenum'].between(0, 20)))     |
        ((labs['itemid'] == 50971) & (labs['valuenum'].between(1, 10)))     |
        ((labs['itemid'] == 50983) & (labs['valuenum'].between(100, 180)))  |
        ((labs['itemid'] == 50882) & (labs['valuenum'].between(10, 50)))    |
        ((labs['itemid'] == 50931) & (labs['valuenum'].between(30, 2000)))
    ]

    lab_events = labs
    print(f"lab_events: kept {len(labs)} rows (creatinine, BUN, WBC, platelets, lactate, potassium, sodium, bicarbonate, glucose)")

# DATETIME CONVERSION

print("CONVERTING DATETIME COLUMNS")

def convert_datetime_columns(df, df_name):
    datetime_patterns = ['time', 'date', 'intime', 'outtime', 'start', 'end',
                         'chart', 'store', 'admit', 'disch']
    converted = 0
    for col in df.columns:
        if any(pattern in col.lower() for pattern in datetime_patterns):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                converted += 1
            except Exception:
                pass
    if converted > 0:
        print(f"    {df_name}: converted {converted} columns")
    return df

# List of tables to convert
tables_to_convert = [
    ('chart_events', chart_events),
    ('icu_stays', icu_stays),
    ('admissions', admissions),
    ('patients', patients),
    ('prescriptions', prescriptions),
    ('emar', emar),
    ('output_events', output_events),
    ('input_events', input_events),
    ('procedure_events', procedure_events),
    ('date_time_events', date_time_events),
    ('ingredient_events', ingredient_events)
]

for name, df in tables_to_convert:
    if df is not None:
        # Update the variable in locals
        exec(f"{name} = convert_datetime_columns(df, '{name}')")


# PATIENTS

print("PROCESSING PATIENTS")

if patients is not None:
    pat = patients.copy()
    pat['anchor_age'] = pd.to_numeric(pat['anchor_age'], errors='coerce')
    pat['age_missing'] = pat['anchor_age'].isnull().astype(int)
    pat['anchor_age'] = pat['anchor_age'].fillna(pat['anchor_age'].median())
    pat['gender'] = pat['gender'].fillna('Unknown')
    pat['age_ge65'] = (pat['anchor_age'] >= 65).astype(int)
    patients = pat
    print("Patients: age median imputed, missing flag created; gender Unknown; age_ge65 flag created")

# ADMISSIONS

print("PROCESSING ADMISSIONS")

if admissions is not None:
    adm = admissions.copy()
    for col in ['insurance', 'language', 'marital_status', 'race', 'admission_type']:
        if col in adm.columns:
            adm[col] = adm[col].fillna('Unknown')
    adm['hospital_mortality'] = adm['hospital_expire_flag'] == 1
    admissions = adm
    print("Admissions: categorical filled with Unknown; hospital_mortality flag created")

# CHART EVENTS

print("PROCESSING CHART EVENTS")

if chart_events is not None:
    chart = chart_events.copy()
    vital_itemids = [220045, 220179, 220180, 220181, 223761, 220277, 220210]
    chart = chart[chart['itemid'].isin(vital_itemids)]

    vital_map = {220045: 'hr', 220179: 'sbp', 220180: 'dbp', 220181: 'map',
                 223761: 'temp', 220277: 'spo2', 220210: 'rr'}
    for itemid, name in vital_map.items():
        mask = chart['itemid'] == itemid
        chart.loc[mask, f'{name}_missing'] = chart.loc[mask, 'valuenum'].isnull().astype(int)

    chart = chart[chart['valuenum'].notna()]

    chart = chart[
        ((chart['itemid'] == 220045) & (chart['valuenum'].between(20, 250)))   |
        ((chart['itemid'].isin([220179, 220180, 220181])) & (chart['valuenum'].between(20, 300))) |
        ((chart['itemid'] == 223761) & (chart['valuenum'].between(86.0, 107.6))) |
        ((chart['itemid'] == 220277) & (chart['valuenum'].between(50, 100)))   |
        ((chart['itemid'] == 220210) & (chart['valuenum'].between(5, 60)))
    ]

    chart_events = chart
    print("Chart_events: missing flags created, impossible values removed")
    print("  NOTE: temperature range corrected to Fahrenheit (86–107.6°F) as stored in MIMIC-IV")

# LAB EVENTS (already processed above)

if lab_events is not None:
    print("Lab_events: creatinine, BUN, WBC, platelets, lactate, potassium, sodium, bicarbonate, glucose retained")

# OUTPUT EVENTS

print("PROCESSING OUTPUT EVENTS")

if output_events is not None:
    output = output_events.copy()
    output['value'] = pd.to_numeric(output['value'], errors='coerce')
    output['urine_missing'] = output['value'].isnull().astype(int)
    output = output[output['value'].notna()]
    output = output[output['value'].between(0, 5000)]
    output_events = output
    print("Output_events: missing flag created, impossible values removed")

# PRESCRIPTIONS

print("PROCESSING PRESCRIPTIONS")

if prescriptions is not None:
    rx = prescriptions.copy()
    rx['starttime'] = pd.to_datetime(rx['starttime'], errors='coerce')
    before = len(rx)
    rx = rx[rx['starttime'].notna()]
    print(f"Prescriptions: dropped {before - len(rx)} rows with missing starttime")
    rx['drug_lower'] = rx['drug'].str.lower().fillna('')
    rx['is_vancomycin'] = rx['drug_lower'].str.contains('vancomycin', na=False).astype(int)
    rx['is_pip_tazo'] = rx['drug_lower'].str.contains('piperacillin|tazobactam|zosyn', na=False).astype(int)
    rx['is_aminoglycoside'] = rx['drug_lower'].str.contains('gentamicin|amikacin|tobramycin', na=False).astype(int)
    rx['is_loop_diuretic'] = rx['drug_lower'].str.contains('furosemide|bumetanide|torsemide', na=False).astype(int)
    rx['drug_missing'] = rx['drug'].isnull().astype(int)
    rx['drug'] = rx['drug'].fillna('Unknown')
    prescriptions = rx
    print("Prescriptions: antibiotic flags created")

print("CLEANING COMPLETE")

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "mimic-iv-3.1"

# Microbiologyevents
if microbiologyevents is None:
    micro_path = RAW_DATA_DIR / "hosp/microbiologyevents.csv.gz"
    if micro_path.exists():
        print("\nLoading microbiologyevents from gz...")
        microbiologyevents = pd.read_csv(micro_path, compression='gzip', low_memory=False)

        micro = microbiologyevents.copy()
        micro['chartdate'] = pd.to_datetime(micro['chartdate'], errors='coerce')
        micro_keep_cols = ['subject_id', 'hadm_id', 'chartdate', 'spec_type_desc',
                           'org_name', 'ab_name', 'interpretation']
        micro_cols = [col for col in micro_keep_cols if col in micro.columns]
        micro = micro[micro_cols]
        
        # Convert object columns to string for parquet
        for col in micro.columns:
            if micro[col].dtype == 'object':
                micro[col] = micro[col].astype(str)
        
        # Save to parquet
        micro.to_parquet(PROCESSED_DIR / "microbiologyevents.parquet", index=False)
        microbiologyevents = micro
        print(f"  microbiologyevents: {len(micro):,} rows saved to parquet")

# Drug codes (drgcodes)
if drug_codes is None:
    drg_path = RAW_DATA_DIR / "hosp/drgcodes.csv.gz"
    if drg_path.exists():
        print("\nLoading drug_codes from gz...")
        drug_codes = pd.read_csv(drg_path, compression='gzip', low_memory=False)
        
        # Convert object columns to string for parquet
        for col in drug_codes.columns:
            if drug_codes[col].dtype == 'object':
                drug_codes[col] = drug_codes[col].astype(str)
        
        # Save to parquet
        drug_codes.to_parquet(PROCESSED_DIR / "drgcodes.parquet", index=False)
        print(f"  drug_codes: {len(drug_codes):,} rows saved to parquet")

        # Pharmacy (if it didn't save)
pharmacy_path = PROCESSED_DIR / "pharmacy.parquet"
if not pharmacy_path.exists() and pharmacy is not None:
    print("\nSaving pharmacy to parquet...")
    pharm = pharmacy.copy()
    for col in pharm.columns:
        if pharm[col].dtype == 'object':
            pharm[col] = pharm[col].astype(str)
    pharm.to_parquet(PROCESSED_DIR / "pharmacy.parquet", index=False)
    print(f"  pharmacy: {len(pharm):,} rows saved to parquet")



#Patients didn't save lets fix

PROCESSED_DIR = Path("C:/Users/charl/Downloads/group05/P22_05/.vscode/data/processed")
RAW_DATA_DIR = Path("C:/Users/charl/Downloads/group05/P22_05/.vscode/data/mimic-iv-3.1")

# Load patients
patients = pd.read_csv(RAW_DATA_DIR / "hosp/patients.csv.gz", compression='gzip')
patients.to_parquet(PROCESSED_DIR / "patients.parquet", index=False)
print(f"Saved patients: {len(patients):,} rows")
