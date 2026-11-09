# Main writer: Kimi Knaider
# Reviewer: Luca Dragutanu
# Contributors:

"""
Configuration and paths for AKI prediction pipeline.
All paths are relative and dynamically resolved.
"""

from pathlib import Path

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

# Root directory (P22_05)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "mimic-iv-3.1"
PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_DIR = DATA_DIR / "cleaned"
FINAL_DIR = DATA_DIR / "final"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# MIMIC TABLE MAPPING
# ============================================================================

# Large tables that need chunking
LARGE_TABLES = {
    'chartevents': 'icu/chartevents.csv.gz',
    'labevents': 'hosp/labevents.csv.gz',
    'outputevents': 'icu/outputevents.csv.gz',
    'emar': 'hosp/emar.csv.gz',
    'prescriptions': 'hosp/prescriptions.csv.gz',
}

# Small tables that fit in memory
SMALL_TABLES = {
    'patients': 'hosp/patients.csv.gz',
    'admissions': 'hosp/admissions.csv.gz',
    'icustays': 'icu/icustays.csv.gz',
    'd_labitems': 'hosp/d_labitems.csv.gz',
    'diagnoses_icd': 'hosp/diagnoses_icd.csv.gz',
    'procedures_icd': 'hosp/procedures_icd.csv.gz',
    'd_icd_diagnoses': 'hosp/d_icd_diagnoses.csv.gz',
    'd_icd_procedures': 'hosp/d_icd_procedures.csv.gz',
    'd_items': 'icu/d_items.csv.gz',
    'microbiologyevents': 'hosp/microbiologyevents.csv.gz',
    'pharmacy': 'hosp/pharmacy.csv.gz',
}

# ============================================================================
# LAB ITEM IDs (MIMIC-IV)
# ============================================================================

LAB_ITEMS = {
    'CREATININE': 50912,           # Creatinine
    'BUN': 51006,                  # BUN
    'SODIUM': 50824,               # Sodium
    'POTASSIUM': 50822,            # Potassium
    'BICARBONATE': 50803,          # Bicarbonate
    'CHLORIDE': 50806,             # Chloride
    'GLUCOSE': 50809,              # Glucose
    'MAGNESIUM': 50960,            # Magnesium
    'PHOSPHATE': 50971,            # Phosphate
    'CALCIUM': 50808,              # Calcium
    'ALBUMIN': 50862,              # Albumin
    'HEMOGLOBIN': 50811,           # Hemoglobin
    'HEMATOCRIT': 51221,           # Hematocrit
    'WBC': 51301,                  # WBC
    'PLATELETS': 51265,            # Platelets
    'LACTATE': 50813,              # Lactate
    'PTT': 51275,                  # PTT
    'PT': 51274,                   # PT
    'INR': 51237,                  # INR
}

# ============================================================================
# VITAL SIGN ITEM IDs (MIMIC-IV)
# ============================================================================

VITAL_ITEMS = {
    'HEART_RATE': 220045,          # Heart Rate
    'SYSTOLIC_BP': 220050,         # Systolic BP
    'DIASTOLIC_BP': 220051,        # Diastolic BP
    'MAP': 220052,                 # Mean Arterial Pressure
    'RESPIRATION': 220210,         # Respiration Rate
    'TEMPERATURE': 223761,         # Temperature (Fahrenheit)
    'SpO2': 220277,                # SpO2 (%)
}

# ============================================================================
# URINE OUTPUT ITEM IDs
# ============================================================================

URINE_ITEMS = {
    'URINE_FOLEY': 226559,
    'URINE_STRAIGHT_CATH': 226560,
    'URINE_VOID': 226561,
    'URINE_VOID_CATHETER': 226584,
}

# ============================================================================
# ICD-9/ICD-10 CODES FOR COMORBIDITIES
# ============================================================================

COMORBIDITY_CODES = {
    'CKD': ['585', 'N18', 'N185', 'N186', 'N187', 'N188', 'N189'],
    'ESRD': ['586', 'N19'],
    'DIABETES': ['250', 'E10', 'E11', 'E12', 'E13', 'E14'],
    'HYPERTENSION': ['401', '402', '403', 'I10', 'I11', 'I12', 'I13'],
    'HEART_FAILURE': ['428', 'I50'],
    'ISCHEMIC_HEART_DISEASE': ['414', 'I25'],
    'MI': ['410', 'I21'],
    'ARRHYTHMIA': ['427', 'I47', 'I48', 'I49'],
    'COPD': ['491', '492', '496', 'J41', 'J42', 'J43', 'J44'],
    'LIVER_DISEASE': ['571', 'K70', 'K71', 'K72', 'K73', 'K74', 'K75', 'K76'],
    'MALIGNANCY': ['140', '141', '142', '143', '144', '145', '146', '147', 'C'],
    'PVD': ['440', '441', 'I70', 'I71', 'I72', 'I73', 'I74'],
}

# ============================================================================
# MEDICATION PATTERNS
# ============================================================================

MEDICATION_PATTERNS = {
    'VANCOMYCIN': ['vancomycin', 'vancocin'],
    'PIPERACILLIN_TAZOBACTAM': ['piperacillin', 'tazobactam', 'pip.*taz'],
    'AMINOGLYCOSIDE': ['gentamicin', 'tobramycin', 'amikacin', 'neomycin'],
    'LOOP_DIURETIC': ['furosemide', 'lasix', 'torsemide', 'bumex', 'ethacrynic'],
    'NSAID': ['ibuprofen', 'naproxen', 'ketoprofen', 'indomethacin'],
    'ACEI_ARB': ['lisinopril', 'enalapril', 'losartan', 'valsartan', 'candesartan', 'olmesartan'],
    'AMPHOTERICIN': ['amphotericin'],
    'VASOPRESSOR': ['norepinephrine', 'epinephrine', 'dopamine', 'vasopressin', 'phenylephrine'],
    'SEDATIVE': ['midazolam', 'propofol', 'lorazepam', 'dexmedetomidine'],
}

# ============================================================================
# STUDY PARAMETERS
# ============================================================================

# Time windows
PREDICTION_WINDOW_HOURS = 24  # First 24 hours after index time
OUTCOME_WINDOW_DAYS = 7       # Within 7 days after index time
OUTCOME_WINDOW_HOURS = 24 * 7

# AKI definition (KDIGO creatinine-based)
AKI_CREAT_DELTA_48H = 0.3     # >= 0.3 mg/dL within 48h
AKI_CREAT_RATIO_7D = 1.5      # >= 1.5x baseline within 7 days

# Exclusion criteria
MIN_AGE = 18
MIN_ICU_LOS_HOURS = 24
BASELINE_CREAT_LOOKBACK_DAYS = 7

# Missing data rules
MAX_MISSING_PCT = 0.3          # Drop predictors with > 30% missing
EXCLUDE_ON_KEY_MISSING = True  # Exclude cohort if key variables missing

# ============================================================================
# ANALYSIS UNIT
# ============================================================================

ANALYSIS_UNIT = "stay_id"  # Each row = one ICU stay (not hadm_id)
"""
Rationale: stay_id is more granular and clinically meaningful.
If a patient has multiple ICU stays in the same admission, 
each is a separate episode of exposure/outcome risk.
"""

# ============================================================================
# FINAL DATASET INFO
# ============================================================================

FINAL_OUTPUT_FILE = FINAL_DIR / "final_aki_dataset.parquet"
FINAL_OUTPUT_CSV = FINAL_DIR / "final_aki_dataset.csv"

FINAL_DATASET_DESCRIPTION = """
AKI Prediction Cohort: MIMIC-IV Adult ICU Patients Exposed to Vancomycin

ANALYSIS UNIT: One row = one ICU stay (stay_id)

WINDOWS:
- Prediction window: First 24 hours after first vancomycin administration
- Outcome window: 7 days after first vancomycin administration

INCLUSION CRITERIA:
- Age >= 18 years
- ICU length of stay >= 24 hours
- First ICU stay per patient
- Received vancomycin during ICU stay
- Valid baseline creatinine value

EXCLUSION CRITERIA:
- Missing key exposure timing
- Missing baseline creatinine
- Chronic kidney disease / ESRD at admission (cannot define incident AKI)
- Patient with AKI at or before index time

EXPOSURE GROUPS:
1. vanco_only: Vancomycin alone (no concurrent piperacillin-tazobactam)
2. vanco_ptz_combo: Vancomycin + piperacillin-tazobactam combination

PRIMARY OUTCOME:
- aki_day7: Binary indicator of AKI within 7 days after index time
- Defined as serum creatinine increase >= 0.3 mg/dL within 48h 
  OR serum creatinine ratio >= 1.5 baseline within 7 days (KDIGO)

SECONDARY OUTCOMES:
- time_to_aki_hours: Hours from index time to AKI event (if occurred)
- creatinine_delta_48h: Maximum creatinine increase in first 48h
- creatinine_ratio_7d: Maximum creatinine ratio over 7 days
"""
