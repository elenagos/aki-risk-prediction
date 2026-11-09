# Main writer: Aadya Charlewar
# Reviewer:  Bartlomiej Kaczor
# Contributors: Kimi Knaider

import pandas as pd
from pathlib import Path

# Set up paths (adjust if needed)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mimic-iv-3.1"
PROCESSED_DIR = DATA_DIR.parent / "processed"

small_tables = {
    'caregivers': "icu/caregiver.csv.gz",
    'd_items': "icu/d_items.csv.gz",
    'icustays': "icu/icustays.csv.gz",
    'datetimeevents': "icu/datetimeevents.csv.gz",
    'inputevents': "icu/inputevents.csv.gz",
    'ingredientevents': "icu/ingredientevents.csv.gz",
    'procedureevents': "icu/procedureevents.csv.gz",
    'admissions': "hosp/admissions.csv.gz",
    'd_labitems': "hosp/d_labitems.csv.gz",
    'd_icd_diagnoses': "hosp/d_icd_diagnoses.csv.gz",
    'd_icd_procedures': "hosp/d_icd_procedures.csv.gz",
    'diagnoses_icd': "hosp/diagnoses_icd.csv.gz",
    'drgcodes': "hosp/drgcodes.csv.gz",
    'hcpcsevents': "hosp/hcpcsevents.csv.gz",
    'microbiologyevents': "hosp/microbiologyevents.csv.gz",
    'omr': "hosp/omr.csv.gz",
    'patients': "hosp/patients.csv.gz",
    'pharmacy': "hosp/pharmacy.csv.gz",
    'poe': "hosp/poe.csv.gz",
    'poe_detail': "hosp/poe_detail.csv.gz",
    'provider': "hosp/provider.csv.gz",
    'procedures_icd': "hosp/procedures_icd.csv.gz",
    'transfers': "hosp/transfers.csv.gz",
    'services': "hosp/services.csv.gz",
}

for name, rel_path in small_tables.items():
    print(f"\nAttempting to load: {name} ({rel_path})")
    full_path = DATA_DIR / rel_path
    if not full_path.exists():
        print(f"   {name}: file not found")
        continue
    try:
        df = pd.read_csv(full_path, compression='gzip', nrows=1000)  # Only load 1000 rows for test
        print(f"   {name}: {df.shape[0]} rows, {df.shape[1]} cols (first 1000 rows loaded)")
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")

print("\nDone. The file that fails or is huge is likely the culprit.")
