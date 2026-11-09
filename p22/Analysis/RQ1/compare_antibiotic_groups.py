# Main writer: Noah Charpentier
# Reviewer: Cahit Kok 
# Contributors: Kimi Knaider

"""
Compares AKI prediction between:
- Vancomycin alone
- Vancomycin + Piperacillin-Tazobactam
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)


# ============================================================================
# IMPORT CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Analysis" / "model"))

from model_config import (
    DATASET_PATH,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    RESULTS_DIR,
    EXCLUDE_COLS, 
    IMPROVED_FEATURES_FINAL,
    KNOWN_UNUSABLE_COLS,
    MISSING_VALUE_THRESHOLD,
    engineer_features,
)


# ============================================================================
# PATHS

MODEL_PATH = Path(MODELS_DIR) / "random_forest.pkl"

RESULTS_DIR = Path(RESULTS_DIR)

BOXPLOT_PATH = RESULTS_DIR / "antibiotic_groups_risk_boxplot.png"


# ============================================================================
# LOAD DATA

print("=" * 10)
print("ANTIBIOTIC GROUP COMPARISON USING RANDOM FOREST")
print("=" * 10)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Create the same engineered features used during model training
df = engineer_features(df)

# Clean target
df = df[df[TARGET].notna()].copy()
df[TARGET] = df[TARGET].astype(int)

# Encode sex if needed
if "sex" in df.columns and not pd.api.types.is_numeric_dtype(df["sex"]):
    df["sex"] = (df["sex"].astype(str).str.upper() == "M").astype(int)

# Convert boolean columns to integers if any remain
bool_cols = df.select_dtypes(include=["bool"]).columns
df[bool_cols] = df[bool_cols].astype(int)

print(f"AKI prevalence: {df[TARGET].mean():.2%}")


# ============================================================================
# FEATURE SELECTION

feature_cols = [
    col for col in IMPROVED_FEATURES_FINAL
    if col in df.columns
    and col not in KNOWN_UNUSABLE_COLS
    and col not in EXCLUDE_COLS
    and pd.api.types.is_numeric_dtype(df[col])
]

feature_cols = [
    col for col in feature_cols
    if df[col].isna().mean() <= MISSING_VALUE_THRESHOLD
]

feature_cols = [
    col for col in feature_cols
    if df[col].nunique(dropna=True) > 1
]

# ============================================================================
# LOAD MODEL + FEATURES

model = joblib.load(MODEL_PATH)


# Safety check: all model features must exist in the dataframe
missing_features = [
    col for col in feature_cols
    if col not in df.columns
]

if missing_features:
    raise ValueError(
        "Some features used by the Random Forest model are missing from the dataset:\n"
        + "\n".join(f"  - {col}" for col in missing_features)
    )

print(f"Loaded Random Forest model from: {MODEL_PATH}")
print(f"Using {len(feature_cols)} model features")


# ============================================================================
# RECREATE SAME PATIENT-LEVEL TEST SPLIT

y = df[TARGET]
groups = df["subject_id"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

_, test_idx = next(splitter.split(df, y, groups=groups))

test_df = df.iloc[test_idx].copy()

X_test = test_df[feature_cols]
y_test = test_df[TARGET]

print("\nPatient-level test set:")
print(f"Test rows: {len(test_df)}")
print(f"Test AKI prevalence: {y_test.mean():.2%}")


# ============================================================================
# SPLIT BY ANTIBIOTIC GROUP

if "exposure" not in test_df.columns:
    raise KeyError("Column 'exposure' is missing. It should be created by engineer_features().")

vanco_only = test_df[test_df["exposure"] == 0].copy()
combo = test_df[test_df["exposure"] == 1].copy()

X_vanco = vanco_only[feature_cols]
y_vanco = vanco_only[TARGET]

X_combo = combo[feature_cols]
y_combo = combo[TARGET]


# ============================================================================
# PREDICT

prob_vanco = model.predict_proba(X_vanco)[:, 1]
prob_combo = model.predict_proba(X_combo)[:, 1]


# ============================================================================
# PRINT BASIC RESULTS

print("\n=== GROUP COMPARISON ===")

print(f"\nVanco only - size: {len(vanco_only)}")
print(f"Observed AKI: {y_vanco.mean():.2%}")
print(f"Mean predicted risk: {prob_vanco.mean():.2%}")
print(f"ROC-AUC: {roc_auc_score(y_vanco, prob_vanco):.3f}")
print(f"PR-AUC: {average_precision_score(y_vanco, prob_vanco):.3f}")

print(f"\nCombo size: {len(combo)}")
print(f"Observed AKI: {y_combo.mean():.2%}")
print(f"Mean predicted risk: {prob_combo.mean():.2%}")
print(f"ROC-AUC: {roc_auc_score(y_combo, prob_combo):.3f}")
print(f"PR-AUC: {average_precision_score(y_combo, prob_combo):.3f}")


# ============================================================================
# ROC CURVE

fpr_v, tpr_v, _ = roc_curve(y_vanco, prob_vanco)
fpr_c, tpr_c, _ = roc_curve(y_combo, prob_combo)

auc_v = roc_auc_score(y_vanco, prob_vanco)
auc_c = roc_auc_score(y_combo, prob_combo)


# ============================================================================
# PRECISION-RECALL CURVE

prec_v, rec_v, _ = precision_recall_curve(y_vanco, prob_vanco)
prec_c, rec_c, _ = precision_recall_curve(y_combo, prob_combo)

pr_v = average_precision_score(y_vanco, prob_vanco)
pr_c = average_precision_score(y_combo, prob_combo)

# ============================================================================
# RISK BOXPLOT

plt.figure(figsize=(6, 6))
plt.boxplot(
    [prob_vanco, prob_combo],
    labels=["Vanco only", "Combo"],
)
plt.ylabel("Predicted AKI risk")
plt.title("Predicted AKI Risk by Antibiotic Group")
plt.tight_layout()
plt.savefig(BOXPLOT_PATH, dpi=300)
plt.close()


# ============================================================================
# DONE

print("\nPlots saved:")
print(f"Risk boxplot: {BOXPLOT_PATH}")
