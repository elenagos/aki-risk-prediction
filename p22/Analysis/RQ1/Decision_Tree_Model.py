# Main writer: Noah Charpentier
# Reviewer:  Cahit Kok
# Contributors:

"""
Decision Tree Model for AKI Prediction

- Uses the same config as the rest of the project
- Uses patient-level train/test split with subject_id
- Keeps imputation inside a Pipeline to avoid leakage
- Removes creatinine-slope style features because they are too close to the AKI definition
- Tunes the tree with GroupKFold
- Reports the same metrics as the other models
"""

from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt


# ============================================================================
# IMPORT PROJECT CONFIG

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Analysis" / "model"))

from model_config import (
    DATASET_PATH,
    TARGET,
    EXCLUDE_COLS,
    IMPROVED_FEATURES_FINAL,
    KNOWN_UNUSABLE_COLS,
    engineer_features,
    MISSING_VALUE_THRESHOLD,
    IMPUTATION_STRATEGY,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    RESULTS_DIR,
)

from model_results import ModelResults


# ============================================================================
# OUTPUT PATHS

TREE_MODEL_PATH = Path(MODELS_DIR) / "decision_tree.pkl"
TREE_RESULTS_PATH = Path(RESULTS_DIR) / "decision_tree_results.json"
TREE_FEATURES_PATH = Path(MODELS_DIR) / "features_decision_tree.txt"
TREE_PLOT_PATH = Path(RESULTS_DIR) / "decision_tree_preview.png"



# ============================================================================
# LOAD DATA

print("=" * 10)
print("DECISION TREE MODEL FOR AKI PREDICTION")
print("Version: without creatinine slope / creatinine outcome-proxy features")
print("=" * 10)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df_model = df.copy()
df_model = engineer_features(df_model)

df_model = df_model[df_model[TARGET].notna()].copy()
df_model[TARGET] = df_model[TARGET].astype(int)

print(f"AKI prevalence: {df_model[TARGET].mean():.2%}")


# ============================================================================
# FEATURE SELECTION USING APPROVED IMPROVED FEATURE SET

# Encode sex if needed
if "sex" in df_model.columns and not pd.api.types.is_numeric_dtype(df_model["sex"]):
    df_model["sex"] = (df_model["sex"].astype(str).str.upper() == "M").astype(int)

# Convert boolean columns to integers if any remain
bool_cols = df_model.select_dtypes(include=["bool"]).columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

# Use ONLY the approved improved feature list from model_config.py
feature_cols = [
    col for col in IMPROVED_FEATURES_FINAL
    if col in df_model.columns
    and col not in KNOWN_UNUSABLE_COLS
    and col not in EXCLUDE_COLS
    and pd.api.types.is_numeric_dtype(df_model[col])
]

# Drop columns with too much missingness
feature_cols = [
    col for col in feature_cols
    if df_model[col].isna().mean() <= MISSING_VALUE_THRESHOLD
]

# Drop constant columns
feature_cols = [
    col for col in feature_cols
    if df_model[col].nunique(dropna=True) > 1
]

print(f"Number of improved features used: {len(feature_cols)}")

missing_features = [
    col for col in IMPROVED_FEATURES_FINAL
    if col not in df_model.columns
]

if missing_features:
    print("\nApproved features not found in dataset:")
    for col in missing_features:
        print(f"  - {col}")
        
X = df_model[feature_cols]
y = df_model[TARGET]
groups = df_model["subject_id"]


# ============================================================================
# PATIENT-LEVEL SPLIT

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

print("\nPatient-level split:")
print(f"Train: {len(X_train)} rows, AKI rate = {y_train.mean():.2%}")
print(f"Test:  {len(X_test)} rows, AKI rate = {y_test.mean():.2%}")


# ============================================================================
# PIPELINE

def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTATION_STRATEGY)),
        ("tree", DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


# ============================================================================
# HYPERPARAMETER TUNING

param_grid = {
    "tree__criterion": ["gini", "entropy"],
    "tree__max_depth": [3, 5, 8, 12],
    "tree__min_samples_split": [20, 50, 100],
    "tree__min_samples_leaf": [10, 25, 50],
}

grid = GridSearchCV(
    estimator=make_pipeline(),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    verbose=1,
)

print("\nTraining Decision Tree with GridSearchCV...")
grid.fit(X_train, y_train, groups=groups_train)

model = grid.best_estimator_

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")


# ============================================================================
# EVALUATION

y_prob = model.predict_proba(X_test)[:, 1]
 
# chosen after testing 0.30 / 0.35 / 0.40
# gives a good balance between recall (~75%) and precision (~35%)
# avoids too many false positives while still detecting most AKI cases (we do not overwhelm doctors)
THRESHOLD = 0.40
y_pred = (y_prob >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

metrics = {
    "roc_auc": float(roc_auc_score(y_test, y_prob)),
    "pr_auc": float(average_precision_score(y_test, y_prob)),
    "brier_score": float(brier_score_loss(y_test, y_prob)),
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall_sensitivity": float(recall_score(y_test, y_pred, zero_division=0)),
    "specificity": float(specificity),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
}

print("\nTEST SET PERFORMANCE:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))


# ============================================================================
# FEATURE IMPORTANCE

tree_model = model.named_steps["tree"]

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": tree_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 20 Decision Tree features:")
print(feature_importance.head(20).to_string(index=False))



# ============================================================================
# SAVE SMALL TREE VISUALIZATION

plt.figure(figsize=(24, 12))
plot_tree(
    tree_model,
    feature_names=feature_cols,
    class_names=["No AKI", "AKI"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
)
plt.title("Decision Tree Preview (First 3 Levels)")
plt.tight_layout()
plt.savefig(TREE_PLOT_PATH, dpi=300)
plt.close()

print(f"Tree preview saved to: {TREE_PLOT_PATH}")
