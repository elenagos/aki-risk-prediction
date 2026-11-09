# Main writer: Noah Charpentier
# Reviewer: Cahit Kok
# Contributors: Kimi Knaider

from pathlib import Path
import sys

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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

RF_MODEL_PATH = Path(MODELS_DIR) / "random_forest.pkl"
RF_RESULTS_PATH = Path(RESULTS_DIR) / "random_forest_results.json"

RF_IMPORTANCE_PLOT_PATH = Path(RESULTS_DIR) / "random_forest_feature_importance.png"
RF_CONFUSION_MATRIX_PATH = Path(RESULTS_DIR) / "random_forest_confusion_matrix.png"

SHAP_SUMMARY_PATH = Path(RESULTS_DIR) / "random_forest_shap_summary.png"
SHAP_BAR_PATH = Path(RESULTS_DIR) / "random_forest_shap_bar.png"
SHAP_VALUES_PATH = Path(RESULTS_DIR) / "random_forest_shap_values.csv"


# ============================================================================
# LOAD DATA

print("=" * 10)
print("RANDOM FOREST MODEL FOR AKI PREDICTION")
print("Version: using IMPROVED_FEATURES_FINAL")
print("=" * 10)

df = pd.read_csv(DATASET_PATH)
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

df_model = df.copy()
df_model = engineer_features(df_model)

df_model = df_model[df_model[TARGET].notna()].copy()
df_model[TARGET] = df_model[TARGET].astype(int)

print(f"AKI prevalence: {df_model[TARGET].mean():.2%}")


# ============================================================================
# FEATURE SELECTION

if "sex" in df_model.columns and not pd.api.types.is_numeric_dtype(df_model["sex"]):
    df_model["sex"] = (df_model["sex"].astype(str).str.upper() == "M").astype(int)

bool_cols = df_model.select_dtypes(include=["bool"]).columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

feature_cols = [
    col for col in IMPROVED_FEATURES_FINAL
    if col in df_model.columns
    and col not in KNOWN_UNUSABLE_COLS
    and col not in EXCLUDE_COLS
    and pd.api.types.is_numeric_dtype(df_model[col])
]

feature_cols = [
    col for col in feature_cols
    if df_model[col].isna().mean() <= MISSING_VALUE_THRESHOLD
]

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
        ("forest", RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])


# ============================================================================
# HYPERPARAMETER TUNING

param_grid = {
    "forest__n_estimators": [300, 500],
    "forest__max_depth": [8, 12, None],
    "forest__min_samples_split": [10, 30],
    "forest__min_samples_leaf": [4, 8],
    "forest__max_features": ["sqrt"],
}

grid = GridSearchCV(
    estimator=make_pipeline(),
    param_grid=param_grid,
    scoring="roc_auc",
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    verbose=1,
)

print("\nTraining Random Forest with GridSearchCV...")
grid.fit(X_train, y_train, groups=groups_train)

model = grid.best_estimator_

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")


# ============================================================================
# EVALUATION

y_prob = model.predict_proba(X_test)[:, 1]

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
# CONFUSION MATRIX PLOT

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Random Forest Confusion Matrix\nThreshold = {THRESHOLD}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["No AKI", "AKI"])
plt.yticks([0, 1], ["No AKI", "AKI"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(RF_CONFUSION_MATRIX_PATH, dpi=300)
plt.close()


# ============================================================================
# FEATURE IMPORTANCE

rf_model = model.named_steps["forest"]

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 Random Forest features:")
print(feature_importance.head(15).to_string(index=False))


# ============================================================================
# FEATURE IMPORTANCE PLOT

top_features = feature_importance.head(15).sort_values("importance")

plt.figure(figsize=(10, 7))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Feature importance")
plt.title("Random Forest: Top 15 Feature Importances")
plt.tight_layout()
plt.savefig(RF_IMPORTANCE_PLOT_PATH, dpi=300)
plt.close()


# ============================================================================
# SAVE MODEL AND RESULTS

joblib.dump(model, RF_MODEL_PATH)

results = ModelResults("Random Forest")
results.set_metrics(metrics)
results.set_hyperparameters(grid.best_params_)
results.set_feature_importance(
    feature_importance.head(20).to_dict(orient="records")
)
results.set_training_info(
    training_samples=len(X_train),
    test_samples=len(X_test),
    training_time_sec=0.0,
    cv_fold_count=5,
)
results.save_json(str(RF_RESULTS_PATH))

print(f"\nRandom Forest model saved to: {RF_MODEL_PATH}")
print(f"Results saved to: {RF_RESULTS_PATH}")
print(f"Confusion matrix saved to: {RF_CONFUSION_MATRIX_PATH}")
print(f"Feature importance plot saved to: {RF_IMPORTANCE_PLOT_PATH}")


# ============================================================================
# SHAP ANALYSIS

print("\nRunning SHAP analysis...")

imputer = model.named_steps["imputer"]

X_test_imputed = imputer.transform(X_test)
X_test_imputed_df = pd.DataFrame(X_test_imputed, columns=feature_cols)

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_imputed_df)

if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
else:
    shap_vals_class1 = shap_values[:, :, 1]

plt.figure()
shap.summary_plot(
    shap_vals_class1,
    X_test_imputed_df,
    plot_type="dot",
    max_display=15,
    show=False,
)
plt.title("SHAP Summary Plot - AKI Positive Class")
plt.tight_layout()
plt.savefig(SHAP_SUMMARY_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"SHAP summary plot saved to: {SHAP_SUMMARY_PATH}")

plt.figure()
shap.summary_plot(
    shap_vals_class1,
    X_test_imputed_df,
    plot_type="bar",
    max_display=15,
    show=False,
)
plt.title("SHAP Mean Absolute Feature Importance")
plt.tight_layout()
plt.savefig(SHAP_BAR_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"SHAP bar plot saved to: {SHAP_BAR_PATH}")

shap_df = pd.DataFrame(shap_vals_class1, columns=feature_cols)
shap_df.to_csv(SHAP_VALUES_PATH, index=False)
print(f"SHAP values saved to: {SHAP_VALUES_PATH}")
