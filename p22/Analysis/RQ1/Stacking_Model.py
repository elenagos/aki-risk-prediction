# Main writer: Noah Charpentier
# Reviewer: Aadya Charlewar
# Contributors: Bartlomiej Kaczor

from pathlib import Path
import sys
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
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
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier


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


# ============================================================================
# OUTPUT PATHS

STACK_MODEL_PATH = Path(MODELS_DIR) / "stacking_model.pkl"
STACK_COMPARISON_PATH = Path(RESULTS_DIR) / "stacking_full_comparison.json"

STACK_ROC_PATH = Path(RESULTS_DIR) / "stacking_roc_curve.png"
STACK_PR_PATH = Path(RESULTS_DIR) / "stacking_precision_recall_curve.png"
STACK_CM_PATH = Path(RESULTS_DIR) / "stacking_confusion_matrix.png"
STACK_CALIBRATION_PATH = Path(RESULTS_DIR) / "stacking_calibration_curve.png"
STACK_MODEL_COMPARISON_BY_METRIC_PATH = Path(RESULTS_DIR) / "stacking_model_comparison_by_metric.png"
STACK_META_COEF_PATH = Path(RESULTS_DIR) / "stacking_meta_model_coefficients.png"


# ============================================================================
# XGBOOST CONFIG FOR STACKING

XGB_STACKING_CONFIG = {
    "n_estimators": 1200,
    "learning_rate": 0.02,
    "max_depth": 3,
    "subsample": 0.75,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "reg_alpha": 0.5,
    "reg_lambda": 2.5,
    "gamma": 1.0,
    "scale_pos_weight_mult": 1.0,
    "eval_metric": "auc",
}


# ============================================================================
# LOAD DATA

print("=" * 10)
print("STACKING ENSEMBLE MODEL FOR AKI PREDICTION")
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

# Encode sex if needed.
if "sex" in df_model.columns and not pd.api.types.is_numeric_dtype(df_model["sex"]):
    df_model["sex"] = (df_model["sex"].astype(str).str.upper() == "M").astype(int)

# Convert boolean columns to integers if any remain.
bool_cols = df_model.select_dtypes(include=["bool"]).columns
df_model[bool_cols] = df_model[bool_cols].astype(int)

# Use ONLY the approved improved feature list from model_config.py.
feature_cols = [
    col for col in IMPROVED_FEATURES_FINAL
    if col in df_model.columns
    and col not in KNOWN_UNUSABLE_COLS
    and col not in EXCLUDE_COLS
    and pd.api.types.is_numeric_dtype(df_model[col])
]

# Drop columns with too much missingness.
feature_cols = [
    col for col in feature_cols
    if df_model[col].isna().mean() <= MISSING_VALUE_THRESHOLD
]

# Drop constant columns.
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
# PATIENT-LEVEL TRAIN / TEST SPLIT

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()

groups_train = groups.iloc[train_idx].copy()
groups_test = groups.iloc[test_idx].copy()

train_patients = set(groups_train.unique())
test_patients = set(groups_test.unique())
overlap = train_patients.intersection(test_patients)

if len(overlap) > 0:
    raise RuntimeError("Patient-level leakage detected: train/test patient overlap is not zero.")

print("\nPatient-level split:")
print(f"Train: {len(X_train)} rows, AKI rate = {y_train.mean():.2%}")
print(f"Test:  {len(X_test)} rows, AKI rate = {y_test.mean():.2%}")
print(f"Train unique patients: {groups_train.nunique()}")
print(f"Test unique patients:  {groups_test.nunique()}")
print(f"Patient overlap: {len(overlap)}")


# ============================================================================
# MODEL DEFINITIONS

class PercentileClipper(BaseEstimator, TransformerMixin):

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_q)
        self.upper_bounds_ = X_df.quantile(self.upper_q)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        X_df = X_df.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1,
        )
        return X_df.values


def make_logistic_regression():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", PercentileClipper(lower_q=0.01, upper_q=0.99)),
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.5,
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def make_random_forest():
    return Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTATION_STRATEGY)),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_xgb_params(y_train_data, param_config):
    """
    Builds XGBoost parameters using the same scale_pos_weight logic
    as the standalone XGBoost script.
    """
    pos = int(y_train_data.sum())
    neg = int((y_train_data == 0).sum())
    base_scale_pos_weight = neg / pos if pos > 0 else 1.0

    return {
        "objective": "binary:logistic",
        "eval_metric": param_config.get("eval_metric", "auc"),
        "random_state": RANDOM_STATE,
        "n_estimators": param_config["n_estimators"],
        "learning_rate": param_config["learning_rate"],
        "max_depth": param_config["max_depth"],
        "subsample": param_config["subsample"],
        "colsample_bytree": param_config["colsample_bytree"],
        "min_child_weight": param_config["min_child_weight"],
        "reg_alpha": param_config["reg_alpha"],
        "reg_lambda": param_config["reg_lambda"],
        "gamma": param_config["gamma"],
        "scale_pos_weight": base_scale_pos_weight * param_config["scale_pos_weight_mult"],
        "tree_method": "hist",
        "n_jobs": -1,
        "verbosity": 0,
    }


def make_xgboost(y_train_data):
    xgb_params = build_xgb_params(y_train_data, XGB_STACKING_CONFIG)

    return Pipeline([
        ("imputer", SimpleImputer(strategy=IMPUTATION_STRATEGY)),
        ("model", XGBClassifier(**xgb_params)),
    ])


base_models = [
    ("logistic_regression", make_logistic_regression()),
    ("random_forest", make_random_forest()),
    ("xgboost", make_xgboost(y_train)),
]

meta_model = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    random_state=RANDOM_STATE,
    class_weight="balanced",
)


print("\nBase models used in stacking:")
for name, model in base_models:
    print(f"\n{name}")
    print(model)


# ============================================================================
# MANUAL GROUP-AWARE STACKING

print("\n" + "=" * 10)
print("TRAINING GROUP-AWARE STACKING ENSEMBLE")
print("=" * 10)

cv = GroupKFold(n_splits=5)

# meta_train will contain out-of-fold probabilities.
# Rows = training rows.
# Columns = base model probabilities.
meta_train = np.zeros((len(X_train), len(base_models)))

fitted_base_models = {}

for model_index, (model_name, base_model) in enumerate(base_models):
    print(f"\nGenerating out-of-fold predictions for: {model_name}")

    oof_prob = cross_val_predict(
        estimator=base_model,
        X=X_train,
        y=y_train,
        groups=groups_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    meta_train[:, model_index] = oof_prob

    print(f"Fitting final {model_name} on the full training set...")
    final_base_model = clone(base_model)
    final_base_model.fit(X_train, y_train)

    fitted_base_models[model_name] = final_base_model


meta_feature_names = [model_name for model_name, _ in base_models]

print("\nMeta-training matrix:")
print(f"Shape: {meta_train.shape}")
print(pd.DataFrame(meta_train, columns=meta_feature_names).head())

print("\nTraining meta-model on out-of-fold predictions...")
meta_model.fit(meta_train, y_train)

print("Stacking ensemble trained successfully.")


# ============================================================================
# EVALUATION HELPERS

def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def print_metrics(name, metrics):
    print("\n" + "=" * 10)
    print(name)
    print("=" * 10)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

THRESHOLD = 0.47

# Create meta-features for the test set using fitted base models.
meta_test = np.zeros((len(X_test), len(base_models)))

for model_index, (model_name, _) in enumerate(base_models):
    meta_test[:, model_index] = fitted_base_models[model_name].predict_proba(X_test)[:, 1]

print("\nMeta-test matrix:")
print(f"Shape: {meta_test.shape}")
print(pd.DataFrame(meta_test, columns=meta_feature_names).head())

y_prob_stack = meta_model.predict_proba(meta_test)[:, 1]
y_pred_stack = (y_prob_stack >= THRESHOLD).astype(int)

stack_metrics = compute_metrics(y_test, y_prob_stack, threshold=THRESHOLD)

print_metrics("TEST SET PERFORMANCE: STACKING ENSEMBLE", stack_metrics)

print("\nClassification report:")
print(classification_report(y_test, y_pred_stack, zero_division=0))

base_results = {}

for model_name, fitted_model in fitted_base_models.items():
    y_prob_base = fitted_model.predict_proba(X_test)[:, 1]
    y_pred_base = (y_prob_base >= THRESHOLD).astype(int)

    metrics = compute_metrics(y_test, y_prob_base, threshold=THRESHOLD)

    base_results[model_name] = {
        "probabilities": y_prob_base,
        "predictions": y_pred_base,
        "metrics": metrics,
    }

    print_metrics(f"TEST SET PERFORMANCE: {model_name.upper()}", metrics)

all_model_metrics = {"stacking": stack_metrics}

for model_name, result in base_results.items():
    all_model_metrics[model_name] = result["metrics"]


# ============================================================================
# PLOTS

# ----------------------------------------------------------------------------
# 1. ROC CURVE COMPARISON

plt.figure(figsize=(8, 6))

fpr, tpr, _ = roc_curve(y_test, y_prob_stack)
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Stacking (AUC = {roc_auc_score(y_test, y_prob_stack):.3f})"
)

for model_name, result in base_results.items():
    y_prob_base = result["probabilities"]
    fpr, tpr, _ = roc_curve(y_test, y_prob_base)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {roc_auc_score(y_test, y_prob_base):.3f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(STACK_ROC_PATH, dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 2. PRECISION-RECALL CURVE COMPARISON

plt.figure(figsize=(8, 6))

precision_values, recall_values, _ = precision_recall_curve(y_test, y_prob_stack)
plt.plot(
    recall_values,
    precision_values,
    linewidth=2,
    label=f"Stacking (PR-AUC = {average_precision_score(y_test, y_prob_stack):.3f})"
)

for model_name, result in base_results.items():
    y_prob_base = result["probabilities"]
    precision_values, recall_values, _ = precision_recall_curve(y_test, y_prob_base)

    plt.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"{model_name} (PR-AUC = {average_precision_score(y_test, y_prob_base):.3f})"
    )

baseline_prevalence = y_test.mean()
plt.axhline(
    baseline_prevalence,
    linestyle="--",
    label=f"AKI prevalence = {baseline_prevalence:.3f}"
)

plt.xlabel("Recall / Sensitivity")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve Comparison")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(STACK_PR_PATH, dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 3. CONFUSION MATRIX FOR STACKING MODEL

cm = confusion_matrix(y_test, y_pred_stack)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title(f"Stacking Confusion Matrix\nThreshold = {THRESHOLD}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.xticks([0, 1], ["No AKI", "AKI"])
plt.yticks([0, 1], ["No AKI", "AKI"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig(STACK_CM_PATH, dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 4. CALIBRATION CURVE

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test,
    y_prob_stack,
    n_bins=10,
    strategy="quantile",
)

plt.figure(figsize=(7, 6))
plt.plot(
    mean_predicted_value,
    fraction_of_positives,
    marker="o",
    linewidth=2,
    label="Stacking"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed AKI frequency")
plt.title("Stacking Model - Calibration Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(STACK_CALIBRATION_PATH, dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 5. MODEL PERFORMANCE COMPARISON BY METRIC

comparison_metrics = [
    "roc_auc",
    "pr_auc",
    "recall_sensitivity",
    "specificity",
    "precision",
    "f1",
]

comparison_df = pd.DataFrame(all_model_metrics).T[comparison_metrics]
comparison_df = comparison_df.rename(columns={"recall_sensitivity": "recall"})

metric_labels = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "recall": "Recall",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
}

model_labels = {
    "stacking": "Stacking",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

plot_df = comparison_df.rename(columns=metric_labels)
plot_df = plot_df.rename(index=model_labels)

plt.figure(figsize=(11, 6))

metrics = plot_df.columns.tolist()
models = plot_df.index.tolist()

x = np.arange(len(metrics))
width = 0.18

for i, model_name in enumerate(models):
    plt.bar(
        x + i * width,
        plot_df.loc[model_name, metrics],
        width,
        label=model_name,
    )

plt.xticks(
    x + width * (len(models) - 1) / 2,
    metrics,
    rotation=0,
)

plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.title("Model Performance Comparison by Metric")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(STACK_MODEL_COMPARISON_BY_METRIC_PATH, dpi=300)
plt.close()


# ----------------------------------------------------------------------------
# 6. META-MODEL COEFFICIENTS

if hasattr(meta_model, "coef_"):
    coefs = meta_model.coef_[0]

    meta_coef_df = pd.DataFrame({
        "base_model": meta_feature_names,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=True)

    plt.figure(figsize=(7, 5))
    plt.barh(meta_coef_df["base_model"], meta_coef_df["coefficient"])
    plt.axvline(0, linestyle="--")
    plt.xlabel("Meta-model coefficient")
    plt.title("Stacking Meta-Model Coefficients")
    plt.tight_layout()
    plt.savefig(STACK_META_COEF_PATH, dpi=300)
    plt.close()

    print("\nMeta-model coefficients:")
    print(meta_coef_df.sort_values("abs_coefficient", ascending=False).to_string(index=False))


# ============================================================================
# SAVE MODEL PACKAGE AND RESULTS

stacking_package = {
    "base_models": fitted_base_models,
    "meta_model": meta_model,
    "feature_cols": feature_cols,
    "base_model_order": meta_feature_names,
    "threshold": THRESHOLD,
    "xgboost_config_used": XGB_STACKING_CONFIG,
}

joblib.dump(stacking_package, STACK_MODEL_PATH)


with open(STACK_COMPARISON_PATH, "w", encoding="utf-8") as f:
    json.dump(all_model_metrics, f, indent=4)

print("\n" + "=" * 10)
print("STACKING MODEL SAVED")
print("=" * 10)

print(f"Model saved to: {STACK_MODEL_PATH}")
print(f"Full comparison saved to: {STACK_COMPARISON_PATH}")
print("\nPlots saved:")
print(f"  ROC curve: {STACK_ROC_PATH}")
print(f"  Precision-Recall curve: {STACK_PR_PATH}")
print(f"  Confusion matrix: {STACK_CM_PATH}")
print(f"  Calibration curve: {STACK_CALIBRATION_PATH}")
print(f"  Model comparison by metric: {STACK_MODEL_COMPARISON_BY_METRIC_PATH}")
print(f"  Meta-model coefficients: {STACK_META_COEF_PATH}")

print("\nFinal stacking metrics:")
for metric, value in stack_metrics.items():
    print(f"  {metric}: {value:.4f}")
