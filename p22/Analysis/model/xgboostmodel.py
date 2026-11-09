# Main writer: Elena Gostiukhina
# Reviewer: Luca Dragutanu
# Contributors: Aadya Charlewar

from __future__ import annotations
#import re
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

from xgboost import XGBClassifier

from model_config import (
    DATASET_PATH,
    TARGET,
    EXCLUDE_COLS,
    IMPROVED_FEATURES_FINAL,
    KNOWN_UNUSABLE_COLS,
    engineer_features,
    MISSING_VALUE_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
    XGBOOST_PARAMS,
    XGB_TUNING_SPACE,
)


#------
# Leakage controls

GROUP_COL = "subject_id"   # patient identifier for grouped splitting

# Columns that should almost never be used as features
MANDATORY_DROP_COLS = {
    TARGET,
    GROUP_COL,
    "hadm_id",
    "stay_id",
    "icu_stay_id",
    "admission_id",
    "patient_id",
    "encounter_id",
    "charttime",
    "storetime",
    "event_time",
    "specimen_id",
    "creatinine_delta_48h",
    "creatinine_ratio_7d",
    "creatinine_slope_24h",
    "creatinine_trend_24h",
}

# Heuristic keywords for likely leakage columns
# Expand this list based on your dataset
LEAKAGE_KEYWORDS = [
    "aki",
    "acute_kidney_injury",
    "kdigo",
    "stage",
    "dialysis",
    "rrt",
    "renal_replacement",
    "post",
    "after_",
    "_after",
    "future",
    "outcome",
    "label",
    "target",
    "during_stay",
    "whole_stay",
    "max_creatinine",
    "min_creatinine",
    "peak_creatinine",
    "delta_creatinine",
    "creatinine_rise",
    "urine_output_6h_after",
    "urine_output_12h_after",
    "urine_output_24h_after_event",
    "discharge",
    "death",
    "mortality",
]

# Columns allowed even if keyword matching is too aggressive
LEAKAGE_EXCEPTIONS = {
    "baseline_creatinine",
    "admission_creatinine",
    "creatinine_first_24h",
    "urine_output_first_24h",
    "vanco_ptz_combo",
}


# -------
# Step 1. Load dataset
def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

   #
    print("STEP 1: DATASET LOADED")
   #
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")

    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found in dataset.")

    if GROUP_COL not in df.columns:
        raise KeyError(
            f"Group column '{GROUP_COL}' not found. "
            f"You need patient-level grouping to avoid leakage."
        )

    print(f"Target prevalence: {df[TARGET].mean():.2%}")
    print(f"Unique {GROUP_COL}: {df[GROUP_COL].nunique()}")

    print("\nTop 15 columns by missingness:")
    print(df.isna().mean().sort_values(ascending=False).head(15))

    return df


# ---
# Step 2. Create allowed engineered features only
def prepare_base_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df_model = df.copy()

    # Creates exposure + engineered interaction features + missingness indicators
    df_model = engineer_features(df_model)

    # Keep only rows with a valid target
    df_model = df_model[df_model[TARGET].notna()].copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    y = df_model[TARGET].copy()
    groups = df_model[GROUP_COL].copy()

   #
    print("STEP 2: TARGET / GROUPS / ENGINEERED FEATURES READY")
   #
    print(f"Target: {TARGET}")
    print(f"Group column: {GROUP_COL}")
    print("Created features using engineer_features() from model_config.py")

    return df_model, y, groups

#-----
# Leakage detection helpers
def is_likely_leakage_column(col: str) -> bool:
    col_lower = col.lower()

    if col in LEAKAGE_EXCEPTIONS:
        return False

    for kw in LEAKAGE_KEYWORDS:
        if kw in col_lower:
            return True

    return False


def find_leakage_prone_columns(columns: list[str]) -> list[str]:
    flagged = []
    for col in columns:
        if col in MANDATORY_DROP_COLS:
            flagged.append(col)
        elif is_likely_leakage_column(col):
            flagged.append(col)
    return sorted(set(flagged))


#-----
# # Step 3. Grouped split: train/test
def group_train_test_split(
        X_df: pd.DataFrame,
        y: pd.Series,
        groups: pd.Series,
        test_size: float,
        random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(X_df, y, groups=groups))

    X_train = X_df.iloc[train_idx].copy()
    X_test = X_df.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    g_train = groups.iloc[train_idx].copy()
    g_test = groups.iloc[test_idx].copy()

    train_patients = set(g_train.unique())
    test_patients = set(g_test.unique())
    overlap = train_patients.intersection(test_patients)

   #
    print("STEP 3: GROUPED TRAIN / TEST SPLIT COMPLETE")
   #
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train prevalence: {y_train.mean():.2%}")
    print(f"Test prevalence: {y_test.mean():.2%}")
    print(f"Train unique patients: {len(train_patients)}")
    print(f"Test unique patients: {len(test_patients)}")
    print(f"Patient overlap: {len(overlap)}")

    if len(overlap) != 0:
        raise RuntimeError("Leakage detected: train/test patient overlap is not zero.")

    return X_train, X_test, y_train, y_test, g_train, g_test

#-----
# Step 4. Grouped split: train/validation
def group_train_val_split(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        g_train: pd.Series,
        val_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size,
        random_state=RANDOM_STATE,
    )

    train_inner_idx, val_idx = next(splitter.split(X_train, y_train, groups=g_train))

    X_train_inner = X_train.iloc[train_inner_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_train_inner = y_train.iloc[train_inner_idx].copy()
    y_val = y_train.iloc[val_idx].copy()
    g_train_inner = g_train.iloc[train_inner_idx].copy()
    g_val = g_train.iloc[val_idx].copy()

    overlap = set(g_train_inner.unique()).intersection(set(g_val.unique()))

   #
    print("STEP 4: GROUPED TRAIN / VALIDATION SPLIT COMPLETE")
   #
    print(f"Inner train shape: {X_train_inner.shape}")
    print(f"Validation shape:  {X_val.shape}")
    print(f"Inner train prevalence: {y_train_inner.mean():.2%}")
    print(f"Validation prevalence:  {y_val.mean():.2%}")
    print(f"Patient overlap: {len(overlap)}")

    if len(overlap) != 0:
        raise RuntimeError("Leakage detected: train/validation patient overlap is not zero.")

    return X_train_inner, X_val, y_train_inner, y_val, g_train_inner, g_val


#-----
# Step 5. Learn preprocessing only from train
def fit_feature_space(X_train_inner_raw: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    all_cols = X_train_inner_raw.columns.tolist()

    # Only use the approved improved feature set from model_config.py.
    # This replaces the old logic where we selected from all available numeric columns.
    approved_cols = [
        col for col in IMPROVED_FEATURES_FINAL
        if col in all_cols
        and col not in KNOWN_UNUSABLE_COLS
        and col not in EXCLUDE_COLS
        and col not in MANDATORY_DROP_COLS
    ]

    # Keep the leakage check as an extra safety layer.
    leakage_cols = find_leakage_prone_columns(approved_cols)

    candidate_cols = [
        col for col in approved_cols
        if col not in leakage_cols
    ]

    # Split numeric / categorical on train only.
    numeric_cols = X_train_inner_raw[candidate_cols].select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool", "uint8", "int8"]
    ).columns.tolist()

    categorical_cols = X_train_inner_raw[candidate_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Missingness filtering based on train only.
    kept_numeric_cols = [
        c for c in numeric_cols
        if X_train_inner_raw[c].isna().mean() <= MISSING_VALUE_THRESHOLD
    ]

    kept_categorical_cols = [
        c for c in categorical_cols
        if X_train_inner_raw[c].isna().mean() <= MISSING_VALUE_THRESHOLD
    ]

    # Constant filtering based on train only.
    kept_numeric_cols = [
        c for c in kept_numeric_cols
        if X_train_inner_raw[c].nunique(dropna=True) > 1
    ]

    kept_categorical_cols = [
        c for c in kept_categorical_cols
        if X_train_inner_raw[c].nunique(dropna=True) > 1
    ]

    missing_approved = [
        col for col in IMPROVED_FEATURES_FINAL
        if col not in all_cols
    ]

   #
    print("STEP 5: FEATURE SPACE FIT ON TRAIN ONLY")
   #
    print(f"Approved improved features found: {len(approved_cols)}")
    print(f"Approved features missing from dataset: {len(missing_approved)}")
    print(f"Leakage-prone approved columns removed: {len(leakage_cols)}")
    print(f"Numeric kept: {len(kept_numeric_cols)}")
    print(f"Categorical kept: {len(kept_categorical_cols)}")

    if leakage_cols:
        print("\nLeakage-prone columns flagged:")
        print(leakage_cols[:50])

    if missing_approved:
        print("\nApproved features not found in dataset:")
        print(missing_approved[:50])

    return kept_numeric_cols, kept_categorical_cols, leakage_cols
#-----
# Step 6. Transform split using train-derived feature space
def transform_features(
        df_raw: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        train_reference_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    X_num = df_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df_raw.index)

    if categorical_cols:
        X_cat = pd.get_dummies(
            df_raw[categorical_cols].copy(),
            columns=categorical_cols,
            dummy_na=True,
            drop_first=False,
            dtype=int,
        )
    else:
        X_cat = pd.DataFrame(index=df_raw.index)

    X = pd.concat([X_num, X_cat], axis=1)

    # if train reference columns are provided, align strictly to train
    if train_reference_columns is not None:
        X = X.reindex(columns=train_reference_columns, fill_value=0)

    return X, X.columns.tolist()


#----
# Step 7. Fit XGBoost
def setup_xgboost_model(y_train_inner: pd.Series) -> XGBClassifier:
    pos = int(y_train_inner.sum())
    neg = int((y_train_inner == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        **XGBOOST_PARAMS,
    )
    
   #
    print("STEP 7: XGBOOST INITIALIZED")
   #
    print(f"scale_pos_weight = {scale_pos_weight:.3f}")
    
    return model

#------
# Step 8. Evaluation
def evaluate_model(
        model: XGBClassifier,
        X_data: pd.DataFrame,
        y_data: pd.Series,
        split_name: str,
        threshold: float = 0.5,
) -> dict:
    y_prob = model.predict_proba(X_data)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_data, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "roc_auc": roc_auc_score(y_data, y_prob),
        "pr_auc": average_precision_score(y_data, y_prob),
        "brier_score": brier_score_loss(y_data, y_prob),
        "accuracy": accuracy_score(y_data, y_pred),
        "precision": precision_score(y_data, y_pred, zero_division=0),
        "recall": recall_score(y_data, y_pred, zero_division=0),
        "f1": f1_score(y_data, y_pred, zero_division=0),
        "specificity": specificity,
        "sensitivity": sensitivity,
    }

   #
    print(f"METRICS: {split_name.upper()}")
   #
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

# ADDING HYPERPARAMETER TUNING
def tune_xgboost_hyperparameters(
    X_train_inner: pd.DataFrame,
    y_train_inner: pd.Series,
    groups_train_inner: pd.Series,
    param_distributions: dict,
    n_iter: int = 30,
    cv_splits: int = 3,     
) -> dict:
    """
    Hyperparameter tuning using group-aware cross-validation.
    Returns the best parameters found.
    """

    #Base model
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=None,
    )

    # GroupKFold for cross-validation
    gfk = GroupKFold(n_splits=cv_splits)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=gfk,
        scoring = "roc_auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    random_search.fit(X_train_inner, y_train_inner, groups=groups_train_inner)

    print("STEP 12: HYPERPARAMETER TUNING COMPLETE")
   #
    print(f"Best CV ROC-AUC: {random_search.best_score_:.4f}")
    print("Best parameters:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")

    return random_search.best_params_



#------
# Main pipeline
if __name__ == "__main__":
    # Step 1
    df = load_dataset(DATASET_PATH)

    # Step 2
    df_model, y, groups = prepare_base_dataframe(df)

    # Raw X before any learned preprocessing
    X_raw = df_model.copy()

    # Step 3: patient-level train/test split
    X_train_raw, X_test_raw, y_train, y_test, g_train, g_test = group_train_test_split(
        X_df=X_raw,
        y=y,
        groups=groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Step 4: patient-level train/val split
    X_train_inner_raw, X_val_raw, y_train_inner, y_val, g_train_inner, g_val = group_train_val_split(
        X_train=X_train_raw,
        y_train=y_train,
        g_train=g_train,
        val_size=0.2,
    )

    # Step 5: fit feature space ONLY on train_inner
    numeric_cols, categorical_cols, leakage_cols = fit_feature_space(X_train_inner_raw)

    # Step 6: transform train/val/test with train-derived columns only
    X_train_inner, train_feature_cols = transform_features(
        X_train_inner_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        train_reference_columns=None,
    )

    X_val, _ = transform_features(
        X_val_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        train_reference_columns=train_feature_cols,
    )

    X_test, _ = transform_features(
        X_test_raw,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        train_reference_columns=train_feature_cols,
    )

   #
    print("STEP 6: FINAL ALIGNED MATRICES")
   #
    print(f"X_train_inner: {X_train_inner.shape}")
    print(f"X_val:         {X_val.shape}")
    print(f"X_test:        {X_test.shape}")
    print(f"Feature count: {len(train_feature_cols)}")
    print(f"Total NaNs in train: {int(X_train_inner.isna().sum().sum())}")
    print(f"Total NaNs in val:   {int(X_val.isna().sum().sum())}")
    print(f"Total NaNs in test:  {int(X_test.isna().sum().sum())}")

    #----
    #  HYPERPARAMETER TUNING (Step 12) 
    best_params = tune_xgboost_hyperparameters(
        X_train_inner=X_train_inner,
        y_train_inner=y_train_inner,
        groups_train_inner=g_train_inner,
        param_distributions=XGB_TUNING_SPACE,
        n_iter=30,
    )

   
    final_params =  {k: v for k, v in XGBOOST_PARAMS.items() if k != 'early_stopping_rounds'}   
    final_params.update(best_params)
    final_params['early_stopping_rounds'] = 30

    # Compute scale_pos_weight from training data
    pos = int(y_train_inner.sum())
    neg = int((y_train_inner == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    final_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        **final_params
    )

    # Train with early stopping on the validation set
    final_model.fit(
        X_train_inner, y_train_inner,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

   #
    print("FINAL MODEL TRAINED WITH TUNED HYPERPARAMETERS")
   #
    if hasattr(final_model, "best_iteration") and final_model.best_iteration is not None:
        print(f"Best iteration: {final_model.best_iteration}")
    if hasattr(final_model, "best_score") and final_model.best_score is not None:
        print(f"Best validation AUC: {final_model.best_score:.4f}")

    #-------
    #  EVALUATION 
    train_metrics = evaluate_model(
        model=final_model,
        X_data=X_train_inner,
        y_data=y_train_inner,
        split_name="train_inner",
        threshold=0.5,
    )

    val_metrics = evaluate_model(
        model=final_model,
        X_data=X_val,
        y_data=y_val,
        split_name="validation",
        threshold=0.5,
    )

    test_metrics = evaluate_model(
        model=final_model,
        X_data=X_test,
        y_data=y_test,
        split_name="test",
        threshold=0.5,
    )

   #
    print("FINAL SUMMARY")
   #
    print(f"Selected numeric columns: {len(numeric_cols)}")
    print(f"Selected categorical columns: {len(categorical_cols)}")
    print(f"Final encoded feature count: {len(train_feature_cols)}")
    print(f"Leakage-prone dropped columns: {len(leakage_cols)}")

    print("\nTrain metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
