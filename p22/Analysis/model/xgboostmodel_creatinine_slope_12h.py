# Main writer: Elena Gostiukhina
# Reviewer: Luca Dragutanu
# Contributors: Aadya Charlewar

from __future__ import annotations
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.isotonic import IsotonicRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    brier_score_loss,
)

from graphsxgboost import (
    save_confusion_matrix,
    save_roc_curve,
    save_pr_curve,
    save_feature_importance,
    save_shap_summary,
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
)

GROUP_COL = "subject_id"

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
    # Creatinine features that look beyond the 12h window are blocked.
    "creatinine_delta_48h",
    "creatinine_ratio_7d",
    "creatinine_slope_24h",
    "creatinine_trend_24h",
}

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

LEAKAGE_EXCEPTIONS = {
    "baseline_creatinine",
    "admission_creatinine",
    "creatinine_first_24h",
    "urine_output_first_24h",
    "vanco_ptz_combo",
    "creatinine_slope_12h",
}

# Force this feature into the model feature space when it exists in the dataset,
# even if model_config.IMPROVED_FEATURES_FINAL has not been updated yet.
ALWAYS_KEEP_FEATURES = {
    "creatinine_slope_12h",
}


def load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    print("=" * 70)
    print("STEP 1: DATASET LOADED")
    print("=" * 70)
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


def add_risk_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "vanco_ptz_combo" in df.columns:
        df["vanco_ptz_risk"] = df["vanco_ptz_combo"] * df.get("num_nephrotoxic_drugs", 1)
        if "age" in df.columns:
            df["vanco_age"] = df["vanco_ptz_combo"] * df["age"]
        df["vanco_baseline"] = df["vanco_ptz_combo"] * df.get("baseline_creatinine", 1.0)

    if "sofa_score" in df.columns:
        df["sofa_nephro"] = df["sofa_score"] * df.get("num_nephrotoxic_drugs", 1)
        df["sofa_vanco"] = df["sofa_score"] * df.get("vanco_ptz_combo", 0)

    if "oliguria_flag_24h" in df.columns:
        df["oliguria_vanco_risk"] = df["oliguria_flag_24h"] * df.get("vanco_ptz_combo", 0)

    if "charlson_index" in df.columns and "age" in df.columns:
        df["age_charlson"] = df["age"] * df["charlson_index"]

    return df


def drop_problematic_features(df: pd.DataFrame) -> pd.DataFrame:
    # Keeps the clinically requested 12h slope, but drop other creatinine trend
    # features that may encode information outside the prediction window.
    keep_exceptions_lower = {c.lower() for c in ALWAYS_KEEP_FEATURES}
    drop_keywords = ["creatinine", "los"]

    keep_cols = [
        col for col in df.columns
        if col.lower() in keep_exceptions_lower
           or not any(k in col.lower() for k in drop_keywords)
    ]

    return df[keep_cols].copy()


def prepare_base_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df_model = df.copy()

    # Matches xgboostmodelnoah.py: create exposure, interactions, and missingness
    # indicators centrally in model_config.engineer_features().
    df_model = engineer_features(df_model)

    if "creatinine_slope_12h" not in df_model.columns:
        raise KeyError(
            "Expected feature 'creatinine_slope_12h' not found after "
            "engineer_features(). Add it to the dataset or create it in model_config.py."
        )

    df_model = df_model[df_model[TARGET].notna()].copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    y = df_model[TARGET].copy()
    groups = df_model[GROUP_COL].copy()

    print("\n" + "=" * 70)
    print("STEP 2: TARGET / GROUPS / ENGINEERED FEATURES READY")
    print("=" * 70)
    print(f"Target: {TARGET}")
    print(f"Group column: {GROUP_COL}")
    print("Created features using engineer_features() from model_config.py")
    print("Required model feature present: creatinine_slope_12h")

    return df_model, y, groups


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

    overlap = set(g_train.unique()).intersection(set(g_test.unique()))

    print("\n" + "=" * 70)
    print("STEP 3: GROUPED TRAIN / TEST SPLIT COMPLETE")
    print("=" * 70)
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Train prevalence: {y_train.mean():.2%}")
    print(f"Test prevalence: {y_test.mean():.2%}")
    print(f"Patient overlap: {len(overlap)}")

    if len(overlap) != 0:
        raise RuntimeError("Leakage detected: train/test patient overlap is not zero.")

    return X_train, X_test, y_train, y_test, g_train, g_test


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
    if len(overlap) != 0:
        raise RuntimeError("Leakage detected: train/validation patient overlap is not zero.")

    return X_train_inner, X_val, y_train_inner, y_val, g_train_inner, g_val


def fit_feature_space(X_train_raw: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    all_cols = X_train_raw.columns.tolist()

    # Matches xgboostmodelnoah.py: start from the approved feature set rather
    # than all available columns. Then explicitly add creatinine_slope_12h.
    approved_source = list(dict.fromkeys(list(IMPROVED_FEATURES_FINAL) + list(ALWAYS_KEEP_FEATURES)))
    approved_cols = [
        col for col in approved_source
        if col in all_cols
        and col not in KNOWN_UNUSABLE_COLS
        and col not in EXCLUDE_COLS
        and col not in MANDATORY_DROP_COLS
    ]

    leakage_cols = find_leakage_prone_columns(approved_cols)
    candidate_cols = [col for col in approved_cols if col not in leakage_cols]

    numeric_cols = X_train_raw[candidate_cols].select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool", "uint8", "int8"]
    ).columns.tolist()

    categorical_cols = X_train_raw[candidate_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    kept_numeric_cols = [
        c for c in numeric_cols
        if X_train_raw[c].isna().mean() <= MISSING_VALUE_THRESHOLD
           and X_train_raw[c].nunique(dropna=True) > 1
    ]

    kept_categorical_cols = [
        c for c in categorical_cols
        if X_train_raw[c].isna().mean() <= MISSING_VALUE_THRESHOLD
           and X_train_raw[c].nunique(dropna=True) > 1
    ]

    missing_approved = [col for col in approved_source if col not in all_cols]

    print("\n" + "=" * 70)
    print("FEATURE SPACE FIT ON TRAIN ONLY")
    print("=" * 70)
    print(f"Approved features found: {len(approved_cols)}")
    print(f"Approved features missing from dataset: {len(missing_approved)}")
    print(f"Leakage-prone approved columns removed: {len(leakage_cols)}")
    print(f"Numeric kept: {len(kept_numeric_cols)}")
    print(f"Categorical kept: {len(kept_categorical_cols)}")
    print(f"Uses creatinine_slope_12h: {'creatinine_slope_12h' in kept_numeric_cols + kept_categorical_cols}")

    if leakage_cols:
        print("\nLeakage-prone columns flagged:")
        print(leakage_cols[:50])

    if missing_approved:
        print("\nApproved features not found in dataset:")
        print(missing_approved[:50])

    if "creatinine_slope_12h" not in kept_numeric_cols + kept_categorical_cols:
        raise RuntimeError(
            "creatinine_slope_12h exists but was not kept in the final feature space. "
            "Check missingness threshold, data type, and constant-value filtering."
        )

    return kept_numeric_cols, kept_categorical_cols, leakage_cols


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

    if train_reference_columns is not None:
        X = X.reindex(columns=train_reference_columns, fill_value=0)

    return X, X.columns.tolist()


def build_xgb_params(y_train: pd.Series, param_config: dict) -> dict:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    base_spw = neg / pos if pos > 0 else 1.0

    params = {
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
        "scale_pos_weight": base_spw * param_config["scale_pos_weight_mult"],
        "early_stopping_rounds": param_config.get("early_stopping_rounds", 80),
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }
    return params


def train_xgboost_with_early_stopping(
        X_train_inner: pd.DataFrame,
        y_train_inner: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model: XGBClassifier,
) -> XGBClassifier:
    model.fit(
        X_train_inner,
        y_train_inner,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


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

    print("\n" + "=" * 70)
    print(f"METRICS: {split_name.upper()}")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def get_best_f1_threshold(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    y_val_prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thresh = precision_recall_curve(y_val, y_val_prob)

    if len(thresh) == 0:
        return 0.5

    f1_scores = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    best_idx = int(np.argmax(f1_scores))
    return float(thresh[best_idx])


def get_param_grid() -> list[dict]:
    grid = {
        "n_estimators": [1200, 2000],
        "learning_rate": [0.02, 0.05],
        "max_depth": [3, 4],
        "subsample": [0.75, 0.9],
        "colsample_bytree": [0.8],
        "min_child_weight": [4, 8],
        "reg_alpha": [0.5, 1.5],
        "reg_lambda": [2.5, 5.0],
        "gamma": [1.0, 2.0],
        "scale_pos_weight_mult": [1.0, 1.2],
        "eval_metric": ["auc"],
        "early_stopping_rounds": [80],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    param_list = [dict(zip(keys, combo)) for combo in product(*values)]
    return param_list


def score_for_selection(metrics: dict, selection_metric: str) -> float:
    if selection_metric == "brier_score":
        return -metrics["brier_score"]
    return metrics[selection_metric]


def run_groupkfold_for_params(
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        g_train: pd.Series,
        param_config: dict,
        n_splits: int = 5,
        selection_metric: str = "brier_score",
) -> dict:
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    fold_thresholds = []
    best_iterations = []

    for fold_num, (tr_idx, val_idx) in enumerate(gkf.split(X_train_raw, y_train, groups=g_train), start=1):
        print("\n" + "=" * 70)
        print(f"TUNING PARAM SET | FOLD {fold_num}/{n_splits}")
        print("=" * 70)
        print(param_config)

        X_fold_train_raw = X_train_raw.iloc[tr_idx].copy()
        X_fold_val_raw = X_train_raw.iloc[val_idx].copy()
        y_fold_train = y_train.iloc[tr_idx].copy()
        y_fold_val = y_train.iloc[val_idx].copy()
        g_fold_train = g_train.iloc[tr_idx].copy()
        g_fold_val = g_train.iloc[val_idx].copy()

        overlap = set(g_fold_train.unique()).intersection(set(g_fold_val.unique()))
        if len(overlap) != 0:
            raise RuntimeError(f"Leakage detected in fold {fold_num}: patient overlap is not zero.")

        X_fold_train_raw = add_risk_interactions(X_fold_train_raw)
        X_fold_val_raw = add_risk_interactions(X_fold_val_raw)

        X_fold_train_raw = drop_problematic_features(X_fold_train_raw)
        X_fold_val_raw = drop_problematic_features(X_fold_val_raw)

        numeric_cols, categorical_cols, _ = fit_feature_space(X_fold_train_raw)

        X_fold_train, fold_feature_cols = transform_features(
            X_fold_train_raw,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            train_reference_columns=None,
        )

        X_fold_val, _ = transform_features(
            X_fold_val_raw,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            train_reference_columns=fold_feature_cols,
        )

        model_params = build_xgb_params(y_fold_train, param_config)
        model = XGBClassifier(**model_params)

        model = train_xgboost_with_early_stopping(
            X_train_inner=X_fold_train,
            y_train_inner=y_fold_train,
            X_val=X_fold_val,
            y_val=y_fold_val,
            model=model,
        )

        metrics = evaluate_model(
            model=model,
            X_data=X_fold_val,
            y_data=y_fold_val,
            split_name=f"tuning_fold_{fold_num}",
            threshold=0.5,
        )
        fold_metrics.append(metrics)

        best_threshold = get_best_f1_threshold(model, X_fold_val, y_fold_val)
        fold_thresholds.append(best_threshold)

        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            best_iterations.append(int(model.best_iteration))

    metric_names = fold_metrics[0].keys()
    mean_metrics = {
        name: float(np.mean([m[name] for m in fold_metrics]))
        for name in metric_names
    }
    std_metrics = {
        name: float(np.std([m[name] for m in fold_metrics]))
        for name in metric_names
    }

    result = {
        "params": param_config,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "mean_threshold": float(np.mean(fold_thresholds)) if fold_thresholds else 0.5,
        "mean_best_iteration": int(np.mean(best_iterations)) if best_iterations else param_config["n_estimators"],
        "selection_score": score_for_selection(mean_metrics, selection_metric),
    }
    return result


def tune_hyperparameters(
        X_train_raw: pd.DataFrame,
        y_train: pd.Series,
        g_train: pd.Series,
        n_splits: int = 5,
        selection_metric: str = "brier_score",
        max_candidates: int | None = 20,
) -> dict:
    all_params = get_param_grid()
    if max_candidates is not None:
        all_params = all_params[:max_candidates]

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING STARTED")
    print("=" * 70)
    print(f"Candidates to evaluate: {len(all_params)}")
    print(f"Selection metric: {selection_metric}")

    results = []
    for i, param_config in enumerate(all_params, start=1):
        print("\n" + "#" * 70)
        print(f"TUNING CANDIDATE {i}/{len(all_params)}")
        print("#" * 70)

        result = run_groupkfold_for_params(
            X_train_raw=X_train_raw,
            y_train=y_train,
            g_train=g_train,
            param_config=param_config,
            n_splits=n_splits,
            selection_metric=selection_metric,
        )
        results.append(result)

        print("\nCandidate summary:")
        print(f"Selection score: {result['selection_score']:.6f}")
        print(f"Mean ROC-AUC:     {result['mean_metrics']['roc_auc']:.4f}")
        print(f"Mean PR-AUC:      {result['mean_metrics']['pr_auc']:.4f}")
        print(f"Mean Brier:       {result['mean_metrics']['brier_score']:.4f}")
        print(f"Mean threshold:   {result['mean_threshold']:.4f}")
        print(f"Mean best iter:   {result['mean_best_iteration']}")

    best_result = max(results, key=lambda x: x["selection_score"])

    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETER SET")
    print("=" * 70)
    print(best_result["params"])
    print("Mean CV metrics:")
    for k, v in best_result["mean_metrics"].items():
        print(f"  {k}: {v:.4f}")

    return best_result


def fit_risk_calibrator(y_true: pd.Series, y_prob: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)
    return calibrator


def apply_risk_calibration(calibrator: IsotonicRegression, y_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict(y_prob)


def derive_clinical_risk_thresholds(
        y_true: pd.Series,
        calibrated_val_prob: np.ndarray,
        n_bins: int = 10,
        low_risk_max: float = 0.10,
        high_risk_min: float = 0.30,
) -> tuple[float, float, pd.DataFrame]:
    df_bins = pd.DataFrame({
        "prob": calibrated_val_prob,
        "y": y_true.values,
    }).sort_values("prob").reset_index(drop=True)

    df_bins["bin"] = pd.qcut(df_bins["prob"], q=n_bins, duplicates="drop")

    bin_stats = (
        df_bins.groupby("bin", observed=False)
        .agg(
            prob_min=("prob", "min"),
            prob_max=("prob", "max"),
            mean_pred=("prob", "mean"),
            observed_risk=("y", "mean"),
            n=("y", "size"),
        )
        .reset_index(drop=True)
    )

    low_to_moderate = None
    moderate_to_high = None

    for _, row in bin_stats.iterrows():
        if low_to_moderate is None and row["observed_risk"] >= low_risk_max:
            low_to_moderate = float(row["prob_min"])
        if moderate_to_high is None and row["observed_risk"] > high_risk_min:
            moderate_to_high = float(row["prob_min"])

    if low_to_moderate is None:
        low_to_moderate = float(np.quantile(calibrated_val_prob, 0.33))
    if moderate_to_high is None:
        moderate_to_high = float(np.quantile(calibrated_val_prob, 0.66))

    if moderate_to_high <= low_to_moderate:
        moderate_to_high = max(float(np.quantile(calibrated_val_prob, 0.66)), low_to_moderate + 1e-6)

    return low_to_moderate, moderate_to_high, bin_stats


def assign_risk_category(
        prob: float,
        low_to_moderate: float,
        moderate_to_high: float,
) -> str:
    if prob < low_to_moderate:
        return "Low"
    elif prob < moderate_to_high:
        return "Moderate"
    return "High"


def get_risk_score(
        model: XGBClassifier,
        X: pd.DataFrame,
        calibrator: IsotonicRegression,
) -> tuple[np.ndarray, np.ndarray]:
    raw_prob = model.predict_proba(X)[:, 1]
    calibrated_prob = apply_risk_calibration(calibrator, raw_prob)
    score = calibrated_prob * 100
    return calibrated_prob, score


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)
    print("Creatinine-related columns in raw dataset:")
    print([c for c in df.columns if "creatinine" in c.lower()])

    df_model, y, groups = prepare_base_dataframe(df)
    X_raw = df_model.copy()

    X_train_raw, X_test_raw, y_train, y_test, g_train, g_test = group_train_test_split(
        X_df=X_raw,
        y=y,
        groups=groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    best_tuning_result = tune_hyperparameters(
        X_train_raw=X_train_raw,
        y_train=y_train,
        g_train=g_train,
        n_splits=5,
        selection_metric="brier_score",
        max_candidates=16,
    )

    best_params = best_tuning_result["params"]
    mean_best_threshold = best_tuning_result["mean_threshold"]
    tuned_n_estimators = best_tuning_result["mean_best_iteration"]

    X_train_inner_raw, X_val_raw, y_train_inner, y_val, g_train_inner, g_val = group_train_val_split(
        X_train=X_train_raw,
        y_train=y_train,
        g_train=g_train,
        val_size=0.2,
    )

    print("\n" + "=" * 70)
    print("FINAL TRAINING PIPELINE")
    print("=" * 70)

    X_train_inner_raw = add_risk_interactions(X_train_inner_raw)
    X_val_raw = add_risk_interactions(X_val_raw)
    X_test_raw = add_risk_interactions(X_test_raw)

    X_train_inner_raw = drop_problematic_features(X_train_inner_raw)
    X_val_raw = drop_problematic_features(X_val_raw)
    X_test_raw = drop_problematic_features(X_test_raw)

    numeric_cols, categorical_cols, leakage_cols = fit_feature_space(X_train_inner_raw)

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

    final_params = build_xgb_params(y_train_inner, best_params)
    final_params["n_estimators"] = tuned_n_estimators

    print("\nBest params selected for final training:")
    print(final_params)

    model = XGBClassifier(**final_params)
    model = train_xgboost_with_early_stopping(
        X_train_inner=X_train_inner,
        y_train_inner=y_train_inner,
        X_val=X_val,
        y_val=y_val,
        model=model,
    )

    train_metrics = evaluate_model(
        model=model,
        X_data=X_train_inner,
        y_data=y_train_inner,
        split_name="train_inner",
        threshold=0.5,
    )

    val_metrics = evaluate_model(
        model=model,
        X_data=X_val,
        y_data=y_val,
        split_name="validation",
        threshold=0.5,
    )

    test_metrics_default = evaluate_model(
        model=model,
        X_data=X_test,
        y_data=y_test,
        split_name="test_default_threshold",
        threshold=0.5,
    )

    best_threshold = get_best_f1_threshold(model, X_val, y_val)

    print(f"\nBest threshold on final validation split: {best_threshold:.4f}")
    print(f"Mean best threshold from tuning CV:       {mean_best_threshold:.4f}")

    test_metrics_best = evaluate_model(
        model=model,
        X_data=X_test,
        y_data=y_test,
        split_name="test_best_threshold",
        threshold=best_threshold,
    )

    val_prob_raw = model.predict_proba(X_val)[:, 1]
    calibrator = fit_risk_calibrator(y_val, val_prob_raw)
    val_prob_calibrated = apply_risk_calibration(calibrator, val_prob_raw)

    low_to_moderate, moderate_to_high, clinical_bin_stats = derive_clinical_risk_thresholds(
        y_true=y_val,
        calibrated_val_prob=val_prob_calibrated,
        n_bins=10,
        low_risk_max=0.10,
        high_risk_min=0.30,
    )

    print("\n" + "=" * 70)
    print("CLINICAL RISK STRATIFICATION FROM VALIDATION DATA")
    print("=" * 70)
    print("Low risk:      observed AKI < 10%")
    print("Moderate risk: observed AKI 10-30%")
    print("High risk:     observed AKI > 30%")
    print(f"Derived threshold Low -> Moderate: {low_to_moderate:.4f}")
    print(f"Derived threshold Moderate -> High: {moderate_to_high:.4f}")
    print("\nValidation bin stats used to derive thresholds:")
    print(clinical_bin_stats)

    test_prob, test_score = get_risk_score(model, X_test, calibrator)
    risk_df = pd.DataFrame({
        "probability": test_prob,
        "risk_score": test_score,
        "risk_category": [
            assign_risk_category(p, low_to_moderate, moderate_to_high) for p in test_prob
        ],
    })

    print("\nRisk score preview:")
    print(risk_df.head())

    category_summary = (
        pd.DataFrame({
            "y_true": y_test.values,
            "probability": test_prob,
            "risk_score": test_score,
            "risk_category": [
                assign_risk_category(p, low_to_moderate, moderate_to_high) for p in test_prob
            ],
        })
        .groupby("risk_category", observed=False)
        .agg(
            n=("y_true", "size"),
            observed_aki_rate=("y_true", "mean"),
            mean_predicted_risk=("probability", "mean"),
            mean_risk_score=("risk_score", "mean"),
        )
        .reset_index()
    )

    print("\nClinical summary by risk category on test set:")
    print(category_summary)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nBest CV params:")
    print(best_params)

    print("\nBest CV mean metrics:")
    for k, v in best_tuning_result["mean_metrics"].items():
        print(f"  {k}: {v:.4f}")

    print("\nTrain metrics:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nValidation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics (threshold=0.5):")
    for k, v in test_metrics_default.items():
        print(f"  {k}: {v:.4f}")

    print("\nTest metrics (best validation threshold):")
    for k, v in test_metrics_best.items():
        print(f"  {k}: {v:.4f}")

    print(
        f"\nClinical category thresholds: "
        f"Low<{low_to_moderate:.4f}, "
        f"Moderate<{moderate_to_high:.4f}, "
        f"High>={moderate_to_high:.4f}"
    )

    y_test_prob = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_test_prob)
    print(f"\nBrier score on test: {brier:.6f}")

    save_confusion_matrix(model, X_test, y_test, "test_best_threshold", best_threshold)
    save_roc_curve(model, X_test, y_test, "test")
    save_pr_curve(model, X_test, y_test, "test")
    save_feature_importance(model, train_feature_cols, top_n=30)
    save_shap_summary(model, X_val, "validation")
