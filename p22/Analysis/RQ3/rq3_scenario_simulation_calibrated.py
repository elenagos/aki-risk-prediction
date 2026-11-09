# Main writer:Kimi Knaider
# Reviewer: Luca Dragutanu
# Contributors:Elena 

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS AND CONFIG LOADING
# =============================================================================


def find_analysis_dir() -> Path:
    """
    Finds the Analysis directory without relying on PyCharm's interpreter paths.

    Expected project layout:
        p22/
          Analysis/
            model/
              model_config.py
            RQ3/
              rq3_scenario_simulation_calibrated_fixed.py
          data/
            final/
              final_aki_dataset_engineered_FIXED.csv
    """
    here = Path(__file__).resolve().parent
    candidates = [here] + list(here.parents)

    for candidate in candidates:
        if (candidate / "model" / "model_config.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find Analysis/model/model_config.py. Place this script inside "
        "Analysis/RQ3 or another folder under Analysis."
    )


def load_model_config(model_config_path: Path) -> ModuleType:
    if not model_config_path.exists():
        raise FileNotFoundError(f"model_config.py not found: {model_config_path}")

    spec = importlib.util.spec_from_file_location("model_config", model_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {model_config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["model_config"] = module
    spec.loader.exec_module(module)
    return module


ANALYSIS_DIR = find_analysis_dir()
MODEL_DIR = ANALYSIS_DIR / "model"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.py"
CONFIG = load_model_config(MODEL_CONFIG_PATH)

DATASET_PATH: str = CONFIG.DATASET_PATH
TARGET: str = CONFIG.TARGET
IMPROVED_FEATURES_FINAL: list[str] = list(CONFIG.IMPROVED_FEATURES_FINAL)
KNOWN_UNUSABLE_COLS: list[str] = list(CONFIG.KNOWN_UNUSABLE_COLS)
EXCLUDE_COLS: list[str] = list(CONFIG.EXCLUDE_COLS)
OUTLIER_PRONE_FEATURES: list[str] = list(CONFIG.OUTLIER_PRONE_FEATURES)
XGBOOST_PARAMS: dict[str, Any] = dict(CONFIG.XGBOOST_PARAMS)
MISSING_VALUE_THRESHOLD: float = float(CONFIG.MISSING_VALUE_THRESHOLD)
TEST_SIZE: float = float(CONFIG.TEST_SIZE)
RANDOM_STATE: int = int(CONFIG.RANDOM_STATE)
engineer_features = CONFIG.engineer_features

RQ3_RESULTS_DIR = ANALYSIS_DIR / "results" / "rq3_calibrated_scenario_simulation"
RQ3_MODELS_DIR = MODEL_DIR / "models"
PLOTS_DIR = RQ3_RESULTS_DIR / "plots"

RQ3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RQ3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONSTANTS
# =============================================================================

GROUP_COL = "subject_id"
VAL_SIZE = 0.20
N_BINS = 10
LOW_RISK_MAX = 0.10
HIGH_RISK_MIN = 0.30
REQUIRED_FEATURE = "creatinine_slope_12h"
MIN_DENOMINATOR = 1e-8

MODEL_OUT = RQ3_MODELS_DIR / "rq3_calibrated_xgboost_model.pkl"
CALIBRATOR_OUT = RQ3_MODELS_DIR / "rq3_isotonic_calibrator.pkl"
FEATURES_OUT = RQ3_MODELS_DIR / "rq3_calibrated_xgboost_features.txt"


# =============================================================================
# HELPERS
# =============================================================================


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for {context}: {missing}")


# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_and_engineer_data() -> pd.DataFrame:
    dataset_path = Path(DATASET_PATH).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    require_columns(df, [TARGET, GROUP_COL], "modeling")

    df = engineer_features(df)

    if REQUIRED_FEATURE not in df.columns:
        raise KeyError(
            f"Required feature '{REQUIRED_FEATURE}' was not found after engineer_features(). "
            "The dataset must include creatinine_slope_12h."
        )

    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(int)
    df = df[df[GROUP_COL].notna()].copy()

    print_header("DATA LOADED")
    print(f"Dataset: {dataset_path}")
    print(f"Rows x columns: {df.shape}")
    print(f"Target: {TARGET}")
    print(f"Target prevalence: {df[TARGET].mean():.2%}")
    print(f"Unique patients: {df[GROUP_COL].nunique()}")
    print(f"Uses required early renal feature: {REQUIRED_FEATURE in df.columns}")

    return df


def convert_basic_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "sex" in df.columns and df["sex"].dtype == "object":
        df["sex"] = df["sex"].map({"M": 1, "F": 0}).astype("float")

    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    return df


def build_initial_feature_list(df: pd.DataFrame) -> list[str]:
    features = [c for c in IMPROVED_FEATURES_FINAL if c in df.columns]

    if REQUIRED_FEATURE in df.columns and REQUIRED_FEATURE not in features:
        features.append(REQUIRED_FEATURE)

    blocked = set(EXCLUDE_COLS) | set(KNOWN_UNUSABLE_COLS) | {TARGET, GROUP_COL}
    features = [c for c in features if c not in blocked]
    features = list(dict.fromkeys(features))

    if REQUIRED_FEATURE not in features:
        raise RuntimeError(
            f"{REQUIRED_FEATURE} was removed from the feature list. "
            "Check EXCLUDE_COLS and KNOWN_UNUSABLE_COLS in model_config.py."
        )

    return features


def group_train_test_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = df[TARGET].astype(int)
    groups = df[GROUP_COL]

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(outer.split(df, y, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    inner_train_idx, val_idx = next(
        inner.split(df_train, df_train[TARGET].astype(int), groups=df_train[GROUP_COL])
    )

    df_train_inner = df_train.iloc[inner_train_idx].copy()
    df_val = df_train.iloc[val_idx].copy()

    train_patients = set(df_train_inner[GROUP_COL].unique())
    val_patients = set(df_val[GROUP_COL].unique())
    test_patients = set(df_test[GROUP_COL].unique())

    if train_patients & val_patients:
        raise RuntimeError("Leakage detected: train/validation patient overlap.")
    if train_patients & test_patients:
        raise RuntimeError("Leakage detected: train/test patient overlap.")
    if val_patients & test_patients:
        raise RuntimeError("Leakage detected: validation/test patient overlap.")

    print_header("GROUPED SPLITS")
    print(f"Train rows:      {len(df_train_inner)}")
    print(f"Validation rows: {len(df_val)}")
    print(f"Test rows:       {len(df_test)}")
    print(f"Train prevalence: {df_train_inner[TARGET].mean():.2%}")
    print(f"Val prevalence:   {df_val[TARGET].mean():.2%}")
    print(f"Test prevalence:  {df_test[TARGET].mean():.2%}")

    return df_train_inner, df_val, df_test


def fit_feature_space(df_train_inner: pd.DataFrame, initial_features: list[str]) -> list[str]:
    numeric_features = (
        df_train_inner[initial_features]
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    kept: list[str] = []
    dropped_rows: list[dict[str, Any]] = []

    for col in numeric_features:
        missing_rate = float(df_train_inner[col].isna().mean())
        nunique = int(df_train_inner[col].nunique(dropna=True))

        if col == REQUIRED_FEATURE:
            if nunique <= 1:
                raise RuntimeError(f"{REQUIRED_FEATURE} exists but is constant in training data.")
            kept.append(col)
            continue

        if missing_rate <= MISSING_VALUE_THRESHOLD and nunique > 1:
            kept.append(col)
        else:
            dropped_rows.append({"feature": col, "missing_rate": missing_rate, "nunique": nunique})

    kept = list(dict.fromkeys(kept))

    if REQUIRED_FEATURE not in kept:
        raise RuntimeError(
            f"{REQUIRED_FEATURE} was not kept in final features. "
            "Check missingness, data type, or constant-value filtering."
        )

    pd.DataFrame(dropped_rows).to_csv(RQ3_RESULTS_DIR / "rq3_dropped_candidate_features.csv", index=False)

    print_header("FEATURE SPACE")
    print(f"Initial candidate features: {len(initial_features)}")
    print(f"Final numeric features:     {len(kept)}")
    print(f"Uses {REQUIRED_FEATURE}:    {REQUIRED_FEATURE in kept}")

    return kept


def fit_clip_bounds(df_train_inner: pd.DataFrame, features: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}

    for col in OUTLIER_PRONE_FEATURES:
        if col not in features:
            continue
        s = df_train_inner[col].dropna()
        if len(s) < 10:
            continue
        lo = float(s.quantile(0.01))
        hi = float(s.quantile(0.99))
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            bounds[col] = (lo, hi)

    return bounds


def apply_clip_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    df = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def make_X_y(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[features].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


# =============================================================================
# MODEL TRAINING AND CALIBRATION
# =============================================================================


def build_xgboost_model(y_train: pd.Series) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    params = dict(XGBOOST_PARAMS)
    params.setdefault("objective", "binary:logistic")
    params.setdefault("eval_metric", "auc")
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("tree_method", "hist")
    params.setdefault("n_jobs", -1)
    params.setdefault("verbosity", 0)
    params["scale_pos_weight"] = scale_pos_weight

    print_header("XGBOOST PARAMETERS")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    for key, value in params.items():
        print(f"{key}: {value}")

    return XGBClassifier(**params)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
    model = build_xgboost_model(y_train)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def print_metrics(model: XGBClassifier, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }

    print_header(f"METRICS: {split_name}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def fit_risk_calibrator(y_true: pd.Series, raw_prob: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_prob, y_true.astype(int))
    return calibrator


def apply_risk_calibration(calibrator: IsotonicRegression, raw_prob: np.ndarray) -> np.ndarray:
    return calibrator.predict(raw_prob)


def predict_calibrated(model: XGBClassifier, calibrator: IsotonicRegression, X: pd.DataFrame) -> np.ndarray:
    raw_prob = model.predict_proba(X)[:, 1]
    return apply_risk_calibration(calibrator, raw_prob)


def derive_clinical_risk_thresholds(
        y_true: pd.Series,
        calibrated_prob: np.ndarray,
        n_bins: int = N_BINS,
        low_risk_max: float = LOW_RISK_MAX,
        high_risk_min: float = HIGH_RISK_MIN,
) -> tuple[float, float, pd.DataFrame]:
    bins_df = pd.DataFrame({"prob": calibrated_prob, "y": y_true.astype(int).values})
    bins_df = bins_df.sort_values("prob").reset_index(drop=True)
    bins_df["bin"] = pd.qcut(bins_df["prob"], q=n_bins, duplicates="drop")

    bin_stats = (
        bins_df.groupby("bin", observed=False)
        .agg(
            prob_min=("prob", "min"),
            prob_max=("prob", "max"),
            mean_pred=("prob", "mean"),
            observed_aki_rate=("y", "mean"),
            n=("y", "size"),
        )
        .reset_index(drop=True)
    )

    low_to_moderate = None
    moderate_to_high = None

    for _, row in bin_stats.iterrows():
        if low_to_moderate is None and row["observed_aki_rate"] >= low_risk_max:
            low_to_moderate = float(row["prob_min"])
        if moderate_to_high is None and row["observed_aki_rate"] > high_risk_min:
            moderate_to_high = float(row["prob_min"])

    if low_to_moderate is None:
        low_to_moderate = float(np.quantile(calibrated_prob, 0.33))
    if moderate_to_high is None:
        moderate_to_high = float(np.quantile(calibrated_prob, 0.66))
    if moderate_to_high <= low_to_moderate:
        moderate_to_high = max(float(np.quantile(calibrated_prob, 0.66)), low_to_moderate + 1e-6)

    return low_to_moderate, moderate_to_high, bin_stats


def assign_risk_category(prob: float, low_to_moderate: float, moderate_to_high: float) -> str:
    if prob < low_to_moderate:
        return "Low"
    if prob < moderate_to_high:
        return "Moderate"
    return "High"


# =============================================================================
# SCENARIOS
# =============================================================================


def multiply_if_present(df: pd.DataFrame, col: str, factor: float, changed: list[str]) -> None:
    if col in df.columns:
        df[col] = df[col] * factor
        changed.append(f"{col} *= {factor}")


def set_if_present(df: pd.DataFrame, col: str, value: float | int, changed: list[str]) -> None:
    if col in df.columns:
        df[col] = value
        changed.append(f"{col} = {value}")


def decrement_if_present(df: pd.DataFrame, col: str, amount: float, changed: list[str]) -> None:
    if col in df.columns:
        df[col] = (df[col] - amount).clip(lower=0)
        changed.append(f"{col} -= {amount}, clipped at 0")


def define_scenarios(X_high_risk: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    scenarios: dict[str, pd.DataFrame] = {}
    metadata_rows: list[dict[str, str]] = []

    baseline = X_high_risk.copy()
    scenarios["A_baseline_observed_exposure"] = baseline
    metadata_rows.append(
        {
            "scenario": "A_baseline_observed_exposure",
            "description": "Observed high-risk feature values; no counterfactual modification.",
            "modified_columns": "",
        }
    )

    # B. Shorter PTZ exposure/duration.
    data = X_high_risk.copy()
    changed: list[str] = []
    for col in ["ptz_duration_hours", "piperacillin_tazobactam_duration_hours"]:
        multiply_if_present(data, col, 0.50, changed)
    if not changed:
        multiply_if_present(data, "piperacillin_tazobactam_use", 0.50, changed)
        multiply_if_present(data, "exposure", 0.50, changed)
        multiply_if_present(data, "high_risk_drug_combo", 0.50, changed)
        decrement_if_present(data, "num_nephrotoxic_drugs", 0.50, changed)
    scenarios["B_50pct_shorter_ptz_exposure"] = data
    metadata_rows.append(
        {
            "scenario": "B_50pct_shorter_ptz_exposure",
            "description": "PTZ duration reduced by 50%; if duration fields are unavailable, PTZ/combo exposure proxies are scaled by 50%.",
            "modified_columns": "; ".join(changed),
        }
    )

    # C. Reduced vancomycin exposure.
    data = X_high_risk.copy()
    changed = []
    for col in [
        "vancomycin_total_dose",
        "vancomycin_daily_dose",
        "vancomycin_dose_mg",
        "vancomycin_dose_mg_kg",
        "vancomycin_auc",
        "vancomycin_trough",
        "vanco_total_dose",
        "vanco_daily_dose",
        "vanco_dose_mg",
        "vanco_auc",
        "vanco_trough",
        "vanco_duration_hours",
        "vancomycin_duration_hours",
    ]:
        multiply_if_present(data, col, 0.50, changed)
    if not changed:
        multiply_if_present(data, "vancomycin_use", 0.50, changed)
        multiply_if_present(data, "vanco_plus_aminoglycoside", 0.50, changed)
        multiply_if_present(data, "vanco_with_diuretic", 0.50, changed)
        multiply_if_present(data, "high_risk_drug_combo", 0.50, changed)
    scenarios["C_50pct_reduced_vancomycin_exposure"] = data
    metadata_rows.append(
        {
            "scenario": "C_50pct_reduced_vancomycin_exposure",
            "description": "Vancomycin dose/AUC/trough/duration reduced by 50%; if unavailable, vancomycin-related exposure proxies are scaled by 50%.",
            "modified_columns": "; ".join(changed),
        }
    )

    # D. PTZ stopped / non-PTZ alternative proxy.
    data = X_high_risk.copy()
    changed = []
    set_if_present(data, "piperacillin_tazobactam_use", 0, changed)
    set_if_present(data, "exposure", 0, changed)
    set_if_present(data, "high_risk_drug_combo", 0, changed)
    decrement_if_present(data, "num_nephrotoxic_drugs", 1, changed)
    for col in ["ptz_duration_hours", "piperacillin_tazobactam_duration_hours"]:
        set_if_present(data, col, 0, changed)
    scenarios["D_ptz_discontinued_or_non_ptz_alternative"] = data
    metadata_rows.append(
        {
            "scenario": "D_ptz_discontinued_or_non_ptz_alternative",
            "description": "PTZ removed from the exposure profile while holding non-antibiotic clinical features fixed.",
            "modified_columns": "; ".join(changed),
        }
    )

    # E. Combined de-escalation.
    data = X_high_risk.copy()
    changed = []
    for col in ["ptz_duration_hours", "piperacillin_tazobactam_duration_hours"]:
        multiply_if_present(data, col, 0.50, changed)
    for col in [
        "vancomycin_total_dose",
        "vancomycin_daily_dose",
        "vancomycin_dose_mg",
        "vancomycin_dose_mg_kg",
        "vancomycin_auc",
        "vancomycin_trough",
        "vanco_total_dose",
        "vanco_daily_dose",
        "vanco_dose_mg",
        "vanco_auc",
        "vanco_trough",
        "vanco_duration_hours",
        "vancomycin_duration_hours",
    ]:
        multiply_if_present(data, col, 0.50, changed)
    multiply_if_present(data, "piperacillin_tazobactam_use", 0.50, changed)
    multiply_if_present(data, "vancomycin_use", 0.50, changed)
    multiply_if_present(data, "exposure", 0.50, changed)
    multiply_if_present(data, "vanco_plus_aminoglycoside", 0.50, changed)
    multiply_if_present(data, "vanco_with_diuretic", 0.50, changed)
    multiply_if_present(data, "high_risk_drug_combo", 0.50, changed)
    decrement_if_present(data, "num_nephrotoxic_drugs", 1, changed)
    scenarios["E_combined_ptz_and_vanco_deescalation"] = data
    metadata_rows.append(
        {
            "scenario": "E_combined_ptz_and_vanco_deescalation",
            "description": "Combined shorter PTZ and reduced vancomycin exposure using available duration/dose fields or fallback proxies.",
            "modified_columns": "; ".join(changed),
        }
    )

    metadata = pd.DataFrame(metadata_rows)
    return scenarios, metadata


# =============================================================================
# RESULTS
# =============================================================================


def build_summary(results: dict[str, np.ndarray]) -> pd.DataFrame:
    baseline = results["A_baseline_observed_exposure"]
    rows: list[dict[str, Any]] = []

    for scenario, risk in results.items():
        absolute_change = risk - baseline
        absolute_reduction = baseline - risk
        relative_reduction = (absolute_reduction / np.maximum(baseline, MIN_DENOMINATOR)) * 100

        rows.append(
            {
                "scenario": scenario,
                "n_high_risk_patients": len(risk),
                "mean_calibrated_risk": float(np.mean(risk)),
                "median_calibrated_risk": float(np.median(risk)),
                "std_calibrated_risk": float(np.std(risk)),
                "mean_absolute_risk_change": float(np.mean(absolute_change)),
                "mean_absolute_risk_reduction": float(np.mean(absolute_reduction)),
                "mean_relative_risk_reduction_pct": float(np.mean(relative_reduction)),
                "patients_with_any_reduction_n": int((absolute_reduction > 0).sum()),
                "patients_with_abs_reduction_ge_0_05_n": int((absolute_reduction >= 0.05).sum()),
                "patients_with_abs_reduction_ge_0_10_n": int((absolute_reduction >= 0.10).sum()),
                "patients_with_relative_reduction_ge_10pct_n": int((relative_reduction >= 10).sum()),
            }
        )

    return pd.DataFrame(rows)


def build_patient_level_results(high_risk_df: pd.DataFrame, results: dict[str, np.ndarray]) -> pd.DataFrame:
    base_cols = [
        c
        for c in [GROUP_COL, TARGET, "baseline_raw_risk", "baseline_calibrated_risk", "baseline_risk_category"]
        if c in high_risk_df.columns
    ]
    out = high_risk_df[base_cols].copy()
    baseline = results["A_baseline_observed_exposure"]

    for scenario, risk in results.items():
        out[f"risk__{scenario}"] = risk
        out[f"absolute_change__{scenario}"] = risk - baseline
        out[f"absolute_reduction__{scenario}"] = baseline - risk
        out[f"relative_reduction_pct__{scenario}"] = (
                                                             (baseline - risk) / np.maximum(baseline, MIN_DENOMINATOR)
                                                     ) * 100

    return out


def save_plots(results: dict[str, np.ndarray], summary_df: pd.DataFrame) -> None:
    scenario_names = list(results.keys())
    short_names = [
                      "Baseline",
                      "PTZ 50%",
                      "Vanco 50%",
                      "PTZ off",
                      "Combined",
                  ][: len(scenario_names)]

    plt.figure(figsize=(12, 7))
    plt.violinplot([results[name] for name in scenario_names], showmeans=True)
    plt.xticks(range(1, len(short_names) + 1), short_names, rotation=20, ha="right")
    plt.ylabel("Calibrated predicted AKI risk")
    plt.title("RQ3: calibrated AKI risk distribution across scenarios")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rq3_calibrated_risk_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()

    plot_df = summary_df[summary_df["scenario"] != "A_baseline_observed_exposure"].copy()
    plt.figure(figsize=(12, 7))
    x = np.arange(len(plot_df))
    values = plot_df["mean_relative_risk_reduction_pct"].values
    bars = plt.bar(x, values)
    plt.xticks(x, [s.replace("_", " ") for s in plot_df["scenario"]], rotation=20, ha="right")
    plt.ylabel("Mean relative risk reduction (%)")
    plt.title("RQ3: mean relative reduction in calibrated AKI risk")
    plt.axhline(y=0, linewidth=0.8)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rq3_mean_relative_risk_reduction.png", dpi=300, bbox_inches="tight")
    plt.close()

    n = len(next(iter(results.values())))
    if n == 0:
        return

    sample_size = min(100, n)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(np.arange(n), size=sample_size, replace=False)

    plt.figure(figsize=(12, 7))
    x = np.arange(len(scenario_names))

    for idx in sample_idx:
        y_values = [results[name][idx] for name in scenario_names]
        plt.plot(x, y_values, marker="o", alpha=0.25, linewidth=0.8)

    mean_values = [np.mean(results[name]) for name in scenario_names]
    plt.plot(x, mean_values, marker="o", linewidth=3, label="Mean trajectory")
    plt.xticks(x, short_names, rotation=20, ha="right")
    plt.ylabel("Calibrated predicted AKI risk")
    plt.title(f"RQ3: individual risk trajectories among high-risk patients (n={sample_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rq3_individual_risk_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print_header("RQ3 CALIBRATED SCENARIO-BASED AKI RISK SIMULATION")
    print(f"Analysis directory: {ANALYSIS_DIR}")
    print(f"Model config:       {MODEL_CONFIG_PATH}")
    print(f"Results directory:  {RQ3_RESULTS_DIR}")

    df = load_and_engineer_data()
    df = convert_basic_types(df)

    initial_features = build_initial_feature_list(df)
    df_train_inner, df_val, df_test = group_train_test_val_split(df)
    final_features = fit_feature_space(df_train_inner, initial_features)
    clip_bounds = fit_clip_bounds(df_train_inner, final_features)

    df_train_inner = apply_clip_bounds(df_train_inner, clip_bounds)
    df_val = apply_clip_bounds(df_val, clip_bounds)
    df_test = apply_clip_bounds(df_test, clip_bounds)
    df_all_for_sim = apply_clip_bounds(df.copy(), clip_bounds)

    X_train, y_train = make_X_y(df_train_inner, final_features)
    X_val, y_val = make_X_y(df_val, final_features)
    X_test, y_test = make_X_y(df_test, final_features)

    model = train_model(X_train, y_train, X_val, y_val)

    train_metrics = print_metrics(model, X_train, y_train, "train_inner")
    val_metrics = print_metrics(model, X_val, y_val, "validation")
    test_metrics = print_metrics(model, X_test, y_test, "test")

    val_raw_prob = model.predict_proba(X_val)[:, 1]
    calibrator = fit_risk_calibrator(y_true=y_val, raw_prob=val_raw_prob)
    val_calibrated_prob = apply_risk_calibration(calibrator, val_raw_prob)

    low_to_moderate, moderate_to_high, calibration_bins = derive_clinical_risk_thresholds(
        y_true=y_val,
        calibrated_prob=val_calibrated_prob,
        n_bins=N_BINS,
        low_risk_max=LOW_RISK_MAX,
        high_risk_min=HIGH_RISK_MIN,
    )

    print_header("CALIBRATED VALIDATION-DERIVED THRESHOLDS")
    print("Low risk:      observed AKI < 10%")
    print("Moderate risk: observed AKI 10-30%")
    print("High risk:     observed AKI > 30%")
    print(f"Low -> Moderate threshold:  {low_to_moderate:.4f}")
    print(f"Moderate -> High threshold: {moderate_to_high:.4f}")

    calibration_bins_path = RQ3_RESULTS_DIR / "rq3_validation_calibration_bins.csv"
    calibration_bins.to_csv(calibration_bins_path, index=False)

    X_all = df_all_for_sim[final_features].copy()
    baseline_raw_prob = model.predict_proba(X_all)[:, 1]
    baseline_calibrated_prob = apply_risk_calibration(calibrator, baseline_raw_prob)

    sim_df = df_all_for_sim.copy()
    sim_df["baseline_raw_risk"] = baseline_raw_prob
    sim_df["baseline_calibrated_risk"] = baseline_calibrated_prob
    sim_df["baseline_risk_category"] = [
        assign_risk_category(p, low_to_moderate, moderate_to_high) for p in baseline_calibrated_prob
    ]

    category_counts = (
        sim_df["baseline_risk_category"]
        .value_counts()
        .rename_axis("risk_category")
        .reset_index(name="n")
    )
    category_counts_path = RQ3_RESULTS_DIR / "rq3_baseline_risk_category_counts.csv"
    category_counts.to_csv(category_counts_path, index=False)

    high_risk_df = sim_df[sim_df["baseline_risk_category"] == "High"].copy()

    print_header("HIGH-RISK COHORT")
    print(category_counts.to_string(index=False))
    print(f"High-risk definition: calibrated baseline risk >= {moderate_to_high:.4f}")
    print(f"High-risk rows retained for RQ3: {len(high_risk_df)}")

    if len(high_risk_df) == 0:
        raise RuntimeError(
            "No high-risk patients found using the calibrated threshold. "
            "Check calibration bins or HIGH_RISK_MIN."
        )

    X_high_risk = high_risk_df[final_features].copy()
    scenarios, scenario_metadata = define_scenarios(X_high_risk)

    scenario_metadata_path = RQ3_RESULTS_DIR / "rq3_scenario_definitions.csv"
    scenario_metadata.to_csv(scenario_metadata_path, index=False)

    print_header("SCENARIO PREDICTIONS AMONG CALIBRATED HIGH-RISK PATIENTS")
    results: dict[str, np.ndarray] = {}

    for scenario_name, X_scenario in scenarios.items():
        X_scenario = X_scenario.reindex(columns=final_features)
        scenario_risk = predict_calibrated(model, calibrator, X_scenario)
        results[scenario_name] = scenario_risk
        print(
            f"{scenario_name}: "
            f"mean={np.mean(scenario_risk):.4f}, "
            f"median={np.median(scenario_risk):.4f}, "
            f"min={np.min(scenario_risk):.4f}, "
            f"max={np.max(scenario_risk):.4f}"
        )

    summary_df = build_summary(results)
    patient_level_df = build_patient_level_results(high_risk_df, results)

    summary_path = RQ3_RESULTS_DIR / "rq3_calibrated_scenario_summary.csv"
    patient_path = RQ3_RESULTS_DIR / "rq3_calibrated_patient_level_results.csv"
    metrics_path = RQ3_RESULTS_DIR / "rq3_model_metrics.csv"
    features_path = RQ3_RESULTS_DIR / "rq3_final_features.txt"
    methods_path = RQ3_RESULTS_DIR / "rq3_methods_text.txt"

    summary_df.to_csv(summary_path, index=False)
    patient_level_df.to_csv(patient_path, index=False)

    metrics_df = pd.DataFrame(
        [
            {"split": "train_inner", **train_metrics},
            {"split": "validation", **val_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    metrics_df.to_csv(metrics_path, index=False)

    features_path.write_text("\n".join(final_features) + "\n")
    FEATURES_OUT.write_text("\n".join(final_features) + "\n")
    joblib.dump(model, MODEL_OUT)
    joblib.dump(calibrator, CALIBRATOR_OUT)

    methods_text = f"""
RQ3 method summary

High-risk patients were identified using validation-derived calibrated risk strata.
An XGBoost model was trained with grouped patient-level splitting to avoid subject-level
leakage. Feature engineering was performed using model_config.engineer_features(), and
the final feature set was based on model_config.IMPROVED_FEATURES_FINAL. The model
included the early renal predictor {REQUIRED_FEATURE}.

Raw XGBoost predicted probabilities were calibrated using isotonic regression on the
held-out validation split. The high-risk threshold was defined as the first validation
risk bin where observed AKI risk exceeded {HIGH_RISK_MIN:.0%}. In this run, the
calibrated Moderate-to-High threshold was {moderate_to_high:.4f}.

Patients with baseline calibrated predicted AKI risk greater than or equal to this
threshold were classified as high-risk and included in the scenario simulation.

For each high-risk patient, baseline clinical characteristics were held fixed while
antibiotic exposure-related variables were modified to represent alternative exposure
scenarios: shorter piperacillin-tazobactam exposure, reduced vancomycin exposure,
PTZ discontinuation or non-PTZ alternative, and combined de-escalation.

The trained model then re-estimated calibrated predicted AKI risk under each scenario.
Risk change was summarized as absolute and relative change compared with the observed
baseline exposure scenario.

These simulations are model-based counterfactual predictions, not causal treatment
effects and not clinical dosing recommendations.
""".strip()
    methods_path.write_text(methods_text)

    save_plots(results, summary_df)

    print_header("DONE")
    print(f"Saved model:               {MODEL_OUT}")
    print(f"Saved calibrator:          {CALIBRATOR_OUT}")
    print(f"Saved feature list:        {FEATURES_OUT}")
    print(f"Scenario definitions:      {scenario_metadata_path}")
    print(f"Scenario summary:          {summary_path}")
    print(f"Patient-level results:     {patient_path}")
    print(f"Calibration bins:          {calibration_bins_path}")
    print(f"Risk category counts:      {category_counts_path}")
    print(f"Model metrics:             {metrics_path}")
    print(f"Methods text:              {methods_path}")
    print(f"Plots directory:           {PLOTS_DIR}")

    print("\nScenario summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
