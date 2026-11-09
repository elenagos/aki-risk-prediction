# Main writer: Bartlomiej Kaczor
# Reviewer:  Elena Gostiukhina
# Contributors: Luca Dragutanu

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_CONFIG_DIR = BASE_DIR / "Analysis" / "model"
STACKING_MODEL_PATH = MODEL_CONFIG_DIR / "models" / "stacking_model.pkl"

sys.path.insert(0, str(MODEL_CONFIG_DIR))

from model_config import engineer_features


stacking_package = None
base_models = None
meta_model = None
feature_cols = None
base_model_order = None
threshold = None
background_X = None
shap_explainer = None


def add_feature_engineering_inputs(df):
    """
    Ensurse engineer_features() can run on partial uploaded patient rows.
    """
    df = df.copy()

    if "vanco_ptz_combo" not in df.columns:
        df["vanco_ptz_combo"] = df["exposure"] if "exposure" in df.columns else 0
    df["vanco_ptz_combo"] = pd.to_numeric(df["vanco_ptz_combo"], errors="coerce").fillna(0)

    if "num_nephrotoxic_drugs" not in df.columns:
        drug_cols = [
            "aminoglycoside_use",
            "nsaid_use",
            "vanco_ptz_combo",
        ]
        present_drug_cols = [col for col in drug_cols if col in df.columns]
        if present_drug_cols:
            df["num_nephrotoxic_drugs"] = (
                df[present_drug_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum(axis=1)
            )
        else:
            df["num_nephrotoxic_drugs"] = 0

    if "sofa_score" not in df.columns:
        sofa_score = pd.Series(np.nan, index=df.index, dtype="float64")
        score_parts = []

        spo2_source = "spo2_min" if "spo2_min" in df.columns else "spo2_mean"
        if spo2_source in df.columns:
            score_parts.append(pd.to_numeric(df[spo2_source], errors="coerce").lt(90).astype(int))

        if "map_min" in df.columns:
            score_parts.append(pd.to_numeric(df["map_min"], errors="coerce").lt(70).astype(int))

        if "baseline_creatinine" in df.columns:
            score_parts.append(
                pd.to_numeric(df["baseline_creatinine"], errors="coerce").gt(1.2).astype(int)
            )

        platelets_source = "platelets_min" if "platelets_min" in df.columns else "platelets_mean"
        if platelets_source in df.columns:
            score_parts.append(
                pd.to_numeric(df[platelets_source], errors="coerce").lt(100).astype(int)
            )

        if score_parts:
            sofa_score = sum(score_parts)

        df["sofa_score"] = sofa_score

    paired_defaults = {
        "bun_mean": "bun_max",
        "bun_max": "bun_mean",
        "lactate_mean": "lactate_max",
        "lactate_max": "lactate_mean",
    }
    for col, fallback_col in paired_defaults.items():
        if col not in df.columns:
            df[col] = df[fallback_col] if fallback_col in df.columns else np.nan

    if "baseline_creatinine" not in df.columns:
        df["baseline_creatinine"] = np.nan

    return df


def train_model(df=None):
    """
    The stacking model is trained by Analysis/RQ1/Stacking_Model.py and saved
    as a package, so the GUI only needs to load it.
    """
    load_model()

    if df is not None:
        set_background_data(df)


def load_model(model_path=STACKING_MODEL_PATH):
    global stacking_package, base_models, meta_model, feature_cols
    global base_model_order, threshold

    if stacking_package is not None:
        return

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Stacking model not found at {model_path}. "
            "Run Analysis/RQ1/Stacking_Model.py first to create stacking_model.pkl."
        )

    stacking_package = joblib.load(model_path)
    base_models = stacking_package["base_models"]
    meta_model = stacking_package["meta_model"]
    feature_cols = stacking_package["feature_cols"]
    base_model_order = stacking_package["base_model_order"]
    threshold = stacking_package.get("threshold", 0.45)


def prepare_feature_frame(data):
    if stacking_package is None:
        load_model()

    if isinstance(data, pd.Series):
        df = data.to_frame().T.copy()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data, columns=feature_cols).copy()

    df = add_feature_engineering_inputs(df)
    df = engineer_features(df)

    if "sex" in df.columns and not pd.api.types.is_numeric_dtype(df["sex"]):
        df["sex"] = (df["sex"].astype(str).str.upper() == "M").astype(int)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def prepare_patient_features(row):
    return prepare_feature_frame(row)


def set_background_data(df, sample_size=50):
    global background_X, shap_explainer

    if df is None or len(df) == 0:
        return

    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    background_X = prepare_feature_frame(sample)
    shap_explainer = None


def get_meta_features(row):
    X = prepare_patient_features(row)

    meta_values = []
    for model_name in base_model_order:
        model = base_models[model_name]
        prob = model.predict_proba(X)[0][1]
        meta_values.append(prob)

    return pd.DataFrame([meta_values], columns=base_model_order, index=X.index)


def predict_patient(row):
    if stacking_package is None:
        load_model()

    meta_X = get_meta_features(row)
    prob = meta_model.predict_proba(meta_X)[0][1]
    pred = int(prob >= threshold)
    return prob, pred


def predict_stacking_from_features(X):
    if stacking_package is None:
        load_model()

    X_df = pd.DataFrame(X, columns=feature_cols)
    meta_values = []

    for model_name in base_model_order:
        model = base_models[model_name]
        meta_values.append(model.predict_proba(X_df)[:, 1])

    meta_X = np.column_stack(meta_values)
    return meta_model.predict_proba(meta_X)[:, 1]


def get_model_objects():
    if stacking_package is None:
        load_model()

    return stacking_package, base_models, meta_model, feature_cols


def explain_patient(row, top_n=10):
    """
    Explains the full stacking prediction using original clinical features.
    """
    global shap_explainer

    if stacking_package is None:
        load_model()

    if background_X is None:
        raise RuntimeError(
            "Clinical SHAP background data is not available. "
            "Call train_model(df) with the app dataset before explaining patients."
        )

    import shap

    X = prepare_patient_features(row)

    if shap_explainer is None:
        shap_explainer = shap.KernelExplainer(predict_stacking_from_features, background_X)

    shap_values = shap_explainer.shap_values(X, nsamples=100)

    if isinstance(shap_values, list):
        patient_shap_values = shap_values[0]
    else:
        patient_shap_values = shap_values

    patient_shap_values = np.asarray(patient_shap_values).reshape(-1)

    explanation = pd.DataFrame(
        {
            "feature": feature_cols,
            "value": X.iloc[0].values,
            "shap_impact": patient_shap_values,
        }
    )
    explanation["abs_impact"] = explanation["shap_impact"].abs()

    return (
        explanation.sort_values("abs_impact", ascending=False)
        .head(top_n)
        .drop(columns="abs_impact")
        .reset_index(drop=True)
    )
