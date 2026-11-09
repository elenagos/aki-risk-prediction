# Main writer: Bartlomiej Kaczor
# Reviewer: Elena Gostiukhina
# Contributors: Luca Dragutanu

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

model = None
imputer = None
feature_cols = None
shap_explainer = None
threshold = 0.35


def train_model(df):
    global model, imputer, feature_cols, shap_explainer
    target = "aki_day7"
    exclude_cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "intime",
        "outtime",
        "admittime",
        "dischtime",
        "index_time",
        "aki_day7",
        "time_to_aki_hours",
        "vanco_starttime",
        "ptz_starttime",
        "creatinine_delta_48h",
        "creatinine_ratio_7d",
    ]
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols
        and df[col].dtype in ["float64", "int64", "bool"]
        and df[col].isna().mean() <= 0.5
    ]
    X_raw = df[feature_cols].copy()
    y = df[target].astype(int).copy()
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X_raw),
        columns=feature_cols,
        index=X_raw.index,
    )
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    shap_explainer = None


def predict_patient(row):
    global model, imputer, feature_cols
    X = prepare_patient_features(row)
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= threshold)
    return prob, pred


def get_model_objects():
    return model, imputer, feature_cols


def prepare_patient_features(row):
    global imputer, feature_cols

    if imputer is None or feature_cols is None:
        raise RuntimeError("Model has not been trained yet. Call train_model(df) first.")

    X_raw = row[feature_cols].to_frame().T
    X_imputed = imputer.transform(X_raw)
    return pd.DataFrame(X_imputed, columns=feature_cols, index=X_raw.index)


def get_shap_explainer():
    global model, shap_explainer

    if model is None:
        raise RuntimeError("Model has not been trained yet. Call train_model(df) first.")

    if shap_explainer is None:
        import shap
        shap_explainer = shap.TreeExplainer(model)

    return shap_explainer


def explain_patient(row, top_n=10):
    X = prepare_patient_features(row)
    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        patient_shap_values = shap_values[1][0]
    elif getattr(shap_values, "ndim", 0) == 3:
        patient_shap_values = shap_values[0, :, 1]
    else:
        patient_shap_values = shap_values[0]

    explanation = pd.DataFrame(
        {
            "feature": feature_cols,
            "value": X.iloc[0].values,
            "shap_impact": patient_shap_values,
        }
    )
    explanation["abs_impact"] = explanation["shap_impact"].abs()

    return (
        explanation
        .sort_values("abs_impact", ascending=False)
        .head(top_n)
        .drop(columns="abs_impact")
        .reset_index(drop=True)
    )
