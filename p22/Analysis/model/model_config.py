# Main writer: Cahit Kok
# Reviewer:  Bartlomiej Kaczor
# Contributors: Luca Dragutanu, Elena Gostiukhina

import os
import numpy as np

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'final', 'final_aki_dataset_engineered_FIXED.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = 'aki_day7'

EXCLUDE_COLS = [
    'stay_id', 'subject_id', 'hadm_id',
    'intime', 'outtime', 'admittime', 'dischtime', 'index_time',
    'aki_day7', 'time_to_aki_hours',
    'vanco_starttime', 'ptz_starttime',
    'creatinine_delta_48h', 'creatinine_ratio_7d',
    'vanco_only', 'vanco_ptz_combo', 
    'creatinine_trend_24h', 'creatinine_slope_24h'

]

CONSERVATIVE = [
    'age', 'sex', 'bmi', 'weight_kg', 'baseline_creatinine', 'anchor_year',
    'ckd_flag', 'esrd_flag', 'diabetes_flag', 'hypertension_flag',
    'heart_failure_flag', 'ischemic_heart_disease_flag', 'mi_flag',
    'arrhythmia_flag', 'copd_flag', 'liver_disease_flag', 'malignancy_flag', 'pvd_flag',
    'charlson_index', 'sofa_score', 'saps_ii_score',
    'heart_rate_mean', 'systolic_bp_mean', 'diastolic_bp_mean',
    'map_mean', 'respiration_mean', 'temperature_mean', 'spo2_mean',
    'bun_mean', 'sodium_mean', 'potassium_mean',
    'hemoglobin_mean', 'wbc_mean', 'platelets_mean', 'lactate_mean',
    'exposure', 'los_hours',
    'vasopressor_use', 'aminoglycoside_use', 'loop_diuretic_use',
    'nsaid_use', 'acei_arb_use', 'amphotericin_use', 'sedative_use',
    'vancomycin_use', 'piperacillin_tazobactam_use',
    'vanco_plus_aminoglycoside', 'vanco_with_diuretic',
    'high_risk_drug_combo', 'num_nephrotoxic_drugs',
]

IMPROVED_RENAL_SIGNALS = [
    'baseline_creatinine',
    'bun_max',
    'bun_min',
    'lactate_change_24h',
    'bun_change_pct_24h',
]

EARLY_STRESS = [
    'lactate_max',
    'heart_rate_variability_24h',
    'lactate_elevation',
    'vasopressor_use',
    'temperature_range_24h',
]

VITAL_VARIABILITY = [
    'heart_rate_range_24h',
    'systolic_bp_range_24h',
    'diastolic_bp_range_24h',
    'map_range_24h',
    'spo2_range_24h',
]

EDA_SUPPORTED_FEATURES = [
    'sofa_score',
    'charlson_index',
    'age_comorbidity_interaction',
    'age_creatinine_interaction',
    'severity_drug_risk',
]

VITAL_EXTREMES = [
    'heart_rate_max', 'heart_rate_min',
    'respiration_max', 'respiration_min',
    'temperature_max', 'temperature_min',
    'spo2_min',
    'lactate_min',
]

LAB_EXTREMES = [
    'wbc_max', 'wbc_min',
    'platelets_max', 'platelets_min',
]

CLINICAL_FLAGS= [
    'fever_flag',
    'hypoxia_flag',
]

DRUG_FLAGS =[
    'vasopressor_use', 'aminoglycoside_use', 'loop_diuretic_use',
    'nsaid_use', 'acei_arb_use', 'amphotericin_use', 'sedative_use',
]

URINE_EXPERIMENT = [
    'urine_output_6h',
    'oliguria_flag_6h',
]

CREATININE_TREND_EXPERIMENT = [
    'creatinine_slope_12h',
]

MISSINGNESS_INDICATORS = {
    'lactate_missing': 'lactate_mean',
    'temperature_missing': 'temperature_mean',
}

ENGINEERED_INTERACTIONS = [
    'bun_creatinine_ratio',
    'lactate_bun_interaction',
    'sofa_drug_risk',
    'creatinine_drug_risk',
]


def engineer_features(df):
    df = df.copy()

    df['exposure'] = df['vanco_ptz_combo'].astype(int)

    df['bun_creatinine_ratio']    = df['bun_mean'] / df['baseline_creatinine'].replace(0, np.nan)
    df['lactate_bun_interaction'] = df['lactate_max'].fillna(df['lactate_mean']) * df['bun_max']
    df['sofa_drug_risk']          = df['sofa_score'] * df['num_nephrotoxic_drugs']
    df['creatinine_drug_risk']    = df['baseline_creatinine'] * df['num_nephrotoxic_drugs']

    for new_col, source_col in MISSINGNESS_INDICATORS.items():
        if source_col in df.columns:
            df[new_col] = df[source_col].isna().astype(int)

    return df

IMPROVED_FEATURES_FINAL = (
    CONSERVATIVE + IMPROVED_RENAL_SIGNALS + EARLY_STRESS + VITAL_VARIABILITY
    + EDA_SUPPORTED_FEATURES + VITAL_EXTREMES + LAB_EXTREMES + CLINICAL_FLAGS + DRUG_FLAGS
    + URINE_EXPERIMENT + CREATININE_TREND_EXPERIMENT
    + list(MISSINGNESS_INDICATORS.keys()) + ENGINEERED_INTERACTIONS
)
IMPROVED_FEATURES_FINAL = list(dict.fromkeys(IMPROVED_FEATURES_FINAL))

KNOWN_UNUSABLE_COLS = [
    'bmi', 'ckd_flag', 'diastolic_bp_mean', 'diastolic_bp_range_24h',
    'esrd_flag', 'hemoglobin_mean', 'map_mean', 'map_range_24h',
    'potassium_mean', 'sodium_mean', 'systolic_bp_mean', 'systolic_bp_range_24h',
    'weight_kg',
]

ARTIFACT_VITALS = [
    'heart_rate_mean', 'heart_rate_max', 'heart_rate_min',
    'heart_rate_range_24h', 'heart_rate_variability_24h',
    'respiration_mean', 'respiration_max', 'respiration_min',
    'temperature_mean', 'temperature_max', 'temperature_min',
    'temperature_range_24h',
    'spo2_mean', 'spo2_min',
    'spo2_range_24h',
]

CLINICAL_LABS_CLIP = [
    'baseline_creatinine',
    'bun_mean', 'bun_max', 'bun_min',
    'bun_change_pct_24h',
    'lactate_mean', 'lactate_max', 'lactate_min',
    'lactate_change_24h',
    'wbc_mean', 'wbc_max', 'wbc_min',
    'platelets_mean', 'platelets_max', 'platelets_min',
    'los_hours',
    'age_creatinine_interaction', 'severity_drug_risk',
    'bun_creatinine_ratio', 'lactate_bun_interaction',
    'creatinine_drug_risk',
    'creatinine_slope_12h',
    'urine_output_6h',
]

OUTLIER_PRONE_FEATURES = ARTIFACT_VITALS + CLINICAL_LABS_CLIP

MISSING_VALUE_THRESHOLD = 0.5
IMPUTATION_STRATEGY = 'median'
RANDOM_STATE = 42

TEST_SIZE = 0.3
STRATIFIED = True

CV_SPLITS = 5
CV_TYPE = 'stratified'  # Options: 'stratified', 'grouped', 'time_based'

METRICS = [
    'roc_auc',
    'pr_auc',
    'accuracy',
    'precision',
    'recall',
    'f1',
    'specificity',
    'sensitivity'
]

# Logistic Regression specific
LOGR_PARAMS = {
    'penalty': 'l1',
    'solver': 'liblinear',
    'max_iter': 5000,
    'random_state': RANDOM_STATE,
    'C': [0.001, 0.01, 0.1, 1, 3, 10]  # For grid search
}

# XGBoost specific
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': RANDOM_STATE,
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'gamma': 1.0,
    'early_stopping_rounds': 30,
    'tree_method': 'hist',
    'n_jobs': -1,
    'verbosity': 0,
}

    # Hyperparameter search space for RandomizedSearchCV   for XGBoost
XGB_TUNING_SPACE = {
    'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7, 10],
    'reg_alpha': [0, 0.5, 1, 2, 3],
    'reg_lambda': [0.5, 1, 2, 3, 5],
    'gamma': [0, 0.5, 1, 1.5, 2],
    'n_estimators': [200, 400, 600, 800, 1000],
}


# Random Forest specific
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# ============================================================================
# MODEL OUTPUT PATHS
# ============================================================================

LOGR_MODEL_PATH = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
RF_MODEL_PATH = os.path.join(MODELS_DIR, 'random_forest.pkl')

LOGR_RESULTS_PATH = os.path.join(RESULTS_DIR, 'logistic_regression_results.json')
XGBOOST_RESULTS_PATH = os.path.join(RESULTS_DIR, 'xgboost_results.json')
RF_RESULTS_PATH = os.path.join(RESULTS_DIR, 'random_forest_results.json')

# ============================================================================
# LOGGING & DISPLAY
# ============================================================================

VERBOSE = True
SAVE_PLOTS = True
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

if SAVE_PLOTS:
    os.makedirs(PLOTS_DIR, exist_ok=True)
