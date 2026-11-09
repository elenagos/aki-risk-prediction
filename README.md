# AKI Risk Prediction

A team research project exploring acute kidney injury (AKI) risk prediction in an ICU cohort using MIMIC-derived data. The project combines clinical data preparation, machine learning, survival analysis, and a Streamlit dashboard with SHAP explanations.

## Publication

**[Early Prediction of Acute Kidney Injury Risk after Nephrotoxic Antibiotic Exposure](https://openreview.net/forum?id=WiDXeSmAhr)**  
Gostiukhina, E., et al. — BNAIC/BeNeLearn 2026, poster presentation.

My work focused primarily on model development, clinical feature engineering, and model interpretation:

- Engineered clinical features and addressed class imbalance using class weighting and decision-threshold tuning.
- Contributed to missing-data imputation, percentile-based outlier clipping, and automated data-quality checks.
- Trained and optimized logistic regression and XGBoost models using patient-grouped cross-validation with GroupKFold.
- Identified the top 15 predictors using SHAP and contributed explanations used in the Streamlit dashboard for patient-level AKI risk assessment.
- Coauthored the research paper presented at BNAIC/BeNeLearn 2026.

## Research and methods

- **Risk prediction:** compare logistic regression, decision trees, random forests, XGBoost, and a stacking ensemble for the `aki_day7` outcome.
- **Time-to-event analysis:** investigate AKI onset with Kaplan–Meier curves, log-rank tests, and Cox proportional hazards modeling.
- **Scenario exploration:** examine model predictions under alternative feature scenarios. These experiments do not establish causal treatment effects.
- **Interpretability and evaluation:** inspect feature contributions, discrimination, calibration, and potential target leakage.

## Interactive dashboard

The Streamlit application supports cohort exploration, individual patient profiles, manual feature entry, prototype risk predictions, and SHAP explanations.

## Repository structure

```text
p22/
├── App.py                         # Streamlit application
├── model_loader_stacking.py       # Dashboard model and explanation utilities
├── model_loader_random_forest.py
├── Cleaning/                      # Cohort preparation and data cleaning
└── Analysis/
    ├── EDA/                       # Feature engineering
    ├── RQ1/                       # Classification experiments
    ├── RQ2/                       # Survival analysis
    ├── RQ3/                       # Calibrated scenario experiments
    └── model/                     # Shared configuration and model utilities
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run p22/App.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

The application requires an authorized local dataset under `p22/data/final/`, such as `final_aki_dataset_engineered_FIXED.csv`. Without a dataset, it stops and displays a missing-data message. Cleaning scripts also require locally configured paths to the source data; review `p22/Cleaning/config.py` before running them.

Dependencies are currently unpinned. This source export has been checked for Python syntax; the full data pipeline and application have not been re-run as part of preparing this public copy.

## Data and reproducibility

This public copy includes source code only. Used dataset is from Mimic-IV, doesn't allow open source publications. Patient-level datasets, derived row-level outputs, trained models, and the original generated analysis outputs are excluded. Access to the source data must be obtained separately through the applicable data provider process.

The archive used to prepare this repository does not contain Git history. Original source attribution comments have been preserved.

## Team and attribution

Originally developed as a Maastricht University group project: [original GitLab repository](https://gitlab.maastrichtuniversity.nl/group-05-p-22/aki).


## Scope and limitations

This is a research project, not a clinically validated decision support system. Reported performance depends on feature availability, evaluation splits, and outcome definitions. Renal measurements can overlap with the definition of AKI, so high discrimination must be interpreted alongside temporal and target leakage checks. 
