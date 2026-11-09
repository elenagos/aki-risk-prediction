# Main writer: Bartlomiej Kaczor
# Reviewer:  Aadya Charlewar
# Contributors: Luca Dragutanu

from pathlib import Path
import pandas as pd
import streamlit as st
from model_loader_stacking import train_model, predict_patient, explain_patient

st.set_page_config(
    page_title="AKI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
    }

    .main {
        background-color: #f8f9fa;
        color: #1f2d3d;
    }

    .stApp,
    .stApp p,
    .stApp span,
    .stApp label {
        color: #1f2d3d;
    }

    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 5px solid #89AA97;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        color: #1f2d3d;
    }

    [data-testid="metric-container"] label,
    [data-testid="metric-container"] p,
    [data-testid="metric-container"] div {
        color: #1f2d3d !important;
    }

    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1f2d3d !important;
        font-weight: 700;
    }

    [data-testid="metric-container"] [data-testid="stMetricLabel"],
    [data-testid="metric-container"] [data-testid="stMetricLabel"] *,
    [data-testid="metric-container"] [data-testid="stMetricValue"],
    [data-testid="metric-container"] [data-testid="stMetricValue"] *,
    [data-testid="metric-container"] [data-testid="stMetricDelta"],
    [data-testid="metric-container"] [data-testid="stMetricDelta"] * {
        color: #1f2d3d !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .sidebar-title {
        color: #FFFFFF !important;
    }

    div.stButton > button {
        background-color: #89AA97 !important;
        color: #FFFFFF !important;
        border: 1px solid #89AA97 !important;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #789987 !important;
        color: #FFFFFF !important;
        border-color: #789987 !important;
    }

    .risk-high {
        border-left-color: #FD7E89 !important;
    }

    .risk-medium {
        border-left-color: #f39c12 !important;
    }

    .risk-low {
        border-left-color: #89AA97 !important;
    }

    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #89AA97;
    }

    .card {
        background-color: #FFFFFF;
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid #e8e3d8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        color: #2c3e50;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }

    .badge-aki {
        background-color: #FD7E89;
        color: white;
    }

    .badge-no-aki {
        background-color: #89AA97;
        color: white;
    }

    .badge-risk-high {
        background-color: #FD7E89;
        color: white;
    }

    .badge-risk-medium {
        background-color: #f39c12;
        color: white;
    }

    .badge-risk-low {
        background-color: #89AA97;
        color: white;
    }

    .prediction-box {
        background-color: #89AA97;
        color: white;
        padding: 1.8rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .prediction-large {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }

    .prediction-label {
        font-size: 0.95rem;
        opacity: 0.95;
        font-weight: 500;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #cfc7b8, transparent);
        margin: 1.5rem 0;
    }

    .data-point {
        padding: 0.75rem;
        border-left: 3px solid #89AA97;
        margin: 0.5rem 0;
        background-color: #F6F1E7;
        border-radius: 4px;
    }

    .data-point-label {
        color: #7f8c8d;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .data-point-value {
        color: #2c3e50;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }

    .dashboard-subtitle {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 1.5rem;
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }

    .main-header {
        background-color: #89AA97;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 900;
        margin: 0;
    }

    .main-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
CSV_PATHS = [
    BASE_DIR / "data" / "final" / "final_aki_dataset_engineered_FIXED.csv",
    BASE_DIR / "data" / "final" / "final_aki_dataset.csv",
]
PARQUET_PATHS = [
    BASE_DIR / "data" / "final" / "final_aki_dataset_engineered_FIXED.parquet",
    BASE_DIR / "data" / "final" / "final_aki_dataset.parquet",
]

COMORBIDITY_COLS = [
    "ckd_flag", "esrd_flag", "diabetes_flag", "hypertension_flag",
    "heart_failure_flag", "ischemic_heart_disease_flag", "mi_flag",
    "arrhythmia_flag", "copd_flag", "liver_disease_flag",
    "malignancy_flag", "pvd_flag",
]

TOXIN_COLS = [
    "vasopressor_use", "aminoglycoside_use", "loop_diuretic_use",
    "nsaid_use", "acei_arb_use", "amphotericin_use", "sedative_use",
]

DATETIME_COLS = [
    "intime", "outtime", "admittime", "dischtime",
    "index_time", "vanco_starttime", "ptz_starttime",
]

HIGH_RISK_THRESHOLD = 0.45
MEDIUM_RISK_THRESHOLD = 0.15


def yes_no(value):
    if pd.isna(value):
        return "Unknown"
    return "Yes" if bool(value) else "No"


def safe_num(v, digits=2, suffix=""):
    if pd.isna(v):
        return "—"
    return f"{v:.{digits}f}{suffix}"


def safe_int(v):
    if pd.isna(v):
        return "—"
    return str(int(v))


def exposure_label(row):
    if bool(row.get("vanco_ptz_combo", False)):
        return "Vancomycin + PTZ"
    if bool(row.get("vanco_only", False)):
        return "Vancomycin only"
    return "Other / unknown"


def show_metric(label, value):
    formatted_value = value if value not in [None, "", "nan"] else "—"
    st.metric(label, formatted_value)

REFERENCE_RANGES = {
"bun_mean": (7, 24),
"baseline_creatinine": (0.6, 1.3),
"heart_rate_mean": (60, 100),
"respiration_mean": (12, 18),
"spo2_mean": (95, 100),
"temperature_mean": (36.5, 37.5),
"lactate_mean": (0.5, 2.0),
"wbc_mean": (4, 11),
}


def normalize_temperature(value):
    if pd.isna(value):
        return value
    value = float(value)
    if value > 45:  # likely Fahrenheit
        value = (value - 32) * 5 / 9
    return value


def feature_status(feature, value):
    if pd.isna(value):
        return "Unknown"

    if feature == "temperature_mean":
        value = normalize_temperature(value)

    low, high = REFERENCE_RANGES[feature]

    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "✔Normal"


def count_abnormal_features(row):
    count = 0

    for feature in REFERENCE_RANGES:
        if feature in row.index:
            status = feature_status(feature, row.get(feature))
            if status in ["Low", "High"]:
                count += 1

    return count


def clinical_status(row):
    abnormal_count = count_abnormal_features(row)

    if abnormal_count >= 3:
        return "High clinical concern"
    elif abnormal_count >= 1:
        return "Moderate clinical concern"
    return "Normal range"


def render_prediction_explanation(row):
    st.markdown('<div class="section-header">Prediction Explanation</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Calculating clinical feature explanation..."):
            explanation = explain_patient(row, top_n=10)
    except ModuleNotFoundError:
        st.warning("SHAP is not installed. Install it with: python -m pip install shap")
        return
    except Exception as exc:
        st.warning(f"Could not calculate prediction explanation: {exc}")
        return

    display_df = explanation.copy()
    display_df["direction"] = display_df["shap_impact"].apply(
        lambda value: "Increases AKI risk" if value > 0 else "Decreases AKI risk"
    )
    display_df["value"] = pd.to_numeric(display_df["value"], errors="coerce").round(3)
    display_df["shap_impact"] = pd.to_numeric(display_df["shap_impact"], errors="coerce").round(4)
    display_df = display_df.rename(
        columns={
            "feature": "Feature",
            "value": "Patient value",
            "shap_impact": "SHAP impact",
            "direction": "Direction",
        }
    )

    st.dataframe(display_df, width="stretch", hide_index=True)

    chart_df = display_df[["Feature", "SHAP impact"]].set_index("Feature")
    st.bar_chart(chart_df)


@st.cache_data
def load_data():
    df = None
    for path in PARQUET_PATHS:
        if path.exists():
            df = pd.read_parquet(path)
            break

    if df is None:
        for path in CSV_PATHS:
            if path.exists():
                df = pd.read_csv(path)
                break

    if df is None:
        st.error(
            "Dataset not found. Put one of the following files in data/final/ next to this app.py file:"
            " final_aki_dataset_engineered_FIXED.csv, final_aki_dataset.csv,"
            " or the corresponding .parquet file."
        )
        st.stop()

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "stay_id" in df.columns:
        df = df.sort_values("stay_id").reset_index(drop=True)

    return df


def selected_patient(df, stay_id):
    return df.loc[df["stay_id"] == stay_id].iloc[0]


def render_dataset_summary(df):
    st.markdown('<div class="dashboard-title"> Dataset Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Overview of all patient stays in the dataset</div>', unsafe_allow_html=True)

    total_patients = len(df)
    aki_count = int(df["aki_day7"].sum()) if "aki_day7" in df.columns else 0
    no_aki_count = total_patients - aki_count
    aki_rate = (aki_count / total_patients * 100) if total_patients else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patient Stays", f"{total_patients:,}")
    c2.metric("AKI Within 7 Days", f"{aki_count:,}")
    c3.metric("No AKI Within 7 Days", f"{no_aki_count:,}")
    c4.metric("AKI Rate", f"{aki_rate:.1f}%")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1])

    with left:
        st.markdown('<div class="section-header">Outcome Distribution</div>', unsafe_allow_html=True)
        outcome_counts = (
            df["aki_day7"]
            .value_counts(dropna=False)
            .rename(index={False: "No AKI", True: "AKI"})
        )
        st.bar_chart(outcome_counts)

    with right:
        st.markdown('<div class="section-header">Dataset Facts</div>', unsafe_allow_html=True)
        sex_counts = df["sex"].value_counts(dropna=False) if "sex" in df.columns else pd.Series(dtype=int)
        combo_count = int(df["vanco_ptz_combo"].sum()) if "vanco_ptz_combo" in df.columns else 0
        vanco_only_count = int(df["vanco_only"].sum()) if "vanco_only" in df.columns else 0

        st.markdown(f"""
        <div class="card">
        <div class="data-point">
            <div class="data-point-label">Mean Age</div>
            <div class="data-point-value">{safe_num(df['age'].mean(), 1) if 'age' in df.columns else '—'} years</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Female Patients</div>
            <div class="data-point-value">{sex_counts.get('F', 0):,}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Male Patients</div>
            <div class="data-point-value">{sex_counts.get('M', 0):,}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Vancomycin Only</div>
            <div class="data-point-value">{vanco_only_count:,}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Vancomycin + PTZ</div>
            <div class="data-point-value">{combo_count:,}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    

    st.markdown('<div class="section-header">Choose Graph</div>', unsafe_allow_html=True)
    graph_choice = st.selectbox(
        "Select graph",
        ["Age Distribution", "AKI Distribution"],
        key="dataset_graph_select"
    )

    if graph_choice == "Age Distribution" and "age" in df.columns:
        age_hist = df["age"].dropna()
        st.bar_chart(age_hist.value_counts(bins=10, sort=False))
    elif graph_choice == "AKI Distribution":
        st.bar_chart(df["aki_day7"].value_counts())


def render_patient_details(df, label_to_stay):
    st.markdown("### Patient Filters")

    f1, f2, f3 = st.columns(3)

    with f1:
        sex_filter = st.selectbox(
            "Sex",
            ["All", "M", "F"]
        )

    with f2:
        min_age, max_age = st.slider(
            "Age Range",
            18,
            100,
            (40, 80)
        )

    with f3:
        aki_filter = st.selectbox(
            "Outcome",
            ["All", "AKI", "No AKI"]
        )

    filtered_df = df.copy()

    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["sex"] == sex_filter]

    filtered_df = filtered_df[
        (filtered_df["age"] >= min_age) &
        (filtered_df["age"] <= max_age)
    ]

    if aki_filter == "AKI":
        filtered_df = filtered_df[filtered_df["aki_day7"] == 1]

    elif aki_filter == "No AKI":
        filtered_df = filtered_df[filtered_df["aki_day7"] == 0]

    filtered_selector = filtered_df[
        ["stay_id", "sex", "age", "aki_day7"]
    ].copy()

    filtered_selector["label"] = filtered_selector.apply(
        lambda r: f"{int(r['stay_id'])} | {r['sex']} | {int(r['age'])}",
        axis=1
    )

    if filtered_selector.empty:
        st.warning("No patients match the selected filters.")
        return

    selected_label = st.selectbox(
        "Select Patient",
        filtered_selector["label"].tolist()
    )

    selected_stay_id = int(selected_label.split("|")[0].strip())

    row = selected_patient(df, selected_stay_id)
    st.markdown('<div class="dashboard-title"> Individual Patient Profile</div>', unsafe_allow_html=True)

    pred_prob, pred_label = predict_patient(row)

    if pred_prob >= HIGH_RISK_THRESHOLD:
        risk_level = "🔴HIGH"
       
    elif pred_prob >= MEDIUM_RISK_THRESHOLD:
        risk_level = "🟠MEDIUM"
   
    else:
        risk_level = "LOW"
        

    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">PREDICTED AKI RISK</div>
        <div class="prediction-large">{pred_prob*100:.1f}%</div>
        <div class="prediction-label">{risk_level} RISK</div>
    </div>
    """, unsafe_allow_html=True)

    render_prediction_explanation(row)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Identifiers & Demographics</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
        <div class="data-point">
            <div class="data-point-label">Stay ID</div>
            <div class="data-point-value">{safe_int(row.get('stay_id'))}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Subject ID</div>
            <div class="data-point-value">{safe_int(row.get('subject_id'))}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Hospital Admission ID</div>
            <div class="data-point-value">{safe_int(row.get('hadm_id'))}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Sex</div>
            <div class="data-point-value">{row.get('sex', '—')}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Age</div>
            <div class="data-point-value">{safe_int(row.get('age'))} years</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Length of Stay</div>
            <div class="data-point-value">{safe_num(row.get('los_hours'), 1)} hours</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">Antibiotic Exposure</div>', unsafe_allow_html=True)
        exposure = exposure_label(row)
        vanco_emoji = "✅" if yes_no(row.get('vancomycin_use')) == "Yes" else "❌"
        ptz_emoji = "✅" if yes_no(row.get('piperacillin_tazobactam_use')) == "Yes" else "❌"

        st.markdown(f"""
        <div class="card">
        <div class="data-point">
            <div class="data-point-label">Exposure Group</div>
            <div class="data-point-value">{exposure}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Vancomycin Use {vanco_emoji}</div>
            <div class="data-point-value">{yes_no(row.get('vancomycin_use'))}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Piperacillin-Tazobactam Use {ptz_emoji}</div>
            <div class="data-point-value">{yes_no(row.get('piperacillin_tazobactam_use'))}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Vancomycin Start</div>
            <div class="data-point-value">{row['vanco_starttime'] if pd.notna(row.get('vanco_starttime')) else '—'}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">PTZ Start</div>
            <div class="data-point-value">{row['ptz_starttime'] if pd.notna(row.get('ptz_starttime')) else '—'}</div>
        </div>
        <div class="data-point">
            <div class="data-point-label">Index Time</div>
            <div class="data-point-value">{row['index_time'] if pd.notna(row.get('index_time')) else '—'}</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Clinical Outcomes & Predictions</div>', unsafe_allow_html=True)

    observed_aki = "AKI" if bool(row.get("aki_day7")) else "No AKI"
    observed_emoji = "⚠️" if observed_aki == "AKI" else "✅"

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        show_metric(f"Observed Outcome {observed_emoji}", observed_aki)
    with s2:
        show_metric("Predicted Risk", f"{pred_prob*100:.1f}%")
    with s3:
        show_metric(f"Predicted Class", risk_level)
    with s4:
        show_metric("Time to AKI", safe_num(row.get("time_to_aki_hours"), 1))
    with s5:
        creat = row.get("baseline_creatinine")
        show_metric("Baseline Creatinine", f"{safe_num(creat, 2)} ({feature_status('baseline_creatinine', creat)})")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Laboratory Values</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        bun = row.get("bun_mean")
        show_metric("BUN", f"{safe_num(bun, 1)} ({feature_status('bun_mean', bun)})")

        lactate = row.get("lactate_mean")
        show_metric("Lactate", f"{safe_num(lactate, 1)} ({feature_status('lactate_mean', lactate)})")

        wbc = row.get("wbc_mean")
        show_metric("WBC", f"{safe_num(wbc, 1)} ({feature_status('wbc_mean', wbc)})")

    with c2:
        show_metric("Platelets Mean", safe_num(row.get("platelets_mean"), 1))

    st.markdown('<div class="section-header">Vital Signs</div>', unsafe_allow_html=True)
    v1, v2 = st.columns(2)

    with v1:
        hr = row.get("heart_rate_mean")
        show_metric("Heart Rate", f"{safe_num(hr, 1)} ({feature_status('heart_rate_mean', hr)})")

        resp = row.get("respiration_mean")
        show_metric("Respiration", f"{safe_num(resp, 1)} ({feature_status('respiration_mean', resp)})")

    with v2:
        spo2 = row.get("spo2_mean")
        show_metric("SpO₂", f"{safe_num(spo2, 1)} ({feature_status('spo2_mean', spo2)})")

        temp = normalize_temperature(row.get("temperature_mean"))
        show_metric("Temperature", f"{safe_num(temp, 1)} °C ({feature_status('temperature_mean', temp)})")
        
    st.markdown('<div class="section-header">Urine Output</div>', unsafe_allow_html=True)
    u1, u2, u3 = st.columns(3)

    with u1:
        show_metric("6h Output", safe_num(row.get("urine_output_6h"), 0))
    with u2:
        show_metric("12h Output", safe_num(row.get("urine_output_12h"), 0))
    with u3:
        show_metric("24h Output", safe_num(row.get("urine_output_24h"), 0))

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Comorbidities</div>', unsafe_allow_html=True)
        comorbidity_names = []
        for col in COMORBIDITY_COLS:
            if col in row.index and bool(row[col]):
                comorbidity_names.append(col.replace("_flag", "").replace("_", " ").title())

        if comorbidity_names:
            for item in comorbidity_names:
                st.markdown(f"<div class='badge badge-aki'>{item}</div>", unsafe_allow_html=True)
        else:
            st.info("No comorbidities detected in this patient record")

    with right:
        st.markdown('<div class="section-header">Nephrotoxic / Severity Indicators</div>', unsafe_allow_html=True)
        tox_names = []
        for col in TOXIN_COLS:
            if col in row.index and bool(row[col]):
                tox_names.append(col.replace("_use", "").replace("_", " ").title())

        if tox_names:
            for item in tox_names:
                st.markdown(f"<div class='badge badge-risk-medium'>{item}</div>", unsafe_allow_html=True)
        else:
            st.info("No nephrotoxic exposures detected in this patient record")


def render_dashboard(df):
    st.markdown('<div class="dashboard-title">Dashboard Overview</div>', unsafe_allow_html=True)
    full_preview = df.head(10).copy()

    full_preview["Regimen"] = full_preview.apply(
        lambda r: "Vancomycin + PTZ"
        if bool(r["vanco_ptz_combo"])
        else "Vancomycin only"
        if bool(r["vanco_only"])
        else "Other",
        axis=1,
    )

    def risk_label(r):
        prob, _ = predict_patient(r)

        if prob >= HIGH_RISK_THRESHOLD:
            return "High"

        elif prob >= MEDIUM_RISK_THRESHOLD:
            return "Medium"

        return "Low"

    full_preview["Model Risk"] = full_preview.apply(risk_label, axis=1)

    full_preview["Clinical Status"] = full_preview.apply(
        clinical_status,
        axis=1
    )

    full_preview["Abnormal Features"] = full_preview.apply(
        count_abnormal_features,
        axis=1
    )

    preview = full_preview[
        [
            "stay_id",
            "Regimen",
            "Model Risk",
            "Clinical Status",
            "Abnormal Features"
        ]
    ].copy()

    preview.columns = [
        "Stay ID",
        "Regimen",
        "Model Risk",
        "Clinical Status",
        "Abnormal Features",
    ]

    st.dataframe(preview, width="stretch", hide_index=True)
    st.markdown('<div class="dashboard-subtitle">Quick monitoring overview for the AKI prototype</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Active Watchlist</div>', unsafe_allow_html=True)

        full_preview = df.head(10).copy()

        full_preview["Regimen"] = full_preview.apply(
            lambda r: "Vancomycin + PTZ"
            if bool(r["vanco_ptz_combo"])
            else "Vancomycin only"
            if bool(r["vanco_only"])
            else "Other",
            axis=1,
        )

        def risk_label(r):
            prob, _ = predict_patient(r)
            if prob >= HIGH_RISK_THRESHOLD:
                return "High"
            elif prob >= MEDIUM_RISK_THRESHOLD:
                return "Medium"
            return "Low"

        full_preview["Risk"] = full_preview.apply(risk_label, axis=1)

        preview = full_preview[["stay_id", "Regimen", "Risk"]].copy()
        preview.columns = ["Stay ID", "Regimen", "Risk"]

        st.dataframe(preview, width="stretch", hide_index=True)

    with col2:
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
        <b>Model:</b> Stacking Ensemble<br>
        <b>Threshold:</b> 0.45<br>
        <b>Dataset:</b> Final AKI dataset<br>
        <b>Missing values:</b> Median imputation
        </div>
        """, unsafe_allow_html=True)

        total_patients = len(df)
        aki_count = int(df["aki_day7"].sum()) if "aki_day7" in df.columns else 0
        st.metric("Total Stays", f"{total_patients:,}")
        st.metric("AKI Cases", f"{aki_count:,}")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-header">Database Status</div>', unsafe_allow_html=True)
        st.success("Connected to final AKI dataset")

    with right:
        st.markdown('<div class="section-header">Data Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <div class="badge badge-risk-low">Cleaning</div>
            <div class="badge badge-risk-low">Missing values</div>
            <div class="badge badge-risk-medium">Threshold</div>
            <div class="badge badge-risk-high">Stacking Ensemble</div>
        </div>
        """, unsafe_allow_html=True)


def render_new_patient_page(row):
    st.markdown('<div class="dashboard-title">New Patient Intake</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">Enter a few clinical values to generate a prototype prediction</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
    "Upload new patient Excel file",
    type=["xlsx", "csv"]
)

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".xlsx"):
            new_patients = pd.read_excel(uploaded_file)
        else:
            new_patients = pd.read_csv(uploaded_file)

        st.dataframe(new_patients, width="stretch")

        selected_row_index = st.selectbox(
            "Select uploaded patient",
            new_patients.index.tolist()
        )

        patient_row = new_patients.loc[selected_row_index]

        if st.button("Predict uploaded patient"):
            prob, pred = predict_patient(patient_row)

            st.success(f"Predicted AKI risk: {prob*100:.1f}%")
            
    temp_default = float(row.get("temperature_mean", 37.0))
    if temp_default > 45:
        temp_default = (temp_default - 32) * 5 / 9
    temp_default = max(30.0, min(42.0, temp_default))
    temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=42.0, value=temp_default)

    age = st.number_input("Age", min_value=0, max_value=100, value=int(row.get("age", 60)))
    hr = st.number_input("Heart Rate", min_value=30, max_value=200, value=int(row.get("heart_rate_mean", 80)))
    bun = st.number_input("BUN", min_value=0.0, max_value=200.0, value=float(row.get("bun_mean", 20.0)))
    creatinine = st.number_input("Creatinine", min_value=0.0, max_value=10.0, value=float(row.get("baseline_creatinine", 1.0)))
    spo2 = st.number_input("SpO₂", min_value=0, max_value=100, value=int(row.get("spo2_mean", 98)))
   
    if st.button("Predict AKI Risk"):
        fake_row = row.copy()
        fake_row["age"] = age
        fake_row["heart_rate_mean"] = hr
        fake_row["bun_mean"] = bun
        fake_row["baseline_creatinine"] = creatinine
        fake_row["spo2_mean"] = spo2
        fake_row["temperature_mean"] = temp

        prob, pred = predict_patient(fake_row)

        if prob >= HIGH_RISK_THRESHOLD:
            risk_level = "🔴HIGH"
           
        elif prob >= MEDIUM_RISK_THRESHOLD:
            risk_level = "🟠MEDIUM"
           
        else:
            risk_level = "LOW"
            

        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-label">PREDICTED AKI RISK</div>
            <div class="prediction-large">{prob*100:.1f}%</div>
            <div class="prediction-label"> {risk_level} RISK</div>
        </div>
        """, unsafe_allow_html=True)

        render_prediction_explanation(fake_row)


def main():
    df = load_data()

    @st.cache_resource
    def load_model(df):
        train_model(df)

    load_model(df)

    st.sidebar.markdown('<div class="sidebar-title"> AKI Prototype</div>', unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border: 2px solid #89AA97;'>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigate to",
        ["Dashboard", "Individual Patient", "New Patient", "Dataset Summary"],
        label_visibility="collapsed"
    )

    selector_df = df[["stay_id", "sex", "age", "aki_day7", "vanco_only", "vanco_ptz_combo"]].copy()
    selector_df["label"] = selector_df.apply(
        lambda r: f"{int(r['stay_id'])} | {r['sex']} | {int(r['age'])} | "
        f"{'Combo' if bool(r['vanco_ptz_combo']) else 'Vanco only' if bool(r['vanco_only']) else 'Other'} | "
        f"{'AKI' if bool(r['aki_day7']) else 'No AKI'}",
        axis=1,
    )

    label_to_stay = dict(zip(selector_df["label"], selector_df["stay_id"]))

    

    st.markdown("""
    <div class="main-header">
        <div class="main-title"> Acute Kidney Injury Dashboard Prototype</div>
        <div class="main-subtitle">Predictive Analytics & Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    if page == "Dashboard":
        render_dashboard(df)
    elif page == "Individual Patient":
        render_patient_details(df, label_to_stay)
    elif page == "New Patient":
        render_new_patient_page(df.iloc[0])
    elif page == "Dataset Summary":
        render_dataset_summary(df)


if __name__ == "__main__":
    main()
