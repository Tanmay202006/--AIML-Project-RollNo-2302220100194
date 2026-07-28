import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "salary_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "salary_feature_columns.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("The model files were not found. Please run the notebook once to export them.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def build_feature_frame(age, experience, gender, education, job_title):
    model, feature_columns = load_artifacts()
    feature_values = {column: 0 for column in feature_columns}

    feature_values["Age"] = float(age)
    feature_values["Years of Experience"] = float(experience)
    feature_values["age_experience_ratio"] = float(age) / (float(experience) + 1)

    if experience < 3:
        experience_level = "Entry"
    elif experience < 8:
        experience_level = "Mid"
    else:
        experience_level = "Senior"

    if experience_level == "Mid" and "experience_level_Mid" in feature_values:
        feature_values["experience_level_Mid"] = 1
    elif experience_level == "Senior" and "experience_level_Senior" in feature_values:
        feature_values["experience_level_Senior"] = 1

    if gender == "Male" and "Gender_Male" in feature_values:
        feature_values["Gender_Male"] = 1

    if education == "Master's" and "Education Level_Master's" in feature_values:
        feature_values["Education Level_Master's"] = 1
    elif education == "PhD" and "Education Level_PhD" in feature_values:
        feature_values["Education Level_PhD"] = 1

    normalized_title = " ".join(job_title.strip().lower().split())
    for column in feature_columns:
        if not column.startswith("Job Title Grouped_"):
            continue
        suffix = column.split("Job Title Grouped_", 1)[1].lower()
        if suffix == "other":
            continue
        if normalized_title == suffix or normalized_title in suffix or suffix in normalized_title:
            feature_values[column] = 1
            break
    else:
        if "Job Title Grouped_Other" in feature_values:
            feature_values["Job Title Grouped_Other"] = 1

    return pd.DataFrame([feature_values], columns=feature_columns)


def predict_salary(age, experience, gender, education, job_title):
    model, _ = load_artifacts()
    features = build_feature_frame(age, experience, gender, education, job_title)
    prediction = model.predict(features)[0]
    return round(float(prediction), 2)


st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("Salary Prediction App")
st.write("Fill in the form below to estimate a salary based on the trained regression model.")

with st.form("salary_form"):
    age = st.number_input("Age", min_value=18, max_value=80, value=35)
    experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    job_title = st.text_input("Job Title", value="Software Engineer")
    submitted = st.form_submit_button("Predict Salary")

if submitted:
    predicted_salary = predict_salary(age, experience, gender, education, job_title)
    st.success(f"Estimated Salary:  ₹{predicted_salary:,.0f}")
