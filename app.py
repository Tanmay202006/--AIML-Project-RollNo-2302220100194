import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "salary_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "salary_feature_columns.pkl"
DATA_PATH = PROJECT_ROOT / "Dataset" / "Salary Data.csv"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("The model files were not found. Please run the notebook once to export them.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


@st.cache_data
def load_salary_data():
    if not DATA_PATH.exists():
        st.error("The dataset file was not found.")
        st.stop()
    return pd.read_csv(DATA_PATH)


def build_feature_frame(age, experience, gender, education, job_title):
    _, feature_columns = load_artifacts()
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


def prepare_training_data(df):
    df = df.dropna(subset=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]).copy()

    for col in ["Gender", "Education Level", "Job Title"]:
        df[col] = df[col].astype(str).str.strip()

    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"]).copy()

    def exp_bucket(x):
        if x < 3:
            return "Entry"
        if x < 8:
            return "Mid"
        return "Senior"

    df["experience_level"] = df["Years of Experience"].apply(exp_bucket)
    df["age_experience_ratio"] = df["Age"] / (df["Years of Experience"] + 1)

    title_counts = df["Job Title"].value_counts()
    common_titles = set(title_counts[title_counts > 1].index.tolist())
    df["Job Title Grouped"] = df["Job Title"].apply(lambda x: x if x in common_titles else "Other")

    cat_cols = ["Gender", "Education Level", "Job Title Grouped"]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    df_encoded["experience_level"] = df["experience_level"]
    df_encoded["age_experience_ratio"] = df["age_experience_ratio"]
    df_encoded = pd.get_dummies(df_encoded, columns=["experience_level"], drop_first=True)

    baseline_cols = ["Age", "Years of Experience"] + [
        c for c in df_encoded.columns if c.startswith(("Gender_", "Education Level_", "Job Title Grouped_"))
    ]
    feature_columns = baseline_cols + ["age_experience_ratio"] + [
        c for c in df_encoded.columns if c.startswith("experience_level_")
    ]

    X = df_encoded[feature_columns]
    y = df_encoded["Salary"]
    return X, y, feature_columns


def evaluate_model(df):
    X, y, _ = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "R²": round(float(r2_score(y_test, predictions)), 3),
        "MAE": round(float(mean_absolute_error(y_test, predictions)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 2),
    }
    comparison_df = pd.DataFrame({"Actual Salary": y_test.to_numpy(), "Predicted Salary": predictions})
    return metrics, comparison_df


st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("Salary Prediction App")
st.write("Estimate salaries with a trained regression model and explore the dataset through interactive charts.")

salary_data = load_salary_data()

prediction_tab, eda_tab, performance_tab = st.tabs(["Prediction", "EDA", "Model Performance"])

with prediction_tab:
    st.subheader("Predict a Salary")
    with st.form("salary_form"):
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
        job_title = st.text_input("Job Title", value="Software Engineer")
        submitted = st.form_submit_button("Predict Salary")

    if submitted:
        predicted_salary = predict_salary(age, experience, gender, education, job_title)
        st.success(f"Estimated Salary: ${predicted_salary:,.0f}")

with eda_tab:
    st.subheader("Exploratory Data Analysis")
    st.caption("These charts help you understand how salary changes across experience, education, and gender.")

    col1, col2 = st.columns(2)
    with col1:
        experience_summary = salary_data.groupby("Years of Experience")["Salary"].mean().reset_index()
        experience_summary = experience_summary.sort_values("Years of Experience")
        st.write("Average Salary by Experience")
        st.line_chart(experience_summary.set_index("Years of Experience"))

    with col2:
        education_summary = salary_data.groupby("Education Level")["Salary"].mean().sort_values(ascending=False)
        st.write("Average Salary by Education Level")
        st.bar_chart(education_summary)

    col3, col4 = st.columns(2)
    with col3:
        gender_summary = salary_data.groupby("Gender")["Salary"].mean().sort_values(ascending=False)
        st.write("Average Salary by Gender")
        st.bar_chart(gender_summary)

    with col4:
        title_counts = salary_data["Job Title"].value_counts().head(10)
        st.write("Top Job Titles in the Dataset")
        st.bar_chart(title_counts)

with performance_tab:
    st.subheader("Model Performance")
    st.caption("The model is evaluated on a held-out test set using standard regression metrics.")

    metrics, comparison_df = evaluate_model(salary_data)

    metric_cols = st.columns(3)
    metric_cols[0].metric("R² Score", f"{metrics['R²']}")
    metric_cols[1].metric("MAE", f"${metrics['MAE']:,.0f}")
    metric_cols[2].metric("RMSE", f"${metrics['RMSE']:,.0f}")

    st.write("Actual vs Predicted Salary")
    st.scatter_chart(comparison_df, x="Actual Salary", y="Predicted Salary")

    performance_summary = pd.DataFrame({"Metric": ["R² Score", "MAE", "RMSE"], "Value": [metrics["R²"], metrics["MAE"], metrics["RMSE"]]}).set_index("Metric")
    st.write("Metric Comparison")
    st.bar_chart(performance_summary)
