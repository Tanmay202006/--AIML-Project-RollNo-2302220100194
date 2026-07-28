from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Dataset" / "Salary Data.csv"
MODEL_PATH = ROOT / "salary_model.pkl"
FEATURES_PATH = ROOT / "salary_feature_columns.pkl"


def build_model():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Age", "Gender", "Education Level", "Job Title", "Years of Experience", "Salary"]).copy()

    for col in ["Gender", "Education Level", "Job Title"]:
        df[col] = df[col].astype(str).str.strip()

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
    fe_cols = baseline_cols + ["age_experience_ratio"] + [
        c for c in df_encoded.columns if c.startswith("experience_level_")
    ]

    X = df_encoded[fe_cols]
    y = df_encoded["Salary"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(fe_cols, FEATURES_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved feature columns to {FEATURES_PATH}")


if __name__ == "__main__":
    build_model()
