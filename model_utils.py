"""Training + prediction helpers. Trains at startup from embedded data (no pickle, no CSV file)."""
import io

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_embedded import load_csv_text

NUMERIC = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]
CATEGORICAL = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "final_exam_score"  # final_grade is derived from this, so it is NOT a feature


def score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    return Pipeline([("pre", pre), ("model", model)])


def load_model() -> Pipeline:
    df = pd.read_csv(io.StringIO(load_csv_text()))
    return build_pipeline().fit(df[FEATURES], df[TARGET])


def predict(pipe: Pipeline, row: dict) -> dict:
    score = float(pipe.predict(pd.DataFrame([row])[FEATURES])[0])
    score = max(0.0, min(100.0, score))
    return {"predicted_score": round(score, 1), "predicted_grade": score_to_grade(score)}
