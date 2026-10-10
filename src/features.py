"""Business features and preprocessing that stay inside model pipelines."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import REPORTS, ensure_dirs
from src.eda import churn_rates, save_figure, tenure_bucket
from src.split import training_frame

LOGGER = logging.getLogger(__name__)
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
           "charge_ratio", "num_services", "is_autopay", "has_support_bundle"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
               "PaymentMethod", "tenure_bucket"]
ENGINEERED = ["tenure_bucket", "avg_monthly_spend", "charge_ratio", "num_services",
              "is_autopay", "has_support_bundle"]
RATIONALE = {
    "tenure_bucket": "Fixed 0-12, 13-24, 25-48, 49+ month groups express customer lifecycle.",
    "avg_monthly_spend": "TotalCharges / tenure summarizes historical spending; at zero tenure use the current monthly plan as a proxy.",
    "charge_ratio": "TotalCharges / (MonthlyCharges * tenure) compares historical bills with today's plan; at zero tenure use neutral 1. This is a noisy proxy, not verified price change.",
    "num_services": "Count phone, extra lines, internet, and six add-ons (0-9); bundle breadth may reflect engagement.",
    "is_autopay": "Automatic bank/card payments may reflect convenience and payment habits.",
    "has_support_bundle": "Both OnlineSecurity and TechSupport may indicate a supported service experience.",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six deterministic features without using labels or learning statistics."""
    result = df.drop(columns=["Churn", "customerID"], errors="ignore").copy()
    result["tenure_bucket"] = tenure_bucket(result["tenure"])
    months = result["tenure"].replace(0, np.nan)
    result["avg_monthly_spend"] = np.where(
        result["tenure"].eq(0), result["MonthlyCharges"], result["TotalCharges"] / months
    )
    denominator = (months * result["MonthlyCharges"]).replace(0, np.nan)
    result["charge_ratio"] = np.where(
        result["tenure"].eq(0), 1.0, result["TotalCharges"] / denominator
    )
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    result["num_services"] = result[services].eq("Yes").sum(axis=1) + result["InternetService"].ne("No").astype(int)
    result["is_autopay"] = result["PaymentMethod"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    result["has_support_bundle"] = (result["OnlineSecurity"].eq("Yes") & result["TechSupport"].eq("Yes")).astype(int)
    result[NUMERIC] = result[NUMERIC].replace([np.inf, -np.inf], np.nan)
    return result


def make_preprocessor() -> ColumnTransformer:
    """Construct an unfitted median/scale and impute/one-hot transformer."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=True)


def make_pipeline(estimator: Any) -> Pipeline:
    """Ensure feature creation and all learned preprocessing are folded with the model."""
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", make_preprocessor()),
        ("model", estimator),
    ])


def plot_feature(df: pd.DataFrame, column: str, show: bool = False) -> pd.DataFrame:
    """Show churn by feature groups (training-only quartiles for continuous plots)."""
    work = add_features(df).assign(Churn=df["Churn"])
    if column in ("avg_monthly_spend", "charge_ratio"):
        work[column] = pd.qcut(work[column], q=4, duplicates="drop").astype("str")
    rates = churn_rates(work, column)
    fig, ax = plt.subplots(figsize=(9, 4))
    rates["churn_rate"].plot.barh(ax=ax)
    ax.set(title=f"Training churn vs {column}", xlabel="Churn rate", xlim=(0, 1))
    save_figure(fig, f"feature_{column}", show)
    return rates


def run_features(show: bool = False) -> dict[str, Any]:
    """Demonstrate preprocessing on training only and persist feature diagnostics."""
    ensure_dirs()
    df = training_frame()
    for column in ENGINEERED:
        plot_feature(df, column, show).to_csv(REPORTS / f"feature_{column}.csv")
    preprocessor = make_preprocessor()
    transformed = preprocessor.fit_transform(add_features(df))
    result = {"training_rows": len(df), "output_columns": transformed.shape[1],
              "finite": bool(np.isfinite(transformed).all())}
    LOGGER.info("Feature demonstration (train only): %s", result)
    return result
