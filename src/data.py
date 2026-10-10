"""Read and inspect the genuine Telco CSV using pandas and numpy."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.config import RAW, PROCESSED, REPORTS, ensure_dirs

LOGGER = logging.getLogger(__name__)


def load_raw(path: Path = RAW) -> pd.DataFrame:
    """Load raw strings as supplied; fail with an actionable missing-file error."""
    if not path.exists():
        raise FileNotFoundError(
            f"Download Kaggle blastchar/telco-customer-churn and place the CSV at {path}"
        )
    return pd.read_csv(path)


def audit(df: pd.DataFrame) -> dict[str, Any]:
    """Return shape, types, examples, missingness, duplicates, and target balance."""
    numeric_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    result = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head().to_dict(orient="records"),
        "missing": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df["customerID"].duplicated().sum()),
        "unique_counts": df.nunique(dropna=False).to_dict(),
        "target_counts": df["Churn"].value_counts().to_dict(),
        "target_proportions": df["Churn"].value_counts(normalize=True).to_dict(),
        "total_charges_blank": int(df["TotalCharges"].astype("str").str.strip().eq("").sum()),
        "total_charges_numeric_missing": int(numeric_charges.isna().sum()),
        "blank_charge_tenures": df.loc[numeric_charges.isna(), "tenure"].tolist(),
    }
    LOGGER.info("Data audit: %s", result)
    return result


def clean(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Apply only row-local cleaning rules; leave unknown charges for train imputation.

    New customers (tenure zero) have accumulated no charges. A missing charge
    for an established customer is unknown and must not be replaced with zero.
    """
    result = df.copy(deep=True)
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].str.strip()
    result = result.drop(columns="customerID", errors="ignore")
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["TotalCharges"] = np.where(
        result["TotalCharges"].isna() & result["tenure"].eq(0),
        0.0, result["TotalCharges"],
    )
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].replace(
            {"No internet service": "No", "No phone service": "No"}
        )
    if "Churn" in result:
        if not result["Churn"].isin(["Yes", "No", 0, 1]).all():
            raise ValueError("Churn must contain only Yes/No or 0/1, with no missing target.")
        result["Churn"] = result["Churn"].replace({"Yes": 1, "No": 0}).astype(int)
    elif require_target:
        raise ValueError("Training data must include Churn.")
    return result.drop_duplicates().reset_index(drop=True)


def cleaning_comparison(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Show how types and missing counts change, including the dropped ID."""
    return pd.DataFrame({
        "before_dtype": before.dtypes.astype(str),
        "after_dtype": after.dtypes.astype(str),
        "before_nulls": before.isna().sum(),
        "after_nulls": after.isna().sum(),
    }).fillna("dropped")


def run_cleaning() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and persist the CSV and before/after audit table."""
    ensure_dirs()
    raw = load_raw()
    result = clean(raw)
    comparison = cleaning_comparison(raw, result)
    result.to_csv(PROCESSED, index=False)
    comparison.to_csv(REPORTS / "cleaning_comparison.csv")
    LOGGER.info("Cleaned %d -> %d rows; saved %s", len(raw), len(result), PROCESSED)
    LOGGER.info("Before/after:\n%s", comparison.to_string())
    return result, comparison


def load_clean() -> pd.DataFrame:
    """Load the previous step's artifact rather than silently rebuilding it."""
    if not PROCESSED.exists():
        raise FileNotFoundError("Run scripts/02_clean.py first.")
    return pd.read_csv(PROCESSED)
