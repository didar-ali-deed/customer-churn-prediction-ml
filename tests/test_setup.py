"""Verify the real input is the expected Telco dataset before building on it."""

import pandas as pd

from src.config import RAW


def test_source_schema() -> None:
    """Reject a missing/wrong CSV rather than silently training on other data."""
    df = pd.read_csv(RAW)
    assert df.shape == (7043, 21)
    assert {"customerID", "tenure", "TotalCharges", "Churn"} <= set(df.columns)
    assert set(df["Churn"].unique()) == {"Yes", "No"}
