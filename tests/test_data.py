"""Test every cleaning rule using the supplied real dataset and edge mutations."""

import numpy as np
import pandas as pd
import pytest

from src.data import clean, load_raw


@pytest.fixture
def raw() -> pd.DataFrame:
    """Return the genuine CSV, independent for each test."""
    return load_raw()


def test_identifier_removed(raw: pd.DataFrame) -> None:
    """The identifier must not become a predictive feature."""
    assert "customerID" not in clean(raw)


def test_blank_charges_zero_only_for_new_customers(raw: pd.DataFrame) -> None:
    """New-customer blanks become zero; unknown established bills stay missing."""
    blank = raw["TotalCharges"].str.strip().eq("")
    result = clean(raw.loc[blank])
    assert result["tenure"].eq(0).all()
    assert result["TotalCharges"].eq(0).all()
    established = raw.iloc[[0]].copy()
    established["TotalCharges"] = "bad"
    assert np.isnan(clean(established).iloc[0]["TotalCharges"])


def test_numeric_conversion(raw: pd.DataFrame) -> None:
    """Charges are numeric and valid values survive conversion."""
    result = clean(raw)
    assert pd.api.types.is_numeric_dtype(result["TotalCharges"])
    assert result.iloc[0]["TotalCharges"] == 29.85


def test_whitespace(raw: pd.DataFrame) -> None:
    """Whitespace must not create spurious categories or target labels."""
    sample = raw.iloc[[0]].copy()
    sample["Contract"] = "  Month-to-month  "
    sample["Churn"] = " No "
    result = clean(sample)
    assert result.iloc[0]["Contract"] == "Month-to-month"
    assert result.iloc[0]["Churn"] == 0


def test_service_normalization(raw: pd.DataFrame) -> None:
    """Absence of a parent service is normalized to absence of its add-ons."""
    result = clean(raw)
    assert result["MultipleLines"].isin(["Yes", "No"]).all()
    for column in ("OnlineSecurity", "TechSupport", "StreamingTV"):
        assert result[column].isin(["Yes", "No"]).all()


def test_target_mapping_and_invalid_label(raw: pd.DataFrame) -> None:
    """Reject unknown targets instead of manufacturing labels."""
    assert set(clean(raw)["Churn"]) == {0, 1}
    sample = raw.iloc[[0]].copy()
    sample["Churn"] = "Maybe"
    with pytest.raises(ValueError, match="Churn"):
        clean(sample)


def test_duplicates_and_no_mutation(raw: pd.DataFrame) -> None:
    """Identical cleaned records collapse; the raw source remains untouched."""
    sample = pd.concat([raw.iloc[[0]], raw.iloc[[0]]], ignore_index=True)
    original = sample.copy(deep=True)
    assert len(clean(sample)) == 1
    pd.testing.assert_frame_equal(sample, original)


def test_cleaning_idempotent(raw: pd.DataFrame) -> None:
    """Reapplying deterministic cleaning must not change cleaned values."""
    result = clean(raw)
    pd.testing.assert_frame_equal(clean(result), result)
