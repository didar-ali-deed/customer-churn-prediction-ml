"""Guard against division errors, unknown-category failures, and leakage."""

import numpy as np
import pandas as pd

from src.features import add_features, make_preprocessor
from src.split import split_data


def test_finite_and_stable_output() -> None:
    """Unknown test categories/missing bills must preserve the fitted schema."""
    train, test, _, _ = split_data()
    pre = make_preprocessor()
    a = pre.fit_transform(add_features(train))
    unseen = test.iloc[:10].copy()
    unseen["Contract"] = "Unseen contract"
    unseen["TotalCharges"] = np.nan
    b = pre.transform(add_features(unseen))
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a.shape[1] == b.shape[1] == len(pre.get_feature_names_out())


def test_training_statistics_are_unchanged() -> None:
    """Transforming extreme future values must not refit training means."""
    train, test, _, _ = split_data()
    pre = make_preprocessor().fit(add_features(train))
    before = pre.named_transformers_["num"].named_steps["scale"].mean_.copy()
    unseen = test.iloc[:5].copy()
    unseen["MonthlyCharges"] = 1e8
    pre.transform(add_features(unseen))
    np.testing.assert_array_equal(before, pre.named_transformers_["num"].named_steps["scale"].mean_)


def test_zero_tenure_and_service_logic() -> None:
    """Zero tenure has finite neutral proxies and service counts are interpretable."""
    train, _, _, _ = split_data()
    sample = train.iloc[[0]].copy()
    sample["tenure"] = 0
    sample["TotalCharges"] = 0.0
    sample["PaymentMethod"] = "Credit card (automatic)"
    sample["OnlineSecurity"] = "Yes"
    sample["TechSupport"] = "Yes"
    result = add_features(sample)
    assert result.iloc[0]["avg_monthly_spend"] == sample.iloc[0]["MonthlyCharges"]
    assert result.iloc[0]["charge_ratio"] == 1
    assert result.iloc[0]["is_autopay"] == result.iloc[0]["has_support_bundle"] == 1
    assert 0 <= result.iloc[0]["num_services"] <= 9
    assert "Churn" not in result and "customerID" not in result
    pd.testing.assert_frame_equal(sample, train.iloc[[0]].assign(
        tenure=0, TotalCharges=0.0, PaymentMethod="Credit card (automatic)",
        OnlineSecurity="Yes", TechSupport="Yes",
    ))
