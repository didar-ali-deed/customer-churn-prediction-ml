"""Ensure single-customer inference matches the saved batch pipeline."""

import json

import joblib
import numpy as np
import pytest

from src.config import MODELS, REPORTS
from src.predict import predict_one
from src.split import split_data


def test_predict_one_matches_batch() -> None:
    """Both entry points must return the same probability and threshold label."""
    train, _, _, _ = split_data()
    record = train.iloc[[0]]
    model = joblib.load(MODELS / "model.joblib")
    expected = float(model.predict_proba(record)[0, 1])
    probability, label = predict_one(record.iloc[0].to_dict())
    threshold = json.loads((REPORTS / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    assert np.isclose(probability, expected)
    assert label == int(expected >= threshold)


def test_invalid_customer_schema() -> None:
    """Fail with a useful message instead of a mysterious transformer exception."""
    with pytest.raises(ValueError, match="Missing customer fields"):
        predict_one({"tenure": 12})
    train, _, _, _ = split_data()
    customer = train.iloc[0].to_dict()
    customer["tenure"] = -1
    with pytest.raises(ValueError, match="tenure"):
        predict_one(customer)


def test_unknown_category_and_zero_tenure() -> None:
    """A future category and a new-customer bill must remain valid inputs."""
    train, _, _, _ = split_data()
    customer = train.iloc[0].to_dict()
    customer.update(Contract="Unseen contract", tenure=0, TotalCharges=" ")
    probability, label = predict_one(customer)
    assert 0 <= probability <= 1 and label in (0, 1)
