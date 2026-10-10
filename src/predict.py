"""Predict one raw-format customer with the saved pipeline and training threshold."""

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS, REPORTS
from src.data import clean
from src.evaluate import evaluation_signature
from src.features import NUMERIC, CATEGORICAL, ENGINEERED
from src.train import load_training_metadata

REQUIRED = [column for column in NUMERIC + CATEGORICAL if column not in ENGINEERED]


def predict_one(customer: dict[str, Any]) -> tuple[float, int]:
    """Return churn probability and 0/1 offer label for a complete customer record.

    Supply the 19 original predictor fields. customerID/Churn, if supplied, are
    discarded. Numeric missing bills can be imputed by the trained pipeline;
    categorical unknowns are accepted by the saved encoder.
    """
    missing = sorted(set(REQUIRED) - set(customer))
    if missing:
        raise ValueError(f"Missing customer fields: {', '.join(missing)}")
    record = pd.DataFrame([{key: customer[key] for key in REQUIRED}])
    for column in CATEGORICAL:
        if column in record and (record[column].isna().any() or not isinstance(record.iloc[0][column], str)):
            raise ValueError(f"{column} must be a nonmissing string.")
    record = clean(record, require_target=False)
    for column in ("tenure", "MonthlyCharges", "SeniorCitizen"):
        value = pd.to_numeric(record.iloc[0][column], errors="coerce")
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{column} must be finite and nonnegative.")
        if column == "SeniorCitizen" and value not in (0, 1):
            raise ValueError("SeniorCitizen must be 0 or 1.")
        record[column] = value
    if record["TotalCharges"].dropna().lt(0).any():
        raise ValueError("TotalCharges must be nonnegative or missing.")
    meta = load_training_metadata()
    threshold_path = REPORTS / "threshold.json"
    if not threshold_path.exists():
        raise FileNotFoundError("Run scripts/07_evaluate.py to select a training threshold first.")
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    if threshold["model_hash"] != meta["model_hash"] or threshold["signature"] != evaluation_signature(meta):
        raise ValueError("Threshold belongs to a different model/data recipe; run evaluation.")
    model = joblib.load(MODELS / "model.joblib")
    probability = float(model.predict_proba(record)[0, 1])
    return probability, int(probability >= threshold["threshold"])
