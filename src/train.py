"""Training-only baseline cross-validation; model comparison is added next."""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import REPORTS, SEED, ensure_dirs
from src.features import make_pipeline
from src.split import split_data

LOGGER = logging.getLogger(__name__)
SCORING = {
    "roc_auc": "roc_auc", "average_precision": "average_precision",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0), "accuracy": "accuracy",
}


def cv_folds() -> StratifiedKFold:
    """Use the same five shuffled stratified folds for every model."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def logistic_model() -> LogisticRegression:
    """Build a reproducible imbalance-aware linear reference model."""
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)


def cv_row(name: str, estimator: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Fit a fresh complete pipeline in every fold and summarize train/CV scores."""
    scores = cross_validate(make_pipeline(estimator), x, y, scoring=SCORING,
                            cv=cv_folds(), return_train_score=True, n_jobs=1,
                            error_score="raise")
    result: dict[str, Any] = {"model": name}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        result[f"{metric}_mean"] = float(np.mean(values))
        result[f"{metric}_std"] = float(np.std(values, ddof=1))
        result[f"train_{metric}_mean"] = float(np.mean(scores[f"train_{metric}"]))
    return result


def comparison_display(table: pd.DataFrame) -> pd.DataFrame:
    """Format mean +/- sample standard deviation across folds for beginners."""
    result = pd.DataFrame(index=table["model"])
    for metric in SCORING:
        result[metric] = [f"{mean:.4f} +/- {std:.4f}" for mean, std in
                          zip(table[f"{metric}_mean"], table[f"{metric}_std"], strict=True)]
    return result


def run_baselines() -> pd.DataFrame:
    """Persist training-only baseline scores and a reproducible split manifest."""
    ensure_dirs()
    x_train, x_test, y_train, _ = split_data()
    split = {"random_state": SEED, "train_rows": len(x_train), "test_rows": len(x_test),
             "train_indices": x_train.index.tolist(), "test_indices": x_test.index.tolist(),
             "train_churn_rate": float(y_train.mean())}
    (REPORTS / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    table = pd.DataFrame([
        cv_row("Dummy", DummyClassifier(strategy="most_frequent", random_state=SEED), x_train, y_train),
        cv_row("LogisticRegression", logistic_model(), x_train, y_train),
    ])
    table.to_csv(REPORTS / "baselines.csv", index=False)
    LOGGER.info("Training-only baselines:\n%s", comparison_display(table).to_string())
    return table
