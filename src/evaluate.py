"""Training-only threshold choice and one cached held-out assessment."""

import hashlib
import json
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_predict

from src.config import MODELS, REPORTS, ensure_dirs
from src.eda import save_figure
from src.split import split_data
from src.train import cv_folds, load_training_metadata

LOGGER = logging.getLogger(__name__)


def cost_counts(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    """Charge five units for a missed churner and one for an unnecessary offer."""
    prediction = probabilities >= threshold
    false_negative = int(np.sum((y == 1) & ~prediction))
    false_positive = int(np.sum((y == 0) & prediction))
    return {"false_negative": false_negative, "false_positive": false_positive,
            "cost": 5 * false_negative + false_positive}


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Scan 91 thresholds on training OOF predictions; ties prefer the lower one."""
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    if len(y) != len(probabilities) or not len(y) or not np.isfinite(probabilities).all():
        raise ValueError("Threshold inputs must be nonempty, aligned, and finite.")
    if not np.isin(y, [0, 1]).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Use binary labels and probabilities between zero and one.")
    table = pd.DataFrame([
        {"threshold": float(t), **cost_counts(y, probabilities, float(t))}
        for t in np.linspace(0.05, 0.95, 91)
    ])
    table["cost_per_customer"] = table["cost"] / len(y)
    chosen = float(table.loc[table["cost"].idxmin(), "threshold"])
    return chosen, table


def metric_row(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Separate probability-ranking metrics from threshold-dependent decisions."""
    prediction = probabilities >= threshold
    return {"threshold": threshold, "roc_auc": float(roc_auc_score(y, probabilities)),
            "pr_auc_ap": float(average_precision_score(y, probabilities)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "brier_score": float(brier_score_loss(y, probabilities)),
            **cost_counts(y, probabilities, threshold)}


def plot_threshold(table: pd.DataFrame, threshold: float, show: bool = False) -> None:
    """Plot offer cost estimated on training OOF predictions, never test labels."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(table["threshold"], table["cost_per_customer"])
    ax.axvline(threshold, color="red", linestyle="--", label=f"Chosen {threshold:.2f}")
    ax.set(xlabel="Probability threshold", ylabel="Cost units per customer",
           title="Training out-of-fold cost: missed churner=5, wasted offer=1")
    ax.legend()
    save_figure(fig, "threshold_cost", show)


def plot_evaluation(predictions: pd.DataFrame, threshold: float, show: bool = False) -> None:
    """Draw four held-out diagnostics from persisted one-time predictions."""
    y = predictions["Churn"].to_numpy()
    p = predictions["probability"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y, p >= threshold),
                           display_labels=["Stayed", "Churned"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Test confusion matrix (fixed training threshold {threshold:.2f})")
    save_figure(fig, "test_confusion_matrix", show)
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="Recall", title="Test ROC curve")
    ax.legend()
    save_figure(fig, "test_roc", show)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AP={average_precision_score(y, p):.3f}")
    ax.axhline(y.mean(), linestyle="--", color="gray", label="Churn prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    ax.legend()
    save_figure(fig, "test_pr", show)
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(predicted, observed, "o-", label="Observed churn")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed churn fraction",
           title="Test calibration (diagnostic only; no test-set recalibration)")
    ax.legend()
    save_figure(fig, "test_calibration", show)


def evaluation_signature(meta: dict[str, Any]) -> str:
    """Tie the assessment to its model, split recipe, and stated cost assumptions."""
    return hashlib.sha256((meta["fingerprint"] + meta["model_hash"]
                           + "seed42_cost5_1_grid0.05_0.95_91").encode()).hexdigest()


def load_evaluation() -> dict[str, Any]:
    """Read a matching final assessment without generating another prediction."""
    meta = load_training_metadata()
    path = REPORTS / "evaluation.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/07_evaluate.py first.")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["signature"] != evaluation_signature(meta):
        raise ValueError("Saved evaluation belongs to a different training recipe/model.")
    return result


def run_evaluation(show: bool = False) -> dict[str, Any]:
    """Select threshold before predicting test once; reruns read saved results.

    OOF predictions avoid in-sample threshold fitting. Hyperparameters were
    selected using this training set, so OOF cost is still selection-biased.
    """
    ensure_dirs()
    meta = load_training_metadata()
    try:
        result = load_evaluation()
        LOGGER.info("Reusing the matching one-time held-out assessment.")
    except (FileNotFoundError, ValueError):
        pipeline = joblib.load(MODELS / "model.joblib")
        x_train, x_test, y_train, y_test = split_data()
        LOGGER.info("Choosing threshold using training out-of-fold predictions only.")
        oof = cross_val_predict(clone(pipeline), x_train, y_train, cv=cv_folds(),
                                method="predict_proba", n_jobs=2)[:, 1]
        threshold, costs = choose_threshold(y_train.to_numpy(), oof)
        costs.to_csv(REPORTS / "threshold_scan.csv", index=False)
        pd.DataFrame({"row_index": x_train.index, "Churn": y_train.to_numpy(),
                      "probability": oof}).to_csv(REPORTS / "train_oof_predictions.csv", index=False)
        chosen_cost = costs.loc[costs["threshold"].eq(threshold)].iloc[0]
        threshold_record = {
            "threshold": threshold, "source": "training_out_of_fold_predictions",
            "missed_churner_cost": 5, "wasted_offer_cost": 1,
            "oof_cost_per_customer": float(chosen_cost["cost_per_customer"]),
            "model_hash": meta["model_hash"], "signature": evaluation_signature(meta),
            "selection_caveat": "OOF uses hyperparameters selected on training CV; cost is not unbiased nested-CV cost.",
        }
        (REPORTS / "threshold.json").write_text(json.dumps(threshold_record, indent=2), encoding="utf-8")
        LOGGER.info("Threshold locked at %.2f before any test prediction.", threshold)
        p = pipeline.predict_proba(x_test)[:, 1]
        predictions = pd.DataFrame({"row_index": x_test.index, "Churn": y_test.to_numpy(),
                                    "probability": p})
        predictions.to_csv(REPORTS / "test_predictions.csv", index=False)
        result = {
            "signature": evaluation_signature(meta), "selected_model": meta["selected_model"],
            "test_rows": len(x_test), "default": metric_row(y_test.to_numpy(), p, 0.5),
            "cost_selected": metric_row(y_test.to_numpy(), p, threshold),
            "tuned_cv_ap": meta["tuned_cv_ap"], "tuned_cv_ap_std": meta["tuned_cv_ap_std"],
            "test_minus_cv_ap": float(average_precision_score(y_test, p) - meta["tuned_cv_ap"]),
        }
        (REPORTS / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    threshold = result["cost_selected"]["threshold"]
    plot_threshold(pd.read_csv(REPORTS / "threshold_scan.csv"), threshold, show)
    plot_evaluation(pd.read_csv(REPORTS / "test_predictions.csv"), threshold, show)
    LOGGER.info("Final held-out results:\n%s", json.dumps(result, indent=2))
    return result
