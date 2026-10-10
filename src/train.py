"""Training-only baselines, identical-fold comparison, and reproducible tuning."""

import json
import logging
import hashlib
from typing import Any

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from xgboost import XGBClassifier

from src.config import REPORTS, SEED, ROOT, MODELS, PROCESSED, ensure_dirs
from src.eda import save_figure
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


def file_hash(path: Any) -> str:
    """Fingerprint a persisted artifact to reject stale downstream reports."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_fingerprint() -> str:
    """Hash data, relevant source, and dependencies independent of newline convention."""
    digest = hashlib.sha256()
    for path in [PROCESSED, ROOT / "requirements.txt", *[
        ROOT / "src" / name for name in
        ["config.py", "data.py", "features.py", "eda.py", "split.py", "train.py"]
    ]]:
        digest.update(path.name.encode())
        digest.update(path.read_text(encoding="utf-8-sig").encode())
    return digest.hexdigest()


def candidate_models() -> dict[str, Any]:
    """Return reproducible linear, bagged-tree, and boosted-tree candidates."""
    return {
        "LogisticRegression": logistic_model(),
        "RandomForest": RandomForestClassifier(n_estimators=180, max_depth=10,
                                                min_samples_leaf=3, class_weight="balanced",
                                                random_state=SEED, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=180, max_depth=3, learning_rate=0.05,
                                 subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                 tree_method="hist", random_state=SEED, n_jobs=1),
    }


def search_space(name: str) -> dict[str, list[Any]]:
    """Use finite beginner-readable grids; at least 30 combinations for each model."""
    if name == "LogisticRegression":
        return {"model__C": np.logspace(-3, 2, 40).tolist()}
    if name == "RandomForest":
        return {
            "model__n_estimators": [120, 180, 260], "model__max_depth": [5, 8, 12, None],
            "model__min_samples_leaf": [1, 3, 7, 12], "model__max_features": ["sqrt", 0.5, 0.8],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        }
    return {
        "model__n_estimators": [80, 140, 220], "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.02, 0.05, 0.1], "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0], "model__min_child_weight": [1, 3, 7],
        "model__reg_lambda": [1, 5, 10], "model__scale_pos_weight": [1, 2.8],
    }


def plot_fit_gap(table: pd.DataFrame, show: bool = False) -> None:
    """Compare training and validation AP to highlight possible overfitting."""
    fig, ax = plt.subplots(figsize=(10, 4))
    positions = np.arange(len(table))
    ax.bar(positions - 0.18, table["train_average_precision_mean"], width=0.36, label="Train AP")
    ax.bar(positions + 0.18, table["average_precision_mean"], width=0.36,
           yerr=table["average_precision_std"], label="CV AP (+/- fold SD)")
    ax.set(xticks=positions, xticklabels=table["model"], ylabel="Average precision",
           ylim=(0, 1), title="Train vs CV: a large gap can indicate overfitting")
    ax.legend()
    save_figure(fig, "model_fit_gap", show)


def load_training_metadata() -> dict[str, Any]:
    """Require a matching model artifact and source/data recipe."""
    path = REPORTS / "training.json"
    if not path.exists() or not (MODELS / "model.joblib").exists():
        raise FileNotFoundError("Run scripts/06_train.py first.")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta["fingerprint"] != training_fingerprint() or meta["model_hash"] != file_hash(MODELS / "model.joblib"):
        raise ValueError("Training artifacts are stale; rerun scripts/06_train.py.")
    return meta


def run_training(force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare, tune 30 candidates by AP, refit on train, and persist the pipeline.

    Repeated notebook/script execution reuses the same matching fitted artifact.
    Selection scores reuse CV folds and are optimistic, not an unbiased nested-CV estimate.
    """
    ensure_dirs()
    if not force:
        try:
            meta = load_training_metadata()
            table = pd.read_csv(REPORTS / "model_comparison.csv")
            LOGGER.info("Reusing matching training artifact: %s", meta["selected_model"])
            return table, meta
        except (FileNotFoundError, ValueError, KeyError):
            pass
    x_train, _, y_train, _ = split_data()
    candidates = candidate_models()
    rows = []
    for name, model in candidates.items():
        LOGGER.info("Comparing %s on five training folds", name)
        rows.append(cv_row(name, model, x_train, y_train))
    table = pd.DataFrame(rows)
    winner = str(table.loc[table["average_precision_mean"].idxmax(), "model"])
    LOGGER.info("Tuning %s: 30 randomized candidates, 5 folds, scoring=average_precision", winner)
    search = RandomizedSearchCV(
        make_pipeline(candidates[winner]), search_space(winner), n_iter=30,
        scoring="average_precision", cv=cv_folds(), random_state=SEED,
        n_jobs=2, refit=True, return_train_score=True, error_score="raise",
    )
    search.fit(x_train, y_train)
    best = search.best_estimator_
    tuned_row = cv_row(f"{winner} (tuned)", clone(best.named_steps["model"]), x_train, y_train)
    table = pd.concat([table, pd.DataFrame([tuned_row])], ignore_index=True)
    table.to_csv(REPORTS / "model_comparison.csv", index=False)
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(REPORTS / "search_results.csv", index=False)
    joblib.dump(best, MODELS / "model.joblib")
    meta = {"fingerprint": training_fingerprint(), "model_hash": file_hash(MODELS / "model.joblib"),
            "selected_model": winner, "best_params": search.best_params_,
            "search_cv_ap": float(search.best_score_), "tuned_cv_ap": tuned_row["average_precision_mean"],
            "tuned_cv_ap_std": tuned_row["average_precision_std"],
            "search_candidates": len(results), "random_state": SEED, "train_rows": len(x_train),
            "score_note": "CV folds reused for selection; unbiased final estimate comes from held-out test."}
    (REPORTS / "training.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    plot_fit_gap(table)
    LOGGER.info("Model comparison:\n%s", comparison_display(table).to_string())
    LOGGER.info("Best training recipe: %s", meta)
    return table, meta
