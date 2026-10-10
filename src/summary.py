"""Read final evidence and demonstrate prediction with a genuine training row."""

import logging
from typing import Any

import pandas as pd

from src.config import REPORTS
from src.evaluate import load_evaluation
from src.predict import predict_one
from src.split import split_data

LOGGER = logging.getLogger(__name__)


def results_table() -> pd.DataFrame:
    """Combine baseline CV and final test metrics while labeling their source."""
    baselines = pd.read_csv(REPORTS / "baselines.csv")
    rows = []
    for row in baselines.itertuples():
        rows.append({"model": row.model, "source": "Training 5-fold CV mean",
                     "roc_auc": row.roc_auc_mean, "pr_auc_ap": row.average_precision_mean,
                     "precision": row.precision_mean, "recall": row.recall_mean, "f1": row.f1_mean})
    final = load_evaluation()
    for key in ("default", "cost_selected"):
        m = final[key]
        rows.append({"model": f"{final['selected_model']} threshold={m['threshold']:.2f}",
                     "source": "Held-out test", **{name: m[name] for name in
                     ("roc_auc", "pr_auc_ap", "precision", "recall", "f1")}})
    return pd.DataFrame(rows)


def run_summary() -> dict[str, Any]:
    """Log project results and predict an actual training customer without labels."""
    train, _, _, _ = split_data()
    customer = train.iloc[0].to_dict()
    probability, label = predict_one(customer)
    result = {"training_customer_row": int(train.index[0]), "probability": probability,
              "label": label, "label_meaning": "1=eligible for a retention review, 0=no flag"}
    LOGGER.info("Results:\n%s", results_table().to_string(index=False))
    LOGGER.info("Real-customer prediction demo: %s", result)
    return result
