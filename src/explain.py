"""SHAP explanations of the frozen pipeline using training customers only."""

import json
import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config import MODELS, REPORTS, SEED, ensure_dirs
from src.eda import save_figure
from src.features import NUMERIC, CATEGORICAL
from src.split import split_data
from src.train import load_training_metadata

LOGGER = logging.getLogger(__name__)


def original_feature(encoded_name: str) -> str:
    """Map a one-hot column back to its business field for grouped importance."""
    short = encoded_name.split("__", 1)[-1]
    for name in sorted(NUMERIC + CATEGORICAL, key=len, reverse=True):
        if short == name or short.startswith(name + "_"):
            return name
    return short


def recommendation(feature: str) -> str:
    """Suggest a testable business response without claiming a causal effect."""
    if feature in ("tenure", "tenure_bucket"):
        return "Pilot stronger onboarding and early check-ins; measure retention with a controlled experiment."
    if feature == "Contract":
        return "Offer clear, optional longer-term plan choices; test whether their value improves retention."
    if feature in ("MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_ratio"):
        return "Review bill clarity and plan value with customers; test targeted bill-review support before offering discounts."
    if feature in ("InternetService", "PhoneService", "MultipleLines"):
        return "Investigate service experience and reliability by plan; validate issues with service-quality data."
    if feature in ("PaymentMethod", "is_autopay", "PaperlessBilling"):
        return "Offer convenient, optional payment reminders or autopay education; test uptake and retention."
    if feature in ("gender", "SeniorCitizen"):
        return "Audit subgroup performance and fairness; do not assume demographic membership calls for different treatment."
    if feature in ("Partner", "Dependents"):
        return "Research different household service needs without assuming family status causes churn."
    return "Pilot clearer support and bundle-value communication; measure whether customer experience improves."


def explain_model() -> tuple[shap.Explanation, pd.DataFrame, dict[str, Any]]:
    """Compute SHAP and verify additive consistency for a fixed training sample."""
    meta = load_training_metadata()
    pipeline = joblib.load(MODELS / "model.joblib")
    train, _, _, _ = split_data()
    sample = train.sample(n=min(200, len(train)), random_state=SEED)
    background_rows = train.sample(n=min(100, len(train)), random_state=SEED)
    transformed = np.asarray(pipeline[:-1].transform(sample))
    background = np.asarray(pipeline[:-1].transform(background_rows))
    names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        raw = shap.LinearExplainer(model, background)(transformed)
        reference_output = model.decision_function(transformed)
        output_units = "log-odds"
    else:
        raw = shap.TreeExplainer(model, data=background,
                                 feature_perturbation="interventional",
                                 model_output="raw")(transformed)
        if isinstance(model, RandomForestClassifier):
            reference_output = model.predict_proba(transformed)[:, 1]
            output_units = "churn probability"
        else:
            reference_output = model.predict(transformed, output_margin=True)
            output_units = "log-odds"
    values = np.asarray(raw.values)
    base = np.asarray(raw.base_values)
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if base.ndim == 2 else base[1]
    if base.ndim == 0:
        base = np.full(len(sample), float(base))
    explanation = shap.Explanation(values=values, base_values=base, data=transformed,
                                    feature_names=names)
    residual = float(np.max(np.abs(base + values.sum(axis=1) - reference_output)))
    if residual > 1e-4:
        raise ValueError(f"SHAP does not reconstruct the model output: residual={residual}")
    probabilities = pipeline.predict_proba(sample)[:, 1]
    high = int(np.argmax(probabilities))
    encoded = pd.DataFrame({"encoded_feature": names,
                            "mean_abs_shap": np.abs(values).mean(axis=0)})
    encoded["feature"] = encoded["encoded_feature"].map(original_feature)
    drivers = encoded.groupby("feature", as_index=False)["mean_abs_shap"].sum().sort_values(
        "mean_abs_shap", ascending=False
    ).reset_index(drop=True)
    drivers["recommendation"] = drivers["feature"].map(recommendation)
    info = {"model_hash": meta["model_hash"], "training_sample_size": len(sample),
            "background_size": len(background), "output_units": output_units,
            "max_additivity_error": residual, "local_sample_position": high,
            "local_training_row_index": int(sample.index[high]),
            "local_churn_probability": float(probabilities[high]),
            "local_baseline_output": float(base[high]),
            "local_final_output": float(reference_output[high]),
            "global_importance_definition": "Sum mean absolute SHAP over encoded columns per business field; magnitude is not direction.",
            "caveat": "Training-only sample; correlated features share credit; interventional background may create unrealistic combinations; explanation is not causation."}
    return explanation, drivers, info


def run_explanations(show: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Save global summary, magnitude bar, local waterfall, and recommendations."""
    ensure_dirs()
    explanation, drivers, info = explain_model()
    # SHAP's beeswarm uses NumPy's global RNG for visual jitter (no seed argument).
    np.random.seed(SEED)
    shap.plots.beeswarm(explanation, max_display=12, show=False)
    plt.title(f"Training SHAP summary ({info['output_units']})")
    save_figure(plt.gcf(), "shap_summary", show)
    shap.plots.bar(explanation, max_display=12, show=False)
    plt.title("Mean absolute SHAP: importance magnitude, not direction")
    save_figure(plt.gcf(), "shap_bar", show)
    high = info["local_sample_position"]
    shap.plots.waterfall(explanation[high], max_display=12, show=False)
    save_figure(plt.gcf(), "shap_local", show)
    local = pd.DataFrame({"encoded_feature": explanation.feature_names,
                          "transformed_value": explanation.data[high],
                          "shap_contribution": explanation.values[high]})
    local["absolute_contribution"] = local["shap_contribution"].abs()
    local.sort_values("absolute_contribution", ascending=False).to_csv(REPORTS / "local_explanation.csv", index=False)
    drivers.to_csv(REPORTS / "shap_drivers.csv", index=False)
    (REPORTS / "explainability.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (REPORTS / "business_recommendations.md").write_text(
        "# Five model-informed hypotheses to test\n\n" + "\n".join(
            f"- **{row.feature}** (importance {row.mean_abs_shap:.3f}): {row.recommendation}"
            for row in drivers.head(5).itertuples()
        ) + "\n\nSHAP explains the frozen model's associations; these recommendations require experiments, not causal assumptions.\n",
        encoding="utf-8",
    )
    LOGGER.info("Top business drivers:\n%s", drivers.head(5).to_string(index=False))
    LOGGER.info("Local explanation: %s", info)
    return drivers, info
