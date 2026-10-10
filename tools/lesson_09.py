"""Create the final project-summary manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 09 with inference and first-person learning notes."""
    write_notebook(
        ROOT / "notebooks/09_summary.ipynb", "09 — Project summary and prediction",
        "Review the complete workflow and its final evidence. Use the saved model and training-selected threshold to predict one genuine customer, then identify what remains unproven.",
        "A project is useful when another person can repeat it and understand its boundaries. The saved pipeline is like a recipe card: future predictions must use the exact same preparation as training.",
        [
            ("Read the final evidence rather than fitting or retesting. The table explicitly separates training-CV means from held-out test results.",
             "from src.summary import results_table, run_summary\nfrom src.predict import predict_one\nfrom src.split import split_data\ndisplay(results_table())",
             "The minority-class baseline is weak despite high accuracy; the final model ranks churners better, and its cost-selected threshold favors coverage over offer precision."),
            ("Demonstrate predict_one(dict) with a real training customer. Supply original predictor fields; the helper cleans, engineers, transforms with fitted statistics, and applies the saved threshold.",
             "train, _, _, _ = split_data()\ncustomer = train.iloc[0].to_dict()\nprobability, label = predict_one(customer)\ndisplay(pd.Series(customer, name='real_training_customer'))\ndisplay(pd.Series({'churn_probability': probability, 'offer_review_flag': label}))\nrun_summary()",
             "The label is a threshold-based review flag, not certainty that the customer will leave or that an offer will help."),
        ],
        ["Changing preprocessing at inference time.", "Treating one random split as proof of future performance.", "Claiming causal business benefits from churn prediction or SHAP."],
        ["The pipeline and threshold provide reproducible inference.", "The holdout stayed outside feature/model/threshold decisions.", "A temporal validation and retention experiment are logical next steps."],
        "Review README.md, run the reproduction commands, and use the ten-step Git history to study each stage.",
        "## What I learned\n\nI learned that leakage can occur before fitting a model: test-driven EDA or threshold choice can also contaminate an evaluation. I kept learned preprocessing inside training folds and chose the threshold from training out-of-fold predictions.\n\nI learned that imbalance makes accuracy a poor guide: a model that never finds churners can still score about 74%. I used average precision, recall, precision, and a dummy baseline to ask whether the model does something useful.\n\nI learned that threshold choice is a business decision. A missed churner cost of five versus a wasted-offer cost of one favors recall, but those assumptions need real business validation.\n\nI learned that interpretability explains a model, not causation. SHAP can highlight useful hypotheses while correlated features and the background sample affect how credit is divided.\n\n## Limitations and next steps\n\nThis historical snapshot lacks a defined future prediction window, temporal test, retention outcomes, and deployment monitoring. Cleaning removes identical records after dropping IDs, a coursework simplification that may remove distinct customers. Engineered bill ratios are noisy proxies, and correlated fields share importance. The selected CV score and OOF threshold cost reuse training model-selection information. The XGBoost lead over logistic regression is small relative to fold variation, and tuning did not materially improve AP. Validate the label/prediction horizon, test a later time period, measure subgroup quality and calibration, and run a controlled retention experiment before operational use.",
    )


if __name__ == "__main__":
    main()
