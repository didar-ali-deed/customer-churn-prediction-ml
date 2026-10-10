"""Create the SHAP and business-insights manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 08."""
    write_notebook(
        ROOT / "notebooks/08_explainability.ipynb", "08 — Explainability and business insights",
        "Explain the frozen model globally and for one high-risk training customer. Turn the top five business fields into hypotheses that the business can test.",
        "SHAP is like splitting credit among teammates: it measures each feature's contribution relative to a background. Correlated teammates can share credit in different ways, so attribution explains this model rather than proving what causes churn.",
        [
            ("Use a seed-42 sample of 200 training customers and a 100-row training background. SHAP receives the same encoded matrix used by the estimator, preserving the saved preprocessing.",
             "from src.explain import run_explanations\nfrom src.config import REPORTS\ndrivers, info = run_explanations(show=True)\ndisplay(pd.Series(info))",
             "The beeswarm shows signed effects and value patterns; the bar shows average magnitude; the waterfall starts at the background output and adds feature contributions for the highest-risk customer in this sample."),
            ("Inspect global drivers grouped back to business fields. For one-hot columns we sum their mean absolute SHAP magnitudes; global magnitude does not tell us whether a field increases risk.",
             "display(drivers.head(10))\ndisplay(pd.read_csv(REPORTS / 'local_explanation.csv').head(10))",
             "Positive local contributions increase the model output for this customer, and negative ones lower it. Numeric values in the local table are standardized; they are not raw bills or tenure."),
            ("Translate the five largest business-field attributions into practical, testable actions. Log-odds contributions add before conversion to probability; tree probability explanations, when applicable, use probability units.",
             "display(drivers.head(5)[['feature', 'recommendation']])\ndisplay(pd.Series({'customer_row': info['local_training_row_index'], 'predicted_churn_probability': info['local_churn_probability'], 'SHAP_units': info['output_units'], 'max_reconstruction_error': info['max_additivity_error']}))",
             "The high-risk explanation reconstructs the frozen model output. Recommendations are experiment proposals; SHAP alone does not show that changing a field prevents churn."),
        ],
        ["Labeling log-odds contributions as probability percentage points.", "Using held-out labels to decide which customer story looks persuasive.", "Treating SHAP rankings as a causal retention playbook."],
        ["Global plots explain patterns across a training sample.", "The local explanation is checked for additive consistency.", "Business actions require validation beyond predictive associations."],
        "09_summary.ipynb: use predict_one and review results, limitations, and reproducibility.",
    )


if __name__ == "__main__":
    main()
