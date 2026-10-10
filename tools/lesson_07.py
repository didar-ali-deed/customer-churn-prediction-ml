"""Create the final-evaluation manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 07, emphasizing training-only threshold selection."""
    write_notebook(
        ROOT / "notebooks/07_evaluation.ipynb", "07 — Threshold and final evaluation",
        "Choose a retention threshold with an explicit cost: missed churner=5, wasted offer=1. Lock that choice using training out-of-fold predictions, then assess the test set once and save the result.",
        "A probability ranks risk; a threshold decides whom to call. It is like choosing how sensitive a smoke alarm should be: missing a fire is more costly than a false alarm, but too many false alarms also have a cost.",
        [
            ("The script has already locked a threshold before making its sole held-out prediction. This notebook reads the matching saved report; on a first run, the same function performs that workflow. No fitting ever receives test rows.",
             "from src.evaluate import run_evaluation, plot_threshold, plot_evaluation\nfrom src.config import REPORTS\nresult = run_evaluation()\ndisplay(pd.DataFrame([result['default'], result['cost_selected']], index=['default_0.50', 'training_cost_selected']))",
             "ROC-AUC and AP stay the same across thresholds because ranking is unchanged; precision, recall, F1, and business cost depend on the decision rule."),
            ("Scan np.linspace(0.05, 0.95, 91) on out-of-fold training predictions. Each customer's score comes from a pipeline that did not fit on that row. Lower thresholds generally catch more churners and send more unnecessary offers.",
             "scan = pd.read_csv(REPORTS / 'threshold_scan.csv')\nthreshold = result['cost_selected']['threshold']\nplot_threshold(scan, threshold, show=True)\ndisplay(scan.nsmallest(5, 'cost'))",
             "The selected threshold minimizes 5*FN+FP on training OOF predictions; costs assume equal value per customer and do not prove an offer prevents churn. Hyperparameter selection still makes this OOF cost somewhat optimistic."),
            ("Read the saved test predictions for confusion, ranking, and calibration plots; this produces no new test inference. Calibration asks whether a predicted 30% risk corresponds to roughly 30% observed churn.",
             "predictions = pd.read_csv(REPORTS / 'test_predictions.csv')\nplot_evaluation(predictions, threshold, show=True)",
             "The confusion matrix shows the offer tradeoff; ROC and PR summarize ranking; departures from the calibration diagonal suggest probabilities need future validation, not recalibration on this test set."),
            ("Compare training CV AP with the single test AP. Fold SD describes variation, not a formal uncertainty interval for this difference.",
             "display(pd.Series({'tuned_CV_AP': result['tuned_cv_ap'], 'CV_fold_SD': result['tuned_cv_ap_std'], 'test_AP': result['cost_selected']['pr_auc_ap'], 'test_minus_CV_AP': result['test_minus_cv_ap']}))",
             "A gap can reflect sampling variation and selection optimism; one random split cannot establish stable performance. No model or threshold is changed after seeing this result."),
        ],
        ["Selecting the cheapest threshold on test labels.", "Confusing average precision with trapezoidal PR-curve area.", "Assuming ranking quality guarantees calibrated probabilities or retention impact."],
        ["The offer threshold is selected entirely from training predictions.", "Held-out predictions are made once and persisted for repeatable notebook execution.", "Cost and calibration complement ranking metrics."],
        "08_explainability.ipynb: explain the already-frozen model using training customers.",
    )


if __name__ == "__main__":
    main()
