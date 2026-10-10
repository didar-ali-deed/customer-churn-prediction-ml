"""Create the model-comparison and tuning manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 06."""
    write_notebook(
        ROOT / "notebooks/06_model_training.ipynb", "06 — Compare and tune models",
        "Compare logistic regression, random forest, and XGBoost with identical training CV. Tune the best by average precision using 30 randomized settings and save its full pipeline.",
        "Model comparison is like trying three tools on the same repair jobs: changing the jobs makes the comparison unfair. Tuning adjusts a tool's settings; repeated trials can make practice scores look better than future results.",
        [
            ("Inspect candidate models and their finite search spaces. Logistic regression uses a linear boundary; random forest averages trees; XGBoost adds trees that correct previous errors.",
             "from src.train import candidate_models, search_space, run_training, comparison_display, plot_fit_gap\ndisplay(pd.Series({name: repr(model) for name, model in candidate_models().items()}, name='candidates'))\ndisplay(pd.Series({name: list(search_space(name)) for name in candidate_models()}, name='searched_parameters'))",
             "All randomized models use seed 42 and the same preprocessing and five folds."),
            ("Select the largest mean training-CV AP, then RandomizedSearchCV tries 30 settings on that candidate. Median/scaler/encoder fitting remains inside every searched fold. The runner and notebook share an artifact fingerprint so a second execution reuses matching results.",
             "comparison, metadata = run_training()\ndisplay(comparison_display(comparison))\ndisplay(pd.Series(metadata['best_params'], name='best_parameters'))\ndisplay(pd.Series({'winner': metadata['selected_model'], 'searched_settings': metadata['search_candidates'], 'tuned_CV_AP': metadata['tuned_cv_ap']}))",
             "Mean +/- SD describes variation across folds, not a confidence interval. The selected CV score is optimistic because the same folds guided model and parameter choice."),
            ("Inspect train-vs-CV average precision for possible overfitting. The tuned model is refit only on all training customers; the test remains reserved.",
             "plot_fit_gap(comparison, show=True)\ndisplay(comparison[['model', 'train_average_precision_mean', 'average_precision_mean', 'average_precision_std']])",
             "A large train/CV gap suggests memorization; weak scores on both suggest limited signal or underfitting. A complex model is not automatically better."),
        ],
        ["Tuning using test AP.", "Assuming the selected CV score is an unbiased final estimate.", "Saving only an estimator and losing its preprocessing recipe."],
        ["Three model families receive identical CV treatment.", "Thirty randomized candidates optimize average precision on training only.", "The saved joblib artifact includes feature creation, preprocessing, and the model."],
        "07_evaluation.ipynb: choose an offer threshold from training out-of-fold predictions, then open the test set once.",
    )


if __name__ == "__main__":
    main()

