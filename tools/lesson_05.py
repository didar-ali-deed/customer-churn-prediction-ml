"""Create the split-and-baseline manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 05."""
    write_notebook(
        ROOT / "notebooks/05_split_and_baseline.ipynb", "05 — Split and baseline",
        "Understand the fixed stratified 80/20 split used since EDA. Measure two reference models using five-fold CV on training customers only.",
        "Stratification is like dealing cards so both hands contain a similar share of rare cards. CV rotates practice exams inside the training set; the held-out test is a sealed final exam.",
        [
            ("Use the shared split with random_state=42. Stratification preserves the minority-class share; 5,616 customers train and 1,405 remain held out.",
             "from src.split import split_data\nfrom src.train import run_baselines, comparison_display\nx_train, x_test, y_train, y_test = split_data()\ndisplay(pd.Series({'train_rows': len(x_train), 'test_rows': len(x_test), 'train_churn_share': y_train.mean(), 'index_overlap': len(set(x_train.index) & set(x_test.index))}))",
             "The partitions do not overlap. The test labels are reserved for notebook 07 and are never passed to fit or tuning."),
            ("Both models include feature creation and preprocessing inside a Pipeline. The dummy always predicts the majority; balanced logistic regression gives minority errors more weight. CV fits each preprocessor on only four folds, then scores on the fifth.",
             "baselines = run_baselines()\ndisplay(comparison_display(baselines))",
             "ROC-AUC measures ranking across both classes; PR-AUC here means average precision, which focuses on finding churners. Recall measures coverage, precision offer efficiency, and F1 their harmonic balance."),
            ("Show why accuracy is misleading. A model can win many easy majority cases while failing its actual business purpose.",
             "dummy = baselines.loc[baselines['model'].eq('Dummy')].iloc[0]\ndisplay(pd.Series({'dummy_CV_accuracy': dummy['accuracy_mean'], 'dummy_CV_recall': dummy['recall_mean'], 'dummy_CV_PR_AUC_AP': dummy['average_precision_mean']}))",
             "The dummy has about 73.6% accuracy but zero churn recall. Its average precision equals churn prevalence; all customers receive the same zero risk score."),
        ],
        ["Comparing models using different folds.", "Fitting encoding before CV.", "Treating high majority-class accuracy as churn detection success."],
        ["The 20% test set is untouched by model decisions.", "All preprocessing is fit inside training folds.", "The logistic baseline must improve on a meaningful dummy reference."],
        "06_model_training.ipynb: compare three models using identical folds and tune the best by average precision.",
    )


if __name__ == "__main__":
    main()
