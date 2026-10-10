"""Create the feature-engineering manuscript."""

from src.config import ROOT
from src.features import ENGINEERED, RATIONALE
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 04 with a separate plot for each business feature."""
    chunks = [
        ("Create row-local features using pandas and numpy; the target is never an input.",
         "from src.split import training_frame\nfrom src.features import add_features, make_preprocessor, plot_feature, run_features\ndf = training_frame()\nengineered = add_features(df)\ndisplay(engineered.head())",
         "Each customer gains six business representations; no population statistics were learned."),
    ]
    for column in ENGINEERED:
        chunks.append((f"**{column}:** {RATIONALE[column]} Compare churn rates using training rows.",
                       f"display(plot_feature(df, '{column}', show=True))",
                       "Differences suggest possible predictive value; correlated features may add little beyond the original columns. CV, rather than this plot, decides usefulness."))
    chunks.append(("Median imputation fills unknown numeric values; scaling puts numeric values on comparable units. One-hot encoding makes category indicator columns and ignores unseen categories. Fit this demonstration on training only; real model CV creates a fresh preprocessor inside each fold.",
                   "pre = make_preprocessor()\ntransformed = pre.fit_transform(engineered)\ndisplay(pd.Series({'rows': transformed.shape[0], 'columns': transformed.shape[1], 'finite': bool(np.isfinite(transformed).all())}))\ndisplay(pd.Series(pre.get_feature_names_out(), name='encoded_columns'))\nrun_features()",
                   "The fitted transformer yields a finite matrix with a fixed output schema for future rows. Different training folds can discover different category vocabularies; stability means transform does not change a fitted schema."))
    write_notebook(
        ROOT / "notebooks/04_feature_engineering.ipynb", "04 — Features and preprocessing",
        "Express six business hypotheses as deterministic features. Build an unfitted preprocessor suitable for honest cross-validation and future customers.",
        "Features are like translating a story into clues a model can read. A pipeline is a sealed recipe: each training fold prepares its own ingredients rather than borrowing knowledge from validation customers.",
        chunks,
        ["Fitting a scaler or median on test data.", "Dividing by zero tenure.", "Assuming feature churn plots establish incremental value or causality."],
        ["Six features have explicit business meanings and zero-denominator rules.", "The target and ID are excluded from model inputs.", "Imputation, scaling, and encoding will stay inside each CV pipeline."],
        "05_split_and_baseline.ipynb: establish a simple, honest reference score.",
    )


if __name__ == "__main__":
    main()
