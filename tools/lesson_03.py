"""Create the EDA manuscript with measured final findings."""

from src.config import ROOT
from src.eda import findings
from src.split import training_frame
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 03 with five data-backed findings."""
    notes = findings(training_frame())
    write_notebook(
        ROOT / "notebooks/03_eda.ipynb", "03 — Exploratory data analysis",
        "Explore how churn relates to customer attributes. Reserve the test split before looking for modeling ideas; every chart below uses only training rows.",
        "EDA is like scouting a route before a trip. The test set is the final exam: reading its answers while preparing would give an unfair estimate of how well we learned.",
        [
            ("The fixed 80/20 stratified split is introduced now to protect the test set; notebook 05 explains it fully. No statistical preprocessor is fit.",
             "from src.split import training_frame\nfrom src.eda import plot_target, plot_rates, plot_numeric, plot_outliers, plot_correlation, churn_rates, tenure_bucket, run_eda\ndf = training_frame()\ndisplay(pd.Series({'training_rows': len(df), 'training_churn_rate': df['Churn'].mean()}))\nplot_target(df, show=True)",
             "The training subset still contains a minority of churners, motivating recall and PR-AUC."),
            ("Plot churn rates and inspect counts: a rate from a tiny group is less reliable.",
             "plot_rates(df, show=True)\nfor column in ['Contract', 'InternetService', 'PaymentMethod']:\n    display(churn_rates(df, column))\ndisplay(churn_rates(df.assign(tenure_bucket=tenure_bucket(df['tenure'])), 'tenure_bucket'))",
             "Contract, service, payment, and tenure groups have different observed churn rates; these are candidate predictors."),
            ("Separate numeric distributions by churn. Density normalization compares shapes even though the groups differ in size.",
             "plot_numeric(df, show=True)",
             "Tenure and charge distributions differ across outcomes; CV must establish whether these patterns generalize."),
            ("Inspect boxplots before deciding whether extreme values are data errors.",
             "plot_outliers(df, show=True)",
             "High bills can be legitimate; no automatic clipping or customer removal is justified."),
            ("Correlations reveal redundancy among numeric fields; they do not measure causality or categorical effects.",
             "plot_correlation(df, show=True)\nnotes = run_eda()\ndisplay(pd.Series(notes, name='Measured findings'))",
             "TotalCharges and tenure are related; later charge ratios add another representation, with redundancy acknowledged."),
        ],
        ["Exploring test labels before choosing features.", "Treating a group association as proof of causation.", "Deleting every point beyond a boxplot whisker."],
        ["All modeling EDA uses training rows.", "Group counts give context to churn rates.", "EDA hypotheses must survive CV."],
        "04_feature_engineering.ipynb: encode business hypotheses with fold-safe preprocessing.",
        "## Five data-backed findings\n\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(notes, 1)),
    )


if __name__ == "__main__":
    main()
