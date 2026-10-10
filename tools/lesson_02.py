"""Create the cleaning lesson manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 02."""
    write_notebook(
        ROOT / "notebooks/02_data_cleaning.ipynb", "02 — Data cleaning",
        "Translate audit findings into reproducible, tested cleaning rules. Save a clean table while preserving the raw CSV.",
        "Cleaning is like standardizing addresses before delivering mail: ' No ' and 'No' should mean the same thing. A rule based on the current row is safe before splitting; a learned median is not and belongs inside training folds.",
        [
            ("Inspect the problematic rows before changing them.",
             "from src.data import load_raw, run_cleaning\nraw = load_raw()\ndisplay(raw.loc[pd.to_numeric(raw['TotalCharges'], errors='coerce').isna(), ['tenure', 'TotalCharges']])",
             "All 11 blanks occur at tenure zero, supporting a cumulative-charge value of zero."),
            ("Run the shared cleaning function. It strips whitespace, removes customerID, converts charges, normalizes service labels, maps Churn, and drops duplicate cleaned records.",
             "df, comparison = run_cleaning()\ndisplay(comparison)\ndisplay(pd.Series({'raw_rows': len(raw), 'clean_rows': len(df), 'remaining_nulls': int(df.isna().sum().sum())}))",
             "The dtype table shows numeric charges and a numeric target; 22 identical cleaned records are removed, leaving 7,021 rows."),
            ("Verify the conditional rule and remaining categories. np.where changes only missing charges at tenure zero; unknown bills at positive tenure remain missing for fold-local median imputation.",
             "display(df.loc[df['tenure'].eq(0), ['tenure', 'TotalCharges']])\ndisplay(df[['MultipleLines', 'OnlineSecurity', 'TechSupport']].apply(lambda col: ', '.join(sorted(col.unique()))).to_frame('values'))\ndisplay(df['Churn'].value_counts().to_frame('customers'))",
             "No parent service means no add-on, while InternetService itself still identifies customers without internet."),
        ],
        ["Replacing every invalid charge with zero regardless of tenure.", "Learning imputation medians from the full dataset.", "Dropping duplicate feature rows with different targets as though they were identical."],
        ["The raw file remains unchanged.", "Row-local rules are explicit and covered by tests.", "No statistical transformer has been fit yet."],
        "03_eda.ipynb: split first and explore only training customers. Deduplication after dropping IDs is a coursework choice: distinct people can share identical attributes; production work should use entity-level deduplication instead.",
    )


if __name__ == "__main__":
    main()

