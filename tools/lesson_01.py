"""Create the data-audit lesson manuscript."""

from src.config import ROOT
from tools.notebook_builder import write_notebook


def main() -> None:
    """Generate notebook 01 without executing it."""
    write_notebook(
        ROOT / "notebooks/01_setup_and_audit.ipynb", "01 — Setup and data audit",
        "Inspect the real CSV before changing it. Find hidden missing values and understand what one row and the target mean.",
        "An audit is like checking ingredients before cooking: a wrong ingredient can spoil every later step. A dtype describes how Python stores a column, not necessarily what that column means.",
        [
            ("Load with pd.read_csv through src, then inspect a few rows and the shape.",
             "from src.data import load_raw, audit\ndf = load_raw()\nreport = audit(df)\ndisplay(df.head())\ndisplay(pd.Series(report['shape'], index=['rows', 'columns']))",
             "Each of the 7,043 rows represents a customer, with 21 columns including the target and ID."),
            ("Compare dtype, missing values, and unique counts. A value can look empty yet still be a string.",
             "display(pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing': df.isna().sum(), 'unique': df.nunique(dropna=False)}))\ndisplay(pd.Series({'duplicate rows': report['duplicate_rows'], 'duplicate IDs': report['duplicate_ids']}))",
             "TotalCharges is stored as text (object/string depending on pandas version); isna alone misses whitespace blanks."),
            ("Coercion is a diagnostic here: identify text that cannot be interpreted as a number, without changing the input.",
             "charges = pd.to_numeric(df['TotalCharges'], errors='coerce')\ndisplay(df.loc[charges.isna(), ['tenure', 'MonthlyCharges', 'TotalCharges']])\ndisplay(pd.Series({k: report[k] for k in ['total_charges_blank', 'total_charges_numeric_missing']}))",
             "There are 11 blank charges, all at tenure zero: new customers have not accumulated a bill."),
            ("Count both labels and plot their share to understand imbalance.",
             "display(df['Churn'].value_counts().to_frame('customers'))\ndisplay(df['Churn'].value_counts(normalize=True).to_frame('share'))\ndf['Churn'].value_counts().plot.bar(title='Raw target balance', ylabel='Customers')\nplt.tight_layout()\nplt.show()",
             "Roughly 26.5% churn; predicting No for everyone can look accurate while missing every churner."),
        ],
        ["Treating whitespace as a real charge.", "Assuming a text dtype means a categorical feature.", "Training before checking the target and ID."],
        ["The raw dataset has 7,043 customers and 21 columns.", "Eleven blank TotalCharges entries need an explicit rule.", "Churn is imbalanced, so accuracy alone will be insufficient."],
        "02_data_cleaning.ipynb: turn the audited rules into tested functions.",
    )


if __name__ == "__main__":
    main()
