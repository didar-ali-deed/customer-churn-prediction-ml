"""Training-only exploratory tables, plots, and measured findings."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src.config import FIGURES, REPORTS, ensure_dirs
from src.split import training_frame

LOGGER = logging.getLogger(__name__)


def tenure_bucket(tenure: pd.Series) -> pd.Series:
    """Apply fixed business bins without learning from any customer."""
    return pd.cut(tenure, bins=[-np.inf, 12, 24, 48, np.inf],
                  labels=["0-12", "13-24", "25-48", "49+"]).astype("str")


def churn_rates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Show both group size and churn rate to avoid overreading small groups."""
    return df.groupby(column, observed=True)["Churn"].agg(
        customers="size", churn_rate="mean"
    ).sort_values("churn_rate", ascending=False)


def save_figure(fig: Figure, name: str, show: bool = False) -> None:
    """Persist a readable plot and optionally display it in a notebook."""
    ensure_dirs()
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_target(df: pd.DataFrame, show: bool = False) -> None:
    """Plot the training target count."""
    fig, ax = plt.subplots(figsize=(6, 4))
    df["Churn"].value_counts().sort_index().rename(index={0: "Stayed", 1: "Churned"}).plot.bar(ax=ax)
    ax.set(title="Training target distribution", ylabel="Customers")
    save_figure(fig, "eda_target", show)


def plot_rates(df: pd.DataFrame, show: bool = False) -> None:
    """Compare categorical and fixed-tenure-group churn rates."""
    work = df.assign(tenure_bucket=tenure_bucket(df["tenure"]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, column in zip(axes.ravel(), ["Contract", "InternetService", "PaymentMethod", "tenure_bucket"], strict=True):
        rates = churn_rates(work, column)
        rates["churn_rate"].plot.barh(ax=ax)
        ax.set(title=f"Churn rate by {column}", xlabel="Churn fraction", xlim=(0, 1))
    save_figure(fig, "eda_rates", show)


def plot_numeric(df: pd.DataFrame, show: bool = False) -> None:
    """Show numeric distributions split by the observed outcome."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"], strict=True):
        sns.histplot(data=df, x=column, hue="Churn", bins=25, stat="density",
                     common_norm=False, element="step", ax=ax)
        ax.set_title(column)
    save_figure(fig, "eda_histograms", show)


def plot_outliers(df: pd.DataFrame, show: bool = False) -> None:
    """Inspect extremes without automatically treating them as errors."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"], strict=True):
        sns.boxplot(data=df, x="Churn", y=column, ax=ax)
    save_figure(fig, "eda_boxplots", show)


def plot_correlation(df: pd.DataFrame, show: bool = False) -> None:
    """Plot numeric correlations, including target associations on training only."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Training numeric correlations (association, not causation)")
    save_figure(fig, "eda_correlation", show)


def findings(df: pd.DataFrame) -> list[str]:
    """Translate five measured training associations into modeling hypotheses."""
    notes = []
    work = df.assign(tenure_bucket=tenure_bucket(df["tenure"]))
    suggestions = {
        "Contract": "encode contract duration; it may capture commitment",
        "InternetService": "encode service type and consider service-related interactions",
        "PaymentMethod": "keep payment categories and test an autopay indicator",
        "tenure_bucket": "retain numeric tenure and test fixed tenure groups",
    }
    for column, suggestion in suggestions.items():
        rates = churn_rates(work, column)
        high, low = rates.iloc[0], rates.iloc[-1]
        notes.append(
            f"{column}: {rates.index[0]} churns at {high.churn_rate:.1%} (n={int(high.customers)}) "
            f"versus {rates.index[-1]} at {low.churn_rate:.1%} (n={int(low.customers)}) "
            f"-> {suggestion}; this is an association, not a causal effect."
        )
    medians = df.groupby("Churn")["MonthlyCharges"].median()
    notes.append(f"Median MonthlyCharges is {medians[1]:.2f} for churners and {medians[0]:.2f} "
                 "for stayers -> test charge-related features without assuming price causes churn.")
    return notes


def run_eda(show: bool = False) -> list[str]:
    """Persist plots, group tables, and evidence using training rows only."""
    ensure_dirs()
    df = training_frame()
    for plot in (plot_target, plot_rates, plot_numeric, plot_outliers, plot_correlation):
        plot(df, show)
    work = df.assign(tenure_bucket=tenure_bucket(df["tenure"]))
    for column in ("Contract", "InternetService", "PaymentMethod", "tenure_bucket"):
        churn_rates(work, column).to_csv(REPORTS / f"eda_{column}.csv")
    notes = findings(df)
    (REPORTS / "eda_findings.md").write_text("\n".join(f"- {x}" for x in notes), encoding="utf-8")
    LOGGER.info("Training-only EDA (%d rows):\n%s", len(df), "\n".join(notes))
    return notes
