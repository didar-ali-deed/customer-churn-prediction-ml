"""One deterministic split shared by EDA, training, and final evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import SEED
from src.data import load_clean


def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Reserve a stratified 20% holdout; never learn any transformation here."""
    df = load_clean()
    return train_test_split(
        df.drop(columns="Churn"), df["Churn"], test_size=0.2,
        stratify=df["Churn"], random_state=SEED,
    )


def training_frame() -> pd.DataFrame:
    """Expose only training rows to exploratory analysis."""
    x_train, _, y_train, _ = split_data()
    return x_train.assign(Churn=y_train)
