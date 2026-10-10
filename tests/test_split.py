"""Protect the shared holdout from overlap and accidental resplitting."""

import numpy as np

from src.data import load_clean
from src.split import split_data
from src.train import cv_folds


def test_split_disjoint_reproducible_stratified() -> None:
    """Each cleaned customer belongs to exactly one stable partition."""
    a, b, ya, yb = split_data()
    again, _, _, _ = split_data()
    assert set(a.index).isdisjoint(b.index)
    assert len(a) + len(b) == len(load_clean())
    np.testing.assert_array_equal(a.index, again.index)
    assert abs(ya.mean() - yb.mean()) < 0.002
    assert "Churn" not in a and "customerID" not in a


def test_cv_never_includes_test_indices() -> None:
    """Training CV fold positions resolve only to training customer indices."""
    train, test, y, _ = split_data()
    seen = []
    for fitting, validation in cv_folds().split(train, y):
        assert set(fitting).isdisjoint(validation)
        assert set(train.iloc[fitting].index).isdisjoint(test.index)
        assert set(train.iloc[validation].index).isdisjoint(test.index)
        seen.extend(validation)
    assert sorted(seen) == list(range(len(train)))
