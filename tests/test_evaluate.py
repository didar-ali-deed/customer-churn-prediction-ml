"""Verify asymmetric costs and prevent repeated held-out inference."""

import numpy as np
import pytest

from src.config import REPORTS
from src.evaluate import choose_threshold, cost_counts, run_evaluation


def test_cost_and_optimal_threshold() -> None:
    """Explicit mistakes give a known cost and the scan finds its minimum."""
    y = np.array([1, 1, 0, 0])
    p = np.array([0.8, 0.3, 0.6, 0.1])
    assert cost_counts(y, p, 0.5) == {"false_negative": 1, "false_positive": 1, "cost": 6}
    t, table = choose_threshold(y, p)
    assert table["cost"].min() == 1
    assert 0.1 < t <= 0.3
    assert cost_counts(y, p, t)["cost"] == table["cost"].min()


def test_invalid_threshold_inputs() -> None:
    """NaNs and nonbinary labels must not silently produce offer decisions."""
    with pytest.raises(ValueError):
        choose_threshold(np.array([1]), np.array([np.nan]))
    with pytest.raises(ValueError):
        choose_threshold(np.array([2]), np.array([0.5]))


def test_cached_evaluation_never_predicts_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A matching report is reused without loading/invoking the saved pipeline."""
    if not (REPORTS / "evaluation.json").exists():
        pytest.skip("Final assessment has not yet been generated.")

    def forbidden(*args: object, **kwargs: object) -> None:
        """Reject a pipeline load on the cached assessment path."""
        raise AssertionError("Cached evaluation attempted to load a model.")

    monkeypatch.setattr("src.evaluate.joblib.load", forbidden)
    result = run_evaluation()
    assert result["test_rows"] == 1405
