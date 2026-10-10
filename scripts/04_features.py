"""Demonstrate business features and training-only preprocessing."""

import logging

import _bootstrap  # noqa: F401
from src.features import run_features


def main() -> None:
    """Save diagnostics for all six engineered features."""
    run_features()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
