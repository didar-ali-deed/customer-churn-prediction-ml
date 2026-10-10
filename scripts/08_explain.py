"""Explain the frozen model with training-only SHAP diagnostics."""

import logging

import _bootstrap  # noqa: F401
from src.explain import run_explanations


def main() -> None:
    """Save two global plots, one local explanation, and business hypotheses."""
    run_explanations()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
