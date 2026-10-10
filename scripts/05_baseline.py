"""Run the shared split and five-fold baseline comparison."""

import logging

import _bootstrap  # noqa: F401
from src.train import run_baselines


def main() -> None:
    """Save dummy and logistic CV results without test evaluation."""
    run_baselines()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
