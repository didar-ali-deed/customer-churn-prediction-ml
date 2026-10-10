"""Compare three models and tune the CV winner, using no test evaluation."""

import logging

import _bootstrap  # noqa: F401
from src.train import run_training


def main() -> None:
    """Persist the winning complete pipeline and training evidence."""
    run_training()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
