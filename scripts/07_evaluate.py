"""Lock a training-selected threshold and perform one held-out assessment."""

import logging

import _bootstrap  # noqa: F401
from src.evaluate import run_evaluation


def main() -> None:
    """Save or reuse final results; never optimize offers on test labels."""
    run_evaluation()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
