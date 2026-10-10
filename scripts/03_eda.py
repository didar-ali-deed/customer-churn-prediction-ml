"""Save all training-only exploratory figures and findings."""

import logging

import _bootstrap  # noqa: F401
from src.eda import run_eda


def main() -> None:
    """Explore training customers without opening the reserved test labels."""
    run_eda()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
