"""Apply tested cleaning rules and save the cleaned data."""

import logging

import _bootstrap  # noqa: F401
from src.data import run_cleaning


def main() -> None:
    """Run deterministic cleaning and log its comparison table."""
    run_cleaning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
