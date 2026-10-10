"""Audit the source CSV without modifying it."""

import logging

import _bootstrap  # noqa: F401
from src.data import audit, load_raw


def main() -> None:
    """Log the complete audit dictionary."""
    audit(load_raw())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
