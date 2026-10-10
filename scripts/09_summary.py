"""Show final results and demonstrate the saved one-customer predictor."""

import logging

import _bootstrap  # noqa: F401
from src.summary import run_summary


def main() -> None:
    """Log a demo from a genuine training customer."""
    run_summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
