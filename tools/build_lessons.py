"""Regenerate all lesson manuscripts before executing them with nbconvert."""

import importlib
import logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Write nine notebooks; run tools.reproduce --notebooks to restore outputs."""
    for number in range(1, 10):
        name = f"tools.lesson_{number:02d}"
        LOGGER.info("Building %s", name)
        importlib.import_module(name).main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
