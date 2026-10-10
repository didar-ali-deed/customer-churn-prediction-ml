"""Run every script and optionally execute all nine notebooks in order."""

import argparse
import logging
import subprocess
import sys

from src.config import ROOT

LOGGER = logging.getLogger(__name__)
RUNNERS = ["01_audit.py", "02_clean.py", "03_eda.py", "04_features.py",
           "05_baseline.py", "06_train.py", "07_evaluate.py", "08_explain.py", "09_summary.py"]


def main() -> None:
    """Reproduce artifacts before checking lint/tests, with optional notebook execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", action="store_true")
    args = parser.parse_args()
    for runner in RUNNERS:
        LOGGER.info("Running %s", runner)
        subprocess.run([sys.executable, str(ROOT / "scripts" / runner)], cwd=ROOT, check=True)
    if args.notebooks:
        for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
            LOGGER.info("Executing %s", notebook.name)
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook",
                            "--execute", "--inplace", "--ExecutePreprocessor.timeout=7200",
                            str(notebook)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
