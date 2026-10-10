"""Execute one step and enforce quality gates before its single commit."""

import argparse
import logging
import subprocess
import sys

from src.config import ROOT

LOGGER = logging.getLogger(__name__)


def run(command: list[str]) -> None:
    """Stop immediately on a failed check and retain its console evidence."""
    if command[0] == "git":
        command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", *command[1:]]
    LOGGER.info("Running %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    """Execute runner/notebook, lint, tests, commit, and display history."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner")
    parser.add_argument("--notebook")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if args.runner:
        run([sys.executable, str(ROOT / args.runner)])
    if args.notebook:
        run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", "--ExecutePreprocessor.timeout=7200", args.notebook])
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "pytest", "-q"])
    run(["git", "add", "."])
    run(["git", "commit", "-m", args.commit])
    run(["git", "log", "--oneline"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
