"""Validate a local clone plus pending final files in an independent fresh venv."""

import json
import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import TextIO

from src.config import RAW, REPORTS, ROOT

LOGGER = logging.getLogger(__name__)


def run(command: list[str], cwd: Path, log: TextIO) -> None:
    """Record complete subprocess evidence while emitting concise progress."""
    LOGGER.info("Validation: %s", " ".join(command))
    env = os.environ.copy()
    env.update(MPLBACKEND="Agg", PYTHONHASHSEED="42")
    subprocess.run(command, cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    log.flush()


def main() -> None:
    """Clone history, overlay pending files, discard derived artifacts, and rebuild.

    A pre-final-commit check must include uncommitted step-10 files. This uses
    an actual git clone, then copies the reviewed working tree over it before
    building an independent venv. No original artifact is deleted or altered.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recheck", type=Path, help="Recheck corrected lessons in an existing validation sandbox.")
    args = parser.parse_args()
    if args.recheck:
        recheck(args.recheck)
        return
    validation_root = ROOT / ".validation"
    validation_root.mkdir(exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix="fresh-", dir=validation_root))
    clone = sandbox / "churn-ml"
    log_path = sandbox / "validation.log"
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}",
           "-c", f"safe.directory={(ROOT / '.git').as_posix()}"]
    with log_path.open("w", encoding="utf-8") as log:
        run([*git, "clone", "--no-hardlinks", ROOT.as_posix(), clone.as_posix()], ROOT, log)
        paths = subprocess.check_output(
            [*git, "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, text=True,
        ).splitlines()
        for relative in paths:
            source = ROOT / relative
            destination = clone / relative
            if source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        (clone / "data/raw").mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAW, clone / "data/raw/telco.csv")
        # Delete only generated files within this newly created validation clone.
        for relative in ("reports", "data/processed", "models"):
            target = (clone / relative).resolve()
            if not target.is_relative_to(clone.resolve()):
                raise ValueError("Refusing cleanup outside the validation clone.")
            for artifact in target.rglob("*"):
                if artifact.is_file() and artifact.name != ".gitkeep":
                    if not artifact.resolve().is_relative_to(target):
                        raise ValueError("Refusing an artifact outside its verified directory.")
                    artifact.unlink()
        run([sys.executable, "-m", "venv", str(clone / ".venv")], clone, log)
        python = clone / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(python), "-m", "pip", "install", "-r", "requirements.txt"], clone, log)
        run([str(python), "-m", "pip", "check"], clone, log)
        run([str(python), "-m", "tools.reproduce", "--notebooks"], clone, log)
    evidence = {
        "status": "passed", "python": sys.version.split()[0],
        "mode": "Actual local Git clone plus pending final working-tree files, new venv, derived artifacts removed",
        "requirements_install": "passed", "pip_check": "passed",
        "scripts_01_through_09": "passed in order", "notebooks_01_through_09": "executed without errors",
        "ruff": "passed", "pytest": "passed",
        "log": log_path.relative_to(ROOT).as_posix(),
        "original_artifacts": "unchanged",
    }
    (REPORTS / "reproduction_check.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    LOGGER.info("Independent reproduction passed; full evidence: %s", log_path)


def recheck(sandbox: Path) -> None:
    """Verify lesson fixes using a previously installed independent clone/venv."""
    sandbox = sandbox.resolve()
    if not sandbox.is_relative_to((ROOT / ".validation").resolve()):
        raise ValueError("Recheck must stay within this project's validation directory.")
    clone = sandbox / "churn-ml"
    python = clone / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise FileNotFoundError("This sandbox does not contain an independent venv.")
    for directory in ("src", "scripts", "tools", "tests"):
        for source in (ROOT / directory).glob("*.py"):
            shutil.copy2(source, clone / directory / source.name)
    for name in ("README.md", "pyproject.toml"):
        shutil.copy2(ROOT / name, clone / name)
    log_path = sandbox / "validation.log"
    with log_path.open("a", encoding="utf-8") as log:
        run([str(python), "-m", "tools.build_lessons"], clone, log)
        run([str(python), "-m", "tools.reproduce", "--notebooks"], clone, log)
    evidence_path = REPORTS / "reproduction_check.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence.update(inline_plot_check="passed", project_venv_kernel_guard="passed",
                    lesson_fix_recheck="All scripts/notebooks rerun in the same independently created venv; test evaluation reused")
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    LOGGER.info("Independent corrected-lesson recheck passed: %s", log_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
