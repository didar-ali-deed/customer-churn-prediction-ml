# Project rules

- Use Python 3.11 and `python -m venv .venv`; no conda, containers, or cloud tools.
- Use pandas and numpy for data loading/manipulation. Other direct dependencies are limited to matplotlib, seaborn, scikit-learn, xgboost, shap, joblib, jupyter, nbformat, nbconvert, pytest, and ruff. Standard-library utilities and their transitive dependencies are allowed.
- Keep reusable logic in src; notebooks explain it and scripts call it. Python functions need type hints and docstrings; runners use logging.
- Set all randomized estimators, splits, searches, and sampling to 42.
- Split before statistical exploration used for model decisions. Fit all learned preprocessing inside training CV folds. Choose a threshold from training out-of-fold predictions, never test labels.
- Evaluate the test set once per reproducible run. Subsequent runners/notebooks read the saved evaluation for the same model/data fingerprint.
- Before each commit execute the step's script and notebook (when introduced), `ruff check .`, and `pytest`. Keep one Conventional Commit per project step; never squash or rewrite history.
- Do not fabricate the source dataset or commit the raw CSV/model binaries. Freeze installed dependency versions with `pip freeze`.
