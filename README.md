# Customer Churn Prediction

A beginner-friendly, local Python project using IBM Telco Customer Churn. Churn means a customer leaves; the goal is to rank customers by risk and choose a retention threshold with an explicit cost tradeoff.

## Setup (Windows PowerShell, Python 3.11)

```powershell
cd churn-ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m ruff check .
python -m pytest
python -m jupyter notebook
```

Activation is optional: use `.venv\Scripts\python.exe` directly if PowerShell blocks activation. On macOS/Linux activate with `source .venv/bin/activate`.

Download [Kaggle blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), extract `WA_Fn-UseC_-Telco-Customer-Churn.csv`, and place it at `data/raw/telco.csv`. The raw data and trained joblib models are intentionally ignored by Git. The supplied original CSV has been copied to this location locally.

## Learning sequence

1. Environment and reproducibility (this scaffold).
2. Audit the real dataset.
3. Clean deterministic data issues.
4. Explore training data and form hypotheses.
5. Engineer features and build fold-safe preprocessing.
6. Stratified split and honest baselines.
7. Compare and tune models with training CV.
8. Select a threshold on training predictions and evaluate held-out data once.
9. Explain model behavior with SHAP.
10. Summarize results and predict one customer.

Each analytical step will have an executed notebook and an equivalent thin Python runner. Libraries are installed in a dedicated venv, and `requirements.txt` is generated with `pip freeze` (including necessary transitive dependencies). Step 1 introduces no analytical notebook or runner; its executable checks are the real-data schema test and tooling checks.
