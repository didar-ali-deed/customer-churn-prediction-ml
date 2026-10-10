"""Check model explanations reconstruct the frozen output."""

import json

from src.config import REPORTS
from src.explain import original_feature


def test_encoded_columns_map_to_business_fields() -> None:
    """Underscores in engineered names must not truncate field attribution."""
    assert original_feature("cat__tenure_bucket_0-12") == "tenure_bucket"
    assert original_feature("num__avg_monthly_spend") == "avg_monthly_spend"
    assert original_feature("cat__Contract_Month-to-month") == "Contract"


def test_saved_shap_additivity() -> None:
    """The persisted explanation must agree with the actual model output."""
    info = json.loads((REPORTS / "explainability.json").read_text(encoding="utf-8"))
    assert info["max_additivity_error"] < 1e-4
    assert info["output_units"] in ("log-odds", "churn probability")
    assert 0 <= info["local_churn_probability"] <= 1
