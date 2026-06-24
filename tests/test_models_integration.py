"""Integration tests that iterate supported models via the /evaluate API.

These tests record/replay real LLM calls via pytest-recording (vcr). They use
the same basic assertions as the NLI integration test.
"""

import pytest

from aidial_admin_evaluation_metrics.app_config import SUPPORTED_MODELS


@pytest.mark.integration
@pytest.mark.vcr
@pytest.mark.parametrize("model", SUPPORTED_MODELS)
def test_models_evaluate_via_api(app_client, test_dial_config, model):
    """Call the `/evaluate` API for each supported model and assert basic output."""
    request = {
        "metric_name": "aidial_rag_eval.nli",
        "config": {"model": model},
        "input": {
            "premise": (
                "The Earth is the third planet from the Sun. The diameter of the Earth "
                "is approximately 12,742 kilometers."
            ),
            "hypothesis": "The Earth has a diameter of about 12,742 km.",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    result = response.json()
    assert result["metric_name"] == "aidial_rag_eval.nli"
    assert "inference" in result["output"]
    assert 0.0 <= result["output"]["inference"]["value"] <= 1.0
    assert "json_explanation" in result["output"]["inference"]["details"]
    assert "highlight" in result["output"]["inference"]["details"]
