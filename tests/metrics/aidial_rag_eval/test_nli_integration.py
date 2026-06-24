"""Integration tests for NLI metric with recorded LLM responses.

These tests use pytest-recording to record real LLM API calls and replay them.
See tests/cassettes.md for detailed documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.app_config import (
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.generation_output import (
    InferenceField,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.nli import (
    NLIMetric,
)


@pytest.mark.integration
@pytest.mark.vcr
def test_nli_direct_evaluation_with_real_llm(test_dial_config):
    """Test NLI metric with real LLM - responses recorded/replayed."""
    llm_factory = create_dial_factory(test_dial_config)

    nli_metric = NLIMetric(llm_factory, AidialRagEvalGroupSettings())
    config = nli_metric.Config(model="gemini-2.5-flash-lite")
    input_data = nli_metric.Input(
        premise="The Earth has a diameter of approximately 12,742 kilometers.",
        hypothesis="The Earth has a diameter of about 12,742 km.",
    )

    # This will make real LLM calls (or replay from cassette)
    result = nli_metric._evaluate(config, input_data)

    assert isinstance(result.inference, InferenceField)
    assert not hasattr(
        result.inference, "message"
    ), f"Expected successful inference but got error: {getattr(result.inference, 'message', 'N/A')}"

    assert (
        0.0 <= result.inference.value <= 1.0
    ), f"Inference value {result.inference.value} is out of valid range [0.0, 1.0]"

    assert (
        result.inference.details.json_explanation
    ), "Missing json_explanation in details"
    assert result.inference.details.highlight, "Missing highlight in details"

    # The actual value depends on the LLM response, but for this highly
    # similar premise/hypothesis pair, we expect a relatively high score
    assert (
        result.inference.value > 0.5
    ), f"Expected high inference score for similar texts, got {result.inference.value}"


@pytest.mark.integration
@pytest.mark.vcr
def test_nli_evaluate_via_api(app_client, test_dial_config):
    """Test NLI with example data from metric definition via API with real LLM."""
    request = {
        "metric_name": "aidial_rag_eval.nli",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "premise": "The Earth is the third planet from the Sun. The diameter of the Earth is approximately 12,742 kilometers.",
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


@pytest.mark.integration
@pytest.mark.vcr
def test_nli_invalid_model_via_api(app_client, test_dial_config):
    """Test NLI metric error handling for invalid model name via API."""
    request = {
        "metric_name": "aidial_rag_eval.nli",
        "config": {"model": "invalid-model-name"},
        "input": {
            "premise": "Python is a programming language.",
            "hypothesis": "Python is used for software development.",
        },
    }

    response = app_client.post("/evaluate", json=request)

    # Pydantic validates the model name at request time
    assert response.status_code == 422

    result = response.json()
    assert "detail" in result
    assert isinstance(result["detail"], list)
    assert len(result["detail"]) > 0

    first_error = result["detail"][0]
    assert "loc" in first_error
    assert "config" in first_error["loc"]
    assert "model" in first_error["loc"]
    assert "msg" in first_error
    assert "type" in first_error
