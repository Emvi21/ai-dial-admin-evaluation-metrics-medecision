"""Unit tests for NLI metric (schema, registry, error handling)."""

from unittest.mock import MagicMock, patch

from aidial_admin_evaluation_metrics.app_config import (
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.nli import (
    NLIMetric,
)


def test_nli_llm_error_via_api(app_client):
    """Test NLI metric when LLM call fails via API."""
    with patch(
        "aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.nli.calculate_inference"
    ) as mock_calc_inference:
        mock_calc_inference.side_effect = Exception("LLM API error")

        request = {
            "metric_name": "aidial_rag_eval.nli",
            "config": {"model": "gemini-2.5-flash-lite"},
            "input": {
                "premise": "The Earth is a planet.",
                "hypothesis": "Earth orbits the Sun.",
            },
        }

        response = app_client.post("/evaluate", json=request)
        assert response.status_code == 200

        result = response.json()
        assert result["metric_name"] == "aidial_rag_eval.nli"
        assert "inference" in result["output"]
        assert "message" in result["output"]["inference"]
        assert "LLM API error" in result["output"]["inference"]["message"]


def test_nli_metric_in_registry(app_client):
    """Test that NLI metric is registered and appears in metrics list."""
    response = app_client.get("/metrics")
    assert response.status_code == 200

    metrics = response.json()["metrics"]
    metric_names = [m["name"] for m in metrics]

    assert "aidial_rag_eval.nli" in metric_names

    nli_metric = next(m for m in metrics if m["name"] == "aidial_rag_eval.nli")
    assert "description" in nli_metric
    assert "config_schema" in nli_metric
    assert "input_schema" in nli_metric
    assert "output_schema" in nli_metric

    assert "model" in nli_metric["config_schema"]["properties"]
    model_prop = nli_metric["config_schema"]["properties"]["model"]

    # With Literal type, Pydantic generates enum inline (no $ref)
    assert "enum" in model_prop
    assert "gemini-2.5-flash-lite" in model_prop["enum"]
    assert "gpt-5-nano-2025-08-07" in model_prop["enum"]


def test_nli_metric_output_schema():
    """Test that output schema is correctly structured."""
    metric = NLIMetric(MagicMock(), AidialRagEvalGroupSettings())

    assert hasattr(metric.Output, "model_fields")
    fields = metric.Output.model_fields

    assert "inference" in fields
