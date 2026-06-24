"""Integration tests for deepeval.contextual_recall with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.deepeval.contextual_recall import (
    ContextualRecallMetric,
)


@pytest.mark.vcr
def test_contextual_recall_evaluate_via_api(app_client):
    request = {
        "metric_name": "deepeval.contextual_recall",
        "config": {"model": "gemini-2.5-flash-lite", "threshold": 0.5},
        "input": {
            "input": "What is the diameter of the Earth?",
            "expected_output": "The diameter of the Earth is approximately 12,742 kilometers.",
            "retrieval_context": [
                "The Earth has a diameter of approximately 12,742 kilometers.",
                "The Earth orbits the Sun at a distance of about 150 million kilometers.",
            ],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "deepeval.contextual_recall"
    output = ContextualRecallMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
    assert "reason" in output.score.details
    assert "verbose_logs" in output.score.details
