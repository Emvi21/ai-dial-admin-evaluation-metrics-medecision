"""Integration tests for deepeval.answer_relevancy with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.deepeval.answer_relevancy import (
    AnswerRelevancyMetric,
)


@pytest.mark.vcr
def test_answer_relevancy_evaluate_via_api(app_client):
    request = {
        "metric_name": "deepeval.answer_relevancy",
        "config": {"model": "gemini-2.5-flash-lite", "threshold": 0.5},
        "input": {
            "input": "What is the diameter of the Earth?",
            "actual_output": "The diameter of the Earth is approximately 12,742 kilometers.",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "deepeval.answer_relevancy"
    output = AnswerRelevancyMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
    # Ensure the recorded evaluation includes a human-readable reason and
    # verbose evaluation logs (non-empty strings). Recorded cassettes should
    # contain these fields when run in record-mode.
    reason = output.score.details.get("reason", "")
    verbose_logs = output.score.details.get("verbose_logs", "")
    assert isinstance(reason, str) and reason.strip() != ""
    assert isinstance(verbose_logs, str) and verbose_logs.strip() != ""
