"""Integration tests for ragas.context_recall with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.context_recall import (
    ContextRecallMetric,
)


@pytest.mark.vcr
def test_context_recall_evaluate_via_api(app_client):
    request = {
        "metric_name": "ragas.context_recall",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "user_input": "Where is the Eiffel Tower located?",
            "reference": "The Eiffel Tower is located in Paris.",
            "retrieved_contexts": [
                "Paris is the capital of France.",
            ],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "ragas.context_recall"
    output = ContextRecallMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
