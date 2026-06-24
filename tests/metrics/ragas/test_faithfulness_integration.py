"""Integration tests for ragas.faithfulness with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.faithfulness import (
    FaithfulnessMetric,
)


@pytest.mark.vcr
def test_faithfulness_evaluate_via_api(app_client):
    request = {
        "metric_name": "ragas.faithfulness",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "user_input": "When was the first super bowl?",
            "response": "The first superbowl was held on Jan 15, 1967",
            "retrieved_contexts": [
                "The First AFL–NFL World Championship Game was an American football game "
                "played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
            ],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "ragas.faithfulness"
    output = FaithfulnessMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
