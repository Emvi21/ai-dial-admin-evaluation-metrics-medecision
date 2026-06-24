"""Integration tests for ragas.answer_relevancy with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.

Requires an embeddings model configured in the app settings (RAGAS__EMBEDDINGS_MODEL)
or passed in the request config.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.answer_relevancy import (
    AnswerRelevancyMetric,
)


@pytest.mark.vcr
def test_answer_relevancy_evaluate_via_api(app_client):
    request = {
        "metric_name": "ragas.answer_relevancy",
        "config": {
            "model": "gemini-2.5-flash-lite",
            "embeddings_model": "text-embedding-ada-002",
        },
        "input": {
            "user_input": "When was the first super bowl?",
            "response": "The first superbowl was held on Jan 15, 1967",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "ragas.answer_relevancy"
    output = AnswerRelevancyMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
