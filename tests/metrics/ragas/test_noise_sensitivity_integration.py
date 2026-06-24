"""Integration tests for ragas.noise_sensitivity with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.ragas.noise_sensitivity import (
    NoiseSensitivityMetric,
)


@pytest.mark.vcr
def test_noise_sensitivity_evaluate_via_api(app_client):
    request = {
        "metric_name": "ragas.noise_sensitivity",
        "config": {"model": "gemini-2.5-flash-lite", "mode": "relevant"},
        "input": {
            "user_input": "When was the Eiffel Tower built?",
            "response": "The Eiffel Tower was built in 1889.",
            "reference": "The Eiffel Tower was constructed from 1887 to 1889.",
            "retrieved_contexts": [
                "The Eiffel Tower was constructed from 1887 to 1889 as the entrance arch "
                "for the 1889 World's Fair."
            ],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "ragas.noise_sensitivity"
    output = NoiseSensitivityMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
