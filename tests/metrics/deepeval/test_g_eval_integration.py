"""Integration tests for deepeval.g_eval with recorded LLM responses.

Run with --record-mode=once against a real DIAL endpoint to record cassettes.
See tests/cassettes.md for documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.metrics.deepeval.g_eval import GEvalMetric


@pytest.mark.vcr
def test_g_eval_evaluate_via_api(app_client):
    request = {
        "metric_name": "deepeval.g_eval",
        "config": {
            "model": "gpt-5-nano-2025-08-07",
            "threshold": 0.5,
            "eval_name": "Correctness",
            "criteria": (
                "Determine whether the actual output is factually correct "
                "based on the expected output."
            ),
            "evaluation_params": ["actual_output", "expected_output"],
        },
        "input": {
            "input": "What is the diameter of the Earth?",
            "actual_output": "The diameter of the Earth is approximately 12,742 kilometers.",
            "expected_output": "The diameter of the Earth is approximately 12,742 kilometers.",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "deepeval.g_eval"
    output = GEvalMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
    reason = output.score.details.get("reason", "")
    verbose_logs = output.score.details.get("verbose_logs", "")
    assert isinstance(reason, str) and reason.strip() != ""
    assert isinstance(verbose_logs, str) and verbose_logs.strip() != ""


@pytest.mark.vcr
def test_g_eval_with_rubric_via_api(app_client):
    request = {
        "metric_name": "deepeval.g_eval",
        "config": {
            "model": "gpt-5-nano-2025-08-07",
            "threshold": 0.5,
            "eval_name": "Correctness",
            "criteria": (
                "Determine whether the actual output is factually correct "
                "based on the expected output."
            ),
            "evaluation_params": ["actual_output", "expected_output"],
            "rubric": [
                {
                    "start": 0,
                    "end": 3,
                    "expected_outcome": "The answer is incorrect or missing key facts.",
                },
                {
                    "start": 4,
                    "end": 6,
                    "expected_outcome": "The answer is partially correct but imprecise.",
                },
                {
                    "start": 7,
                    "end": 10,
                    "expected_outcome": "The answer is accurate and matches the expected output.",
                },
            ],
        },
        "input": {
            "input": "What is the diameter of the Earth?",
            "actual_output": "The diameter of the Earth is approximately 12,742 kilometers.",
            "expected_output": "The diameter of the Earth is approximately 12,742 kilometers.",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    data = response.json()
    assert data["metric_name"] == "deepeval.g_eval"
    output = GEvalMetric.Output.model_validate(data["output"])
    assert output.score.type == "value"
    assert 0.0 <= output.score.value <= 1.0
    reason = output.score.details.get("reason", "")
    verbose_logs = output.score.details.get("verbose_logs", "")
    assert isinstance(reason, str) and reason.strip() != ""
    assert isinstance(verbose_logs, str) and verbose_logs.strip() != ""
