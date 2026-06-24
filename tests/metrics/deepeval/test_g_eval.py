"""Unit tests for GEvalMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import GEval as DeepEvalGEval
from deepeval.test_case import LLMTestCaseParams

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.g_eval import (
    GEvalMetric,
    GEvalParam,
    Rubric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_geval_score():
    metric = GEvalMetric(make_mock_dial_factory(), DeepEvalGroupSettings())
    config = metric.Config(
        model="gemini-2.5-flash-lite",
        threshold=0.5,
        eval_name="Correctness",
        criteria="Determine whether the actual output is factually correct based on the expected output.",
        evaluation_params=[
            GEvalParam.ACTUAL_OUTPUT,
            GEvalParam.EXPECTED_OUTPUT,
        ],
    )
    input_data = metric.Input(
        input="What is the diameter of the Earth?",
        actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
        expected_output="The diameter of the Earth is approximately 12,742 kilometers.",
    )

    with patch.object(
        DeepEvalGEval,
        "a_measure",
        new=mock_a_measure(1.0, "Factually correct.", "Steps..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(1.0)
    assert result.score.details.get("reason") == "Factually correct."
    assert result.score.details.get("verbose_logs") == "Steps..."


@pytest.mark.asyncio
async def test_geval_error():
    metric = GEvalMetric(make_mock_dial_factory(), DeepEvalGroupSettings())
    config = metric.Config(
        model="gemini-2.5-flash-lite",
        eval_name="Correctness",
        criteria="Determine whether the actual output is factually correct.",
        evaluation_params=[
            GEvalParam.ACTUAL_OUTPUT,
            GEvalParam.EXPECTED_OUTPUT,
        ],
    )
    input_data = metric.Input(
        input="Q?",
        actual_output="A.",
        expected_output="A.",
    )

    with patch.object(
        DeepEvalGEval,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message


def test_geval_rejects_both_criteria_and_evaluation_steps(app_client):
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepeval.g_eval",
            "config": {
                "eval_name": "Both",
                "criteria": "Evaluate correctness.",
                "evaluation_steps": ["Step 1: check output.", "Step 2: score."],
                "evaluation_params": ["actual_output"],
            },
            "input": {"input": "Q?", "actual_output": "A."},
        },
    )
    assert response.status_code == 422


def test_geval_rejects_neither_criteria_nor_evaluation_steps(app_client):
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepeval.g_eval",
            "config": {
                "eval_name": "Neither",
                "evaluation_params": ["actual_output"],
            },
            "input": {"input": "Q?", "actual_output": "A."},
        },
    )
    assert response.status_code == 422


def test_geval_rejects_unknown_evaluation_param(app_client):
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepeval.g_eval",
            "config": {
                "eval_name": "Injection attempt",
                "criteria": "Evaluate something.",
                "evaluation_params": ["api_key"],
            },
            "input": {"input": "Q?", "actual_output": "A."},
        },
    )
    assert response.status_code == 422


def test_geval_param_is_subset_of_llm_test_case_params():
    """Ensure every GEvalParam value maps to a valid LLMTestCaseParams value.

    This catches upstream deepeval enum renames/removals before they cause
    silent runtime failures in build_metric.
    """
    llm_test_case_values = {p.value for p in LLMTestCaseParams}
    for param in GEvalParam:
        assert param.value in llm_test_case_values, (
            f"GEvalParam.{param.name} ('{param.value}') has no matching LLMTestCaseParams value. "
            "Update GEvalParam to reflect the deepeval library change."
        )


@pytest.mark.asyncio
async def test_geval_rubric_score():
    metric = GEvalMetric(make_mock_dial_factory(), DeepEvalGroupSettings())
    config = metric.Config(
        model="gemini-2.5-flash-lite",
        threshold=0.5,
        eval_name="Correctness",
        criteria="Determine whether the actual output is factually correct.",
        evaluation_params=[
            GEvalParam.ACTUAL_OUTPUT,
            GEvalParam.EXPECTED_OUTPUT,
        ],
        rubric=[
            Rubric(start=0, end=4, expected_outcome="Incorrect or missing."),
            Rubric(start=5, end=7, expected_outcome="Partially correct."),
            Rubric(start=8, end=10, expected_outcome="Fully correct."),
        ],
    )
    input_data = metric.Input(
        input="What is the diameter of the Earth?",
        actual_output="The diameter of the Earth is approximately 12,742 kilometers.",
        expected_output="The diameter of the Earth is approximately 12,742 kilometers.",
    )

    with patch.object(
        DeepEvalGEval,
        "a_measure",
        new=mock_a_measure(1.0, "Fully correct.", "Steps..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(1.0)


def test_geval_rejects_invalid_rubric_score_range(app_client):
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepeval.g_eval",
            "config": {
                "eval_name": "OutOfBounds",
                "criteria": "Evaluate correctness.",
                "evaluation_params": ["actual_output"],
                "rubric": [
                    {
                        "start": 0,
                        "end": 11,
                        "expected_outcome": "Out of bounds.",
                    },
                ],
            },
            "input": {"input": "Q?", "actual_output": "A."},
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("less than or equal to 10" in error["msg"] for error in detail)


def test_geval_rejects_overlapping_rubrics(app_client):
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "deepeval.g_eval",
            "config": {
                "eval_name": "Overlap",
                "criteria": "Evaluate correctness.",
                "evaluation_params": ["actual_output"],
                "rubric": [
                    {"start": 0, "end": 5, "expected_outcome": "Poor."},
                    {"start": 4, "end": 8, "expected_outcome": "Good."},
                ],
            },
            "input": {"input": "Q?", "actual_output": "A."},
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("Overlapping" in error["msg"] for error in detail)
