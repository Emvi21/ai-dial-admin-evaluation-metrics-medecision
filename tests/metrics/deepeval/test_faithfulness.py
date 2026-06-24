"""Unit tests for FaithfulnessMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import FaithfulnessMetric as DeepEvalFaithfulnessMetric

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.faithfulness import (
    FaithfulnessMetric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_faithfulness_score():
    metric = FaithfulnessMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", threshold=0.5)
    input_data = metric.Input(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        retrieval_context=["France's capital city is Paris."],
    )

    with patch.object(
        DeepEvalFaithfulnessMetric,
        "a_measure",
        new=mock_a_measure(1.0, "Fully supported.", "Claims..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == 1.0
    assert result.score.details.get("reason") == "Fully supported."


@pytest.mark.asyncio
async def test_faithfulness_error():
    metric = FaithfulnessMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="Q?", actual_output="A.", retrieval_context=["Context."]
    )

    with patch.object(
        DeepEvalFaithfulnessMetric,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
