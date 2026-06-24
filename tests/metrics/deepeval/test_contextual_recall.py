"""Unit tests for ContextualRecallMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import (
    ContextualRecallMetric as DeepEvalContextualRecallMetric,
)

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.contextual_recall import (
    ContextualRecallMetric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_contextual_recall_score():
    metric = ContextualRecallMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="What is the speed of light?",
        expected_output="The speed of light is approximately 299,792 km/s.",
        retrieval_context=["Light travels at ~299,792 km/s."],
    )

    with patch.object(
        DeepEvalContextualRecallMetric,
        "a_measure",
        new=mock_a_measure(1.0, "All statements covered.", "..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_contextual_recall_error():
    metric = ContextualRecallMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="Q?",
        expected_output="A.",
        retrieval_context=["Context."],
    )

    with patch.object(
        DeepEvalContextualRecallMetric,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
