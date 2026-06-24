"""Unit tests for ContextualRelevancyMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import (
    ContextualRelevancyMetric as DeepEvalContextualRelevancyMetric,
)

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.contextual_relevancy import (
    ContextualRelevancyMetric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_contextual_relevancy_score():
    metric = ContextualRelevancyMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="What is the speed of light?",
        retrieval_context=[
            "Light travels at ~299,792 km/s.",
            "Water boils at 100°C.",
        ],
    )

    with patch.object(
        DeepEvalContextualRelevancyMetric,
        "a_measure",
        new=mock_a_measure(0.5, "Half relevant.", "..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_contextual_relevancy_error():
    metric = ContextualRelevancyMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="Q?",
        retrieval_context=["Context."],
    )

    with patch.object(
        DeepEvalContextualRelevancyMetric,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
