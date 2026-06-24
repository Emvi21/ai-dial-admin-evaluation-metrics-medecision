"""Unit tests for ContextualPrecisionMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import (
    ContextualPrecisionMetric as DeepEvalContextualPrecisionMetric,
)

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.contextual_precision import (
    ContextualPrecisionMetric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_contextual_precision_score():
    metric = ContextualPrecisionMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="What is the speed of light?",
        actual_output="The speed of light is approximately 299,792 km/s.",
        expected_output="The speed of light is approximately 299,792 km/s.",
        retrieval_context=[
            "Light travels at ~299,792 km/s.",
            "Water boils at 100°C.",
        ],
    )

    with patch.object(
        DeepEvalContextualPrecisionMetric,
        "a_measure",
        new=mock_a_measure(0.8, "Good precision.", "Verdicts..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_contextual_precision_error():
    metric = ContextualPrecisionMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        input="Q?",
        actual_output="A.",
        expected_output="A.",
        retrieval_context=["Context."],
    )

    with patch.object(
        DeepEvalContextualPrecisionMetric,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
