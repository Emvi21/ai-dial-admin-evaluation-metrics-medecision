"""Unit tests for AnswerRelevancyMetric."""

from unittest.mock import patch

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric as DeepEvalAnswerRelevancyMetric,
)

from aidial_admin_evaluation_metrics.app_config import DeepEvalGroupSettings
from aidial_admin_evaluation_metrics.metrics.deepeval.answer_relevancy import (
    AnswerRelevancyMetric,
)
from tests.metrics.deepeval.helpers import (
    make_mock_dial_factory,
    mock_a_measure,
)


@pytest.mark.asyncio
async def test_answer_relevancy_score():
    metric = AnswerRelevancyMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", threshold=0.5)
    input_data = metric.Input(
        input="What is AI?", actual_output="AI is Artificial Intelligence."
    )

    with patch.object(
        DeepEvalAnswerRelevancyMetric,
        "a_measure",
        new=mock_a_measure(0.9, "Relevant.", "Steps..."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.9)
    assert result.score.details.get("reason") == "Relevant."
    assert result.score.details.get("verbose_logs") == "Steps..."


@pytest.mark.asyncio
async def test_answer_relevancy_error():
    metric = AnswerRelevancyMetric(
        make_mock_dial_factory(), DeepEvalGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", threshold=0.5)
    input_data = metric.Input(input="What is AI?", actual_output="AI is great.")

    with patch.object(
        DeepEvalAnswerRelevancyMetric,
        "a_measure",
        side_effect=RuntimeError("LLM error"),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
