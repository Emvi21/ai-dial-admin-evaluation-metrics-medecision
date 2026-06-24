"""Unit tests for ContextPrecisionMetric."""

from unittest.mock import patch

import pytest
from ragas.metrics.collections import (
    ContextPrecisionWithReference as RagasContextPrecision,
)

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.metrics.ragas.context_precision import (
    ContextPrecisionMetric,
)
from tests.metrics.ragas.helpers import make_mock_dial_factory, mock_ascore


@pytest.mark.asyncio
async def test_context_precision_score():
    metric = ContextPrecisionMetric(
        make_mock_dial_factory(), RagasGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        reference="The Eiffel Tower was completed in 1889.",
        retrieved_contexts=[
            "The Eiffel Tower was completed in 1889 for the World's Fair.",
            "Paris is the capital of France.",
        ],
    )

    with patch.object(
        RagasContextPrecision,
        "ascore",
        new=mock_ascore(0.5, "First context is relevant, second is not."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.5)
    assert "relevant" in (result.score.details.reason or "")


@pytest.mark.asyncio
async def test_context_precision_error():
    metric = ContextPrecisionMetric(
        make_mock_dial_factory(), RagasGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        reference="The Eiffel Tower was completed in 1889.",
        retrieved_contexts=["Some context."],
    )

    with patch.object(
        RagasContextPrecision, "ascore", side_effect=RuntimeError("LLM error")
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
