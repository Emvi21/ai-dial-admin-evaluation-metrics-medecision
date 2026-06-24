"""Unit tests for ContextRecallMetric."""

from unittest.mock import patch

import pytest
from ragas.metrics.collections import ContextRecall as RagasContextRecall

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.metrics.ragas.context_recall import (
    ContextRecallMetric,
)
from tests.metrics.ragas.helpers import make_mock_dial_factory, mock_ascore


@pytest.mark.asyncio
async def test_context_recall_score():
    metric = ContextRecallMetric(make_mock_dial_factory(), RagasGroupSettings())
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        reference="The Eiffel Tower was completed in 1889.",
        retrieved_contexts=[
            "The Eiffel Tower was completed in 1889 for the World's Fair.",
        ],
    )

    with patch.object(
        RagasContextRecall,
        "ascore",
        new=mock_ascore(
            1.0, "All reference statements are attributed to context."
        ),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(1.0)
    assert (
        result.score.details.reason
        == "All reference statements are attributed to context."
    )


@pytest.mark.asyncio
async def test_context_recall_error():
    metric = ContextRecallMetric(make_mock_dial_factory(), RagasGroupSettings())
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        reference="The Eiffel Tower was completed in 1889.",
        retrieved_contexts=["Some context."],
    )

    with patch.object(
        RagasContextRecall, "ascore", side_effect=RuntimeError("LLM error")
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
