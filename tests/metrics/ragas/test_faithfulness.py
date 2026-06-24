"""Unit tests for FaithfulnessMetric."""

from unittest.mock import patch

import pytest
from ragas.metrics.collections import Faithfulness as RagasFaithfulness

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.metrics.ragas.faithfulness import (
    FaithfulnessMetric,
)
from tests.metrics.ragas.helpers import make_mock_dial_factory, mock_ascore


@pytest.mark.asyncio
async def test_faithfulness_score():
    metric = FaithfulnessMetric(make_mock_dial_factory(), RagasGroupSettings())
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="When was the Eiffel Tower built?",
        response="The Eiffel Tower was built in 1889.",
        retrieved_contexts=[
            "The Eiffel Tower was constructed from 1887 to 1889."
        ],
    )

    with patch.object(
        RagasFaithfulness,
        "ascore",
        new=mock_ascore(1.0, "All claims supported by context."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(1.0)
    assert result.score.details.reason == "All claims supported by context."


@pytest.mark.asyncio
async def test_faithfulness_error():
    metric = FaithfulnessMetric(make_mock_dial_factory(), RagasGroupSettings())
    config = metric.Config(model="gemini-2.5-flash-lite")
    input_data = metric.Input(
        user_input="When was the Eiffel Tower built?",
        response="The Eiffel Tower was built in 1889.",
        retrieved_contexts=["Built 1887-1889."],
    )

    with patch.object(
        RagasFaithfulness, "ascore", side_effect=RuntimeError("LLM error")
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
