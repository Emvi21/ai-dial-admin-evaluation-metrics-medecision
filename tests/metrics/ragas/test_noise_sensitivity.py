"""Unit tests for NoiseSensitivityMetric."""

from unittest.mock import patch

import pytest
from ragas.metrics.collections import NoiseSensitivity as RagasNoiseSensitivity

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.metrics.ragas.noise_sensitivity import (
    NoiseSensitivityMetric,
)
from tests.metrics.ragas.helpers import make_mock_dial_factory, mock_ascore


@pytest.mark.asyncio
async def test_noise_sensitivity_relevant_mode():
    metric = NoiseSensitivityMetric(
        make_mock_dial_factory(), RagasGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", mode="relevant")
    input_data = metric.Input(
        user_input="When was the Eiffel Tower built?",
        response="The Eiffel Tower was built in 1889.",
        reference="The Eiffel Tower was constructed from 1887 to 1889.",
        retrieved_contexts=[
            "The Eiffel Tower was constructed from 1887 to 1889."
        ],
    )

    with patch.object(
        RagasNoiseSensitivity,
        "ascore",
        new=mock_ascore(0.0, "No errors introduced by relevant context."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.0)
    assert (
        result.score.details.reason
        == "No errors introduced by relevant context."
    )


@pytest.mark.asyncio
async def test_noise_sensitivity_irrelevant_mode():
    metric = NoiseSensitivityMetric(
        make_mock_dial_factory(), RagasGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", mode="irrelevant")
    input_data = metric.Input(
        user_input="When was the Eiffel Tower built?",
        response="The Eiffel Tower was built in 1889.",
        reference="The Eiffel Tower was constructed from 1887 to 1889.",
        retrieved_contexts=[
            "Paris is the capital of France.",
            "The Eiffel Tower was constructed from 1887 to 1889.",
        ],
    )

    with patch.object(
        RagasNoiseSensitivity,
        "ascore",
        new=mock_ascore(0.1, "Minor error from irrelevant context."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_noise_sensitivity_error():
    metric = NoiseSensitivityMetric(
        make_mock_dial_factory(), RagasGroupSettings()
    )
    config = metric.Config(model="gemini-2.5-flash-lite", mode="relevant")
    input_data = metric.Input(
        user_input="When was the Eiffel Tower built?",
        response="The Eiffel Tower was built in 1889.",
        reference="The Eiffel Tower was constructed from 1887 to 1889.",
        retrieved_contexts=["Some context."],
    )

    with patch.object(
        RagasNoiseSensitivity, "ascore", side_effect=RuntimeError("LLM error")
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
