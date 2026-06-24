"""Unit tests for AnswerRelevancyMetric."""

from unittest.mock import patch

import pytest
from ragas.metrics.collections import AnswerRelevancy as RagasAnswerRelevancy

from aidial_admin_evaluation_metrics.app_config import RagasGroupSettings
from aidial_admin_evaluation_metrics.metrics.ragas.answer_relevancy import (
    AnswerRelevancyMetric,
)
from tests.metrics.ragas.helpers import make_mock_dial_factory, mock_ascore


@pytest.mark.asyncio
async def test_answer_relevancy_score():
    settings = RagasGroupSettings(embeddings_model="text-embedding-ada-002")
    metric = AnswerRelevancyMetric(
        make_mock_dial_factory(),
        settings,
    )
    config = metric.Config(
        model="gemini-2.5-flash-lite",
        embeddings_model="text-embedding-ada-002",
    )
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        response="The Eiffel Tower was completed in 1889.",
    )

    with patch.object(
        RagasAnswerRelevancy,
        "ascore",
        new=mock_ascore(0.95, "Response directly addresses the question."),
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "value"
    assert result.score.value == pytest.approx(0.95)
    assert (
        result.score.details.reason
        == "Response directly addresses the question."
    )


def test_answer_relevancy_no_embeddings_model(app_client):
    """embeddings_model does not allow None — API returns 422."""
    response = app_client.post(
        "/evaluate",
        json={
            "metric_name": "ragas.answer_relevancy",
            "config": {
                "model": "gemini-2.5-flash-lite",
                "embeddings_model": None,
            },
            "input": {
                "user_input": "What year was the Eiffel Tower completed?",
                "response": "The Eiffel Tower was completed in 1889.",
            },
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_answer_relevancy_error():
    settings = RagasGroupSettings(embeddings_model="text-embedding-ada-002")
    metric = AnswerRelevancyMetric(
        make_mock_dial_factory(),
        settings,
    )
    config = metric.Config(
        model="gemini-2.5-flash-lite",
        embeddings_model="text-embedding-ada-002",
    )
    input_data = metric.Input(
        user_input="What year was the Eiffel Tower completed?",
        response="The Eiffel Tower was completed in 1889.",
    )

    with patch.object(
        RagasAnswerRelevancy, "ascore", side_effect=RuntimeError("LLM error")
    ):
        result = await metric.evaluate_async(config, input_data)

    assert result.score.type == "error"
    assert "LLM error" in result.score.message
