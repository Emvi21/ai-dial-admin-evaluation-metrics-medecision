"""Integration tests for RagEvalMetric with recorded LLM responses.

These tests use pytest-recording to record real LLM API calls and replay them.
See tests/cassettes.md for detailed documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.app_config import (
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.rag_eval import (
    RagEvalMetric,
)


@pytest.mark.integration
@pytest.mark.vcr
def test_rag_eval_direct_evaluation_with_real_llm(test_dial_config):
    """Test RagEvalMetric with real LLM - responses recorded/replayed."""
    llm_factory = create_dial_factory(test_dial_config)

    # Create RagEvalMetric with real LLM factory
    rag_eval_metric = RagEvalMetric(llm_factory, AidialRagEvalGroupSettings())
    config = rag_eval_metric.Config(
        matcher="citation", model="gemini-2.5-flash-lite"
    )
    input_data = rag_eval_metric.Input(
        question="What is the diameter of Earth?",
        ground_truth_answer="The diameter of the Earth is approximately 12,742 kilometers.",
        ground_truth_facts=[
            "The diameter of the Earth is approximately 12,742 kilometers"
        ],
        context=[
            "The Earth is the third planet from the Sun.",
            "The diameter of the Earth is approximately 12,742 kilometers, making it the fifth largest planet.",
        ],
        answer="Earth has a diameter of about 12,742 km.",
    )

    # This will make real LLM calls (or replay from cassette)
    result = rag_eval_metric._evaluate(config, input_data)

    # Retrieval outputs
    assert result.recall.type == "value"
    assert result.recall.value == 1.0

    # The fact was found in second chunk, so rank is 1 (0-based)
    assert result.recall.details.facts_ranks == [1]
    assert result.recall.details.context_relevance == [0, 1]
    assert len(result.recall.details.context_highlight) == 2
    assert all(
        len(highlight) > 0
        for highlight in result.recall.details.context_highlight
    )

    assert result.precision.type == "value"
    assert result.precision.value == 0.5

    assert result.f1.type == "value"
    assert result.f1.value == pytest.approx(2.0 / 3)

    assert result.mrr.type == "value"
    assert result.mrr.value == 0.5

    # Generation outputs
    assert result.ground_truth_to_answer.type == "value"
    assert result.ground_truth_to_answer.value == 1.0
    assert result.ground_truth_to_answer.details.json_explanation != ""
    assert result.ground_truth_to_answer.details.highlight != ""

    assert result.answer_to_ground_truth.type == "value"
    assert result.answer_to_ground_truth.value == 1.0
    assert result.answer_to_ground_truth.details.json_explanation != ""
    assert result.answer_to_ground_truth.details.highlight != ""

    assert result.context_to_answer.type == "value"
    assert result.context_to_answer.value == 1.0
    assert result.context_to_answer.details.json_explanation != ""
    assert result.context_to_answer.details.highlight != ""


@pytest.mark.integration
@pytest.mark.vcr
def test_rag_eval_evaluate_via_api(app_client):
    """Test RagEvalMetric via API with example data and real LLM."""
    request = {
        "metric_name": "aidial_rag_eval.rag_eval",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "question": "What is the diameter of Earth?",
            "ground_truth_answer": "The diameter of the Earth is approximately 12,742 kilometers.",
            "ground_truth_facts": [
                "The diameter of the Earth is approximately 12,742 kilometers"
            ],
            "context": [
                "The Earth is the third planet from the Sun.",
                "The diameter of the Earth is approximately 12,742 kilometers, making it the fifth largest planet.",
            ],
            "answer": "Earth has a diameter of about 12,742 km.",
            "documents": ["earth_facts.txt"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    result = response.json()
    assert result["metric_name"] == "aidial_rag_eval.rag_eval"

    from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.rag_eval import (
        RagEvalMetric,
    )

    parsed = RagEvalMetric.Output.model_validate(result["output"])

    # Retrieval fields
    assert parsed.recall.type == "value"
    assert parsed.recall.value == 1.0
    assert isinstance(parsed.recall.details.facts_ranks, list)
    assert len(parsed.recall.details.facts_ranks) == 1
    assert isinstance(parsed.recall.details.context_relevance, list)
    assert isinstance(parsed.recall.details.context_highlight, list)

    assert parsed.precision.type == "value"
    assert parsed.precision.value == 0.5

    assert parsed.f1.type == "value"
    assert parsed.f1.value == pytest.approx(2.0 / 3)

    assert parsed.mrr.type == "value"
    assert parsed.mrr.value == 0.5

    # Generation fields
    assert parsed.ground_truth_to_answer.type == "value"
    assert parsed.ground_truth_to_answer.value == 1.0
    assert parsed.ground_truth_to_answer.details.json_explanation != ""
    assert parsed.ground_truth_to_answer.details.highlight != ""

    assert parsed.answer_to_ground_truth.type == "value"
    assert parsed.answer_to_ground_truth.value == 1.0
    assert parsed.answer_to_ground_truth.details.json_explanation != ""
    assert parsed.answer_to_ground_truth.details.highlight != ""

    assert parsed.context_to_answer.type == "value"
    assert parsed.context_to_answer.value == 1.0
    assert parsed.context_to_answer.details.json_explanation != ""
    assert parsed.context_to_answer.details.highlight != ""


@pytest.mark.integration
@pytest.mark.vcr
def test_rag_eval_evaluate_via_api_partial(app_client):
    """Test RagEvalMetric with partially matching answer via API."""
    request = {
        "metric_name": "aidial_rag_eval.rag_eval",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "question": "What is the diameter of Earth?",
            "ground_truth_answer": "The diameter of the Earth is approximately 12,742 kilometers. The Earth has two moons.",
            "ground_truth_facts": [
                "The diameter of the Earth is approximately 12,742 kilometers",
                "The Earth is the third planet from the Sun",
                "The Earth has two moons",
            ],
            "context": [
                "The Earth is the third planet from the Sun.",
                "The diameter of the Earth is approximately 12,742 kilometers, making it the fifth largest planet.",
            ],
            "answer": "Earth has a diameter of about 12,742 km. It is a third planet. And it has a single moon.",
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    result = response.json()
    assert result["metric_name"] == "aidial_rag_eval.rag_eval"

    parsed = RagEvalMetric.Output.model_validate(result["output"])

    # Retrieval
    # The first fact is found in the second chunk (rank 1)
    # The second fact is found in the first chunk (rank 0)
    # The third fact is not found (-1) in the context
    assert parsed.recall.type == "value"
    assert parsed.recall.value == pytest.approx(2.0 / 3)
    assert parsed.recall.details.facts_ranks == [1, 0, -1]
    assert parsed.recall.details.context_relevance == [1, 1]
    assert parsed.precision.type == "value"
    assert parsed.precision.value == pytest.approx(1.0)
    assert parsed.f1.type == "value"
    assert parsed.f1.value == pytest.approx(0.8)
    assert parsed.mrr.type == "value"
    assert parsed.mrr.value == 0.5

    # Generation
    # Only the diameter statement is entailed from the ground truth
    assert parsed.ground_truth_to_answer.type == "value"
    assert parsed.ground_truth_to_answer.value == pytest.approx(1.0 / 3)
    assert parsed.ground_truth_to_answer.details.json_explanation != ""
    assert parsed.ground_truth_to_answer.details.highlight != ""

    # Answer to ground truth: one entailment, one contradiction
    assert parsed.answer_to_ground_truth.type == "value"
    assert parsed.answer_to_ground_truth.value == pytest.approx(0.5)
    assert parsed.answer_to_ground_truth.details.json_explanation != ""
    assert parsed.answer_to_ground_truth.details.highlight != ""

    # Context to answer: two entailed statements out of three
    assert parsed.context_to_answer.type == "value"
    assert parsed.context_to_answer.value == pytest.approx(2.0 / 3)
    assert parsed.context_to_answer.details.json_explanation != ""
    assert parsed.context_to_answer.details.highlight != ""
