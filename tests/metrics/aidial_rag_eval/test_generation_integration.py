"""Integration tests for GenerationMetric with recorded LLM responses.

These tests use pytest-recording to record real LLM API calls and replay them.
See tests/cassettes.md for detailed documentation.
"""

import pytest

from aidial_admin_evaluation_metrics.app_config import (
    AidialRagEvalGroupSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.generation import (
    GenerationMetric,
)


@pytest.mark.integration
@pytest.mark.vcr
def test_generation_direct_evaluation_with_real_llm(test_dial_config):
    """Test GenerationMetric with real LLM - responses recorded/replayed."""
    llm_factory = create_dial_factory(test_dial_config)

    generation_metric = GenerationMetric(
        llm_factory, AidialRagEvalGroupSettings()
    )
    config = generation_metric.Config(model="gemini-2.5-flash-lite")
    input_data = generation_metric.Input(
        question="What is the diameter of Earth?",
        ground_truth_answer="The diameter of the Earth is approximately 12,742 kilometers.",
        context=[
            "The Earth is the third planet from the Sun.",
            "The diameter of the Earth is approximately 12,742 kilometers, making it the fifth largest planet.",
        ],
        answer="Earth has a diameter of about 12,742 km.",
    )

    # This will make real LLM calls (or replay from cassette)
    result = generation_metric._evaluate(config, input_data)

    print(result)
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
def test_generation_evaluate_via_api(app_client):
    """Test GenerationMetric with example data via API with real LLM."""
    request = {
        "metric_name": "aidial_rag_eval.generation",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "question": "What is the diameter of Earth?",
            "ground_truth_answer": "The diameter of the Earth is approximately 12,742 kilometers.",
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
    assert result["metric_name"] == "aidial_rag_eval.generation"

    parsed_result = GenerationMetric.Output.model_validate(result["output"])

    assert parsed_result.ground_truth_to_answer.type == "value"
    assert parsed_result.ground_truth_to_answer.value == 1.0
    assert parsed_result.ground_truth_to_answer.details.json_explanation != ""
    assert parsed_result.ground_truth_to_answer.details.highlight != ""

    assert parsed_result.answer_to_ground_truth.type == "value"
    assert parsed_result.answer_to_ground_truth.value == 1.0
    assert parsed_result.answer_to_ground_truth.details.json_explanation != ""
    assert parsed_result.answer_to_ground_truth.details.highlight != ""

    assert parsed_result.context_to_answer.type == "value"
    assert parsed_result.context_to_answer.value == 1.0
    assert parsed_result.context_to_answer.details.json_explanation != ""
    assert parsed_result.context_to_answer.details.highlight != ""


@pytest.mark.integration
@pytest.mark.vcr
def test_generation_evaluate_via_api_partial(app_client):
    """Test GenerationMetric with example data via API with real LLM."""
    request = {
        "metric_name": "aidial_rag_eval.generation",
        "config": {"model": "gemini-2.5-flash-lite"},
        "input": {
            "question": "What is the diameter of Earth?",
            "ground_truth_answer": "The diameter of the Earth is approximately 12,742 kilometers. The Earth has two moons.",
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
    assert result["metric_name"] == "aidial_rag_eval.generation"

    parsed_result = GenerationMetric.Output.model_validate(result["output"])

    # Only the statement about the diameter is entailed from the ground truth
    # the third planet is not mentioned in the ground truth,
    # and the moon statement contradicts the ground truth
    assert parsed_result.ground_truth_to_answer.type == "value"
    assert parsed_result.ground_truth_to_answer.value == pytest.approx(1.0 / 3)
    assert parsed_result.ground_truth_to_answer.details.json_explanation != ""
    assert parsed_result.ground_truth_to_answer.details.highlight != ""

    # The diameter statement could be entailed from the answer,
    # but the number of moons statement contradicts the answer,
    assert parsed_result.answer_to_ground_truth.type == "value"
    assert parsed_result.answer_to_ground_truth.value == pytest.approx(0.5)
    assert parsed_result.answer_to_ground_truth.details.json_explanation != ""
    assert parsed_result.answer_to_ground_truth.details.highlight != ""

    # The diameter and being third planet are entailed from the context, but the moon statement is not
    assert parsed_result.context_to_answer.type == "value"
    assert parsed_result.context_to_answer.value == pytest.approx(2.0 / 3)
    assert parsed_result.context_to_answer.details.json_explanation != ""
    assert parsed_result.context_to_answer.details.highlight != ""
