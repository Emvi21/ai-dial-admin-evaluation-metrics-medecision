"""Tests for retrieval metric integration."""

import pandas as pd
import pytest

from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.retrieval import (
    RetrievalMetric,
)
from aidial_admin_evaluation_metrics.metrics.aidial_rag_eval.retrieval_output import (
    RecallField,
    extract_recall,
    extract_retrieval_metric,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)


def test_metric_basic_evaluation():
    """Test that metric can be instantiated and perform basic evaluation."""
    retrieval_metric = RetrievalMetric()
    config = retrieval_metric.Config()
    input_data = retrieval_metric.Input(
        ground_truth_facts=["fact1"],
        retrieved_context=["This has fact1"],
    )
    result = retrieval_metric._evaluate(config, input_data)

    # Verify fields are not errors
    assert not isinstance(result.recall, MetricError)
    assert not isinstance(result.precision, MetricError)
    assert not isinstance(result.f1, MetricError)
    assert not isinstance(result.mrr, MetricError)

    # Verify metrics are in valid range
    assert 0.0 <= result.recall.value <= 1.0
    assert 0.0 <= result.precision.value <= 1.0
    assert 0.0 <= result.f1.value <= 1.0
    assert 0.0 <= result.mrr.value <= 1.0


def test_retrieval_evaluate_via_api(app_client):
    """Test retrieval metric evaluation through /evaluate endpoint."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["fact1", "fact2"],
            "retrieved_context": ["has fact1", "unrelated"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.5,
                "details": {
                    "facts_ranks": [0, -1],
                    "context_relevance": [1, 0],
                    "context_highlight": [
                        '{"match": [{"text": "has ", "facts": []}, {"text": "fact1", "facts": [0]}]}',
                        '{"match": [{"text": "unrelated", "facts": []}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 0.5, "details": {}},
            "f1": {"type": "value", "value": 0.5, "details": {}},
            "mrr": {"type": "value", "value": 0.5, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_example_via_api(app_client):
    """Test retrieval with example data from metric definition."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": [
                "The diameter of the Earth is approximately 12,742 kilometers.",
                "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
            ],
            "retrieved_context": [
                "The Earth, our home planet, is the third planet from the sun.",
                "It's the only planet known to have an atmosphere containing free oxygen and oceans of liquid water on its surface. The diameter of the Earth is approximately 12,742 kilometers.",
                "The Pacific Ocean is extending from the Arctic Ocean in the north to the Southern Ocean in the south.",
            ],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.5,
                "details": {
                    "facts_ranks": [1, -1],
                    "context_relevance": [0, 1, 0],
                    "context_highlight": [
                        '{"match": [{"text": "The Earth, our home planet, is the third planet from the sun.", "facts": []}]}',
                        '{"match": [{"text": "It\'s the only planet known to have an atmosphere containing free oxygen and oceans of liquid water on its surface. ", "facts": []}, {"text": "The diameter of the Earth is approximately 12,742 kilometers.", "facts": [0]}]}',
                        '{"match": [{"text": "The Pacific Ocean is extending from the Arctic Ocean in the north to the Southern Ocean in the south.", "facts": []}]}',
                    ],
                },
            },
            "precision": {
                "type": "value",
                "value": 0.3333333333333333,
                "details": {},
            },
            "f1": {"type": "value", "value": 0.4, "details": {}},
            "mrr": {"type": "value", "value": 0.25, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_perfect_match_via_api(app_client):
    """Test retrieval with perfect match scenario."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["fact1"],
            "retrieved_context": ["This contains fact1"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 1.0,
                "details": {
                    "facts_ranks": [0],
                    "context_relevance": [1],
                    "context_highlight": [
                        '{"match": [{"text": "This contains ", "facts": []}, {"text": "fact1", "facts": [0]}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 1.0, "details": {}},
            "f1": {"type": "value", "value": 1.0, "details": {}},
            "mrr": {"type": "value", "value": 1.0, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_no_match_via_api(app_client):
    """Test retrieval when no facts are found."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["specific_fact"],
            "retrieved_context": ["unrelated content", "more unrelated"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.0,
                "details": {
                    "facts_ranks": [-1],
                    "context_relevance": [0, 0],
                    "context_highlight": [
                        '{"match": [{"text": "unrelated content", "facts": []}]}',
                        '{"match": [{"text": "more unrelated", "facts": []}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 0.0, "details": {}},
            "f1": {"type": "value", "value": 0.0, "details": {}},
            "mrr": {"type": "value", "value": 0.0, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_empty_context(app_client):
    """Test retrieval with empty context list."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["fact1"],
            "retrieved_context": [],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.0,
                "details": {
                    "facts_ranks": [-1],
                    "context_relevance": [],
                    "context_highlight": [],
                },
            },
            "precision": {"type": "value", "value": 0.0, "details": {}},
            "f1": {"type": "value", "value": 0.0, "details": {}},
            "mrr": {"type": "value", "value": 0.0, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_json_serialization(app_client):
    """Test that response is properly JSON-serialized (no numpy types)."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["fact1", "fact2"],
            "retrieved_context": ["fact1", "other"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.5,
                "details": {
                    "facts_ranks": [0, -1],
                    "context_relevance": [1, 0],
                    "context_highlight": [
                        '{"match": [{"text": "fact1", "facts": [0]}]}',
                        '{"match": [{"text": "other", "facts": []}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 0.5, "details": {}},
            "f1": {"type": "value", "value": 0.5, "details": {}},
            "mrr": {"type": "value", "value": 0.5, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_details_list_lengths(app_client):
    """Test that details lists have correct lengths."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["fact1", "fact2", "fact3"],
            "retrieved_context": ["context1", "context2"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 0.0,
                "details": {
                    "facts_ranks": [-1, -1, -1],
                    "context_relevance": [0, 0],
                    "context_highlight": [
                        '{"match": [{"text": "context1", "facts": []}]}',
                        '{"match": [{"text": "context2", "facts": []}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 0.0, "details": {}},
            "f1": {"type": "value", "value": 0.0, "details": {}},
            "mrr": {"type": "value", "value": 0.0, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_citation_matcher_via_api(app_client):
    """Test retrieval with explicit citation matcher config (fuzzy matching)."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {"matcher": "citation"},
        "input": {
            "ground_truth_facts": ["FACT1"],
            "retrieved_context": ["This has fact1 inside"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    output = response.json()["output"]
    # Citation matcher does fuzzy substring matching (case-insensitive),
    # so "FACT1" matches "fact1" even when it's within a larger text
    assert output["recall"]["value"] == 1.0
    assert output["precision"]["value"] == 1.0


def test_retrieval_exact_matcher_via_api(app_client):
    """Test retrieval with exact matcher (requires entire context chunk to equal fact)."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {"matcher": "exact"},
        "input": {
            "ground_truth_facts": ["fact1"],
            "retrieved_context": ["fact1", "other"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    expected_response = {
        "metric_name": "aidial_rag_eval.retrieval",
        "output": {
            "recall": {
                "type": "value",
                "value": 1.0,
                "details": {
                    "facts_ranks": [0],
                    "context_relevance": [1, 0],
                    "context_highlight": [
                        '{"match": [{"text": "fact1", "facts": [0]}]}',
                        '{"match": [{"text": "other", "facts": []}]}',
                    ],
                },
            },
            "precision": {"type": "value", "value": 0.5, "details": {}},
            "f1": {"type": "value", "value": 0.6666666666666666, "details": {}},
            "mrr": {"type": "value", "value": 1.0, "details": {}},
        },
    }
    actual_response = response.json()
    assert actual_response == expected_response


def test_retrieval_exact_matcher_no_match_via_api(app_client):
    """Test that exact matcher fails when fact is only a substring of context chunk."""
    request = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {"matcher": "exact"},
        "input": {
            "ground_truth_facts": ["fact1"],
            "retrieved_context": ["This has fact1 inside"],
        },
    }

    response = app_client.post("/evaluate", json=request)
    assert response.status_code == 200

    output = response.json()["output"]
    # Exact matcher requires entire context chunk to equal the fact, not just contain it
    assert output["recall"]["value"] == 0.0
    assert output["precision"]["value"] == 0.0


def test_retrieval_default_matcher_is_citation(app_client):
    """Test that default matcher (when not specified) is citation."""
    # Request without matcher config
    request_no_config = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {},
        "input": {
            "ground_truth_facts": ["FACT1"],
            "retrieved_context": ["This has fact1 inside"],
        },
    }

    # Request with explicit citation matcher
    request_with_citation = {
        "metric_name": "aidial_rag_eval.retrieval",
        "config": {"matcher": "citation"},
        "input": {
            "ground_truth_facts": ["FACT1"],
            "retrieved_context": ["This has fact1 inside"],
        },
    }

    response_no_config = app_client.post("/evaluate", json=request_no_config)
    response_with_citation = app_client.post(
        "/evaluate", json=request_with_citation
    )

    assert response_no_config.status_code == 200
    assert response_with_citation.status_code == 200

    # Both should produce the same results (fuzzy matching)
    assert response_no_config.json() == response_with_citation.json()
    assert response_no_config.json()["output"]["recall"]["value"] == 1.0


# ---------------------------------------------------------------------------
# Unit tests for extract_retrieval_metric and extract_recall
# ---------------------------------------------------------------------------

_RECALL_ROW_DICT = {
    "facts_ranks": pd.array([-1]),
    "context_relevance": pd.array([0]),
    "context_highlight": pd.array(['{"match": []}']),
}


def _recall_series(value):
    """Build a pd.Series row with the given recall value."""
    return pd.Series(
        {
            "recall": value,
            "facts_ranks": pd.array([-1]),
            "context_relevance": pd.array([0]),
            "context_highlight": pd.array(['{"match": []}']),
        }
    )


# --- extract_retrieval_metric -----------------------------------------------


@pytest.mark.parametrize("name", ["precision", "f1", "mrr"])
def test_extract_retrieval_metric_valid_dict(name):
    row = {name: 0.75}
    result = extract_retrieval_metric(row, name)
    assert isinstance(result, MetricOutputField)
    assert result.value == 0.75


@pytest.mark.parametrize("name", ["precision", "f1", "mrr"])
def test_extract_retrieval_metric_valid_series(name):
    row = pd.Series({name: 0.5})
    result = extract_retrieval_metric(row, name)
    assert isinstance(result, MetricOutputField)
    assert result.value == 0.5


@pytest.mark.parametrize("name", ["precision", "f1", "mrr"])
def test_extract_retrieval_metric_nan_returns_error(name):
    row = {name: float("nan")}
    result = extract_retrieval_metric(row, name)
    assert isinstance(result, MetricError)


@pytest.mark.parametrize("name", ["precision", "f1", "mrr"])
def test_extract_retrieval_metric_inf_returns_error(name):
    row = {name: float("inf")}
    result = extract_retrieval_metric(row, name)
    assert isinstance(result, MetricError)


@pytest.mark.parametrize("name", ["precision", "f1", "mrr"])
def test_extract_retrieval_metric_non_numeric_returns_error(name):
    row = {name: "bad"}
    result = extract_retrieval_metric(row, name)
    assert isinstance(result, MetricError)


# --- extract_recall ----------------------------------------------------------


def test_extract_recall_valid_dict():
    row = {"recall": 0.5, **_RECALL_ROW_DICT}
    result = extract_recall(row)
    assert isinstance(result, RecallField)
    assert result.value == 0.5


def test_extract_recall_valid_series():
    result = extract_recall(_recall_series(1.0))
    assert isinstance(result, RecallField)
    assert result.value == 1.0


def test_extract_recall_nan_returns_error():
    row = {"recall": float("nan"), **_RECALL_ROW_DICT}
    result = extract_recall(row)
    assert isinstance(result, MetricError)


def test_extract_recall_inf_returns_error():
    row = {"recall": float("inf"), **_RECALL_ROW_DICT}
    result = extract_recall(row)
    assert isinstance(result, MetricError)


def test_extract_recall_non_numeric_returns_error():
    row = {"recall": "bad", **_RECALL_ROW_DICT}
    result = extract_recall(row)
    assert isinstance(result, MetricError)
