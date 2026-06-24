"""Tests for the set_comparison metric."""

import pytest
from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app_config import DialConfig
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory
from aidial_admin_evaluation_metrics.metrics import create_metrics_registry

registry = create_metrics_registry(
    create_dial_factory(
        DialConfig(dial_url="http://test", dial_api_key=SecretStr("test"))
    )
)


def test_set_comparison_perfect_match():
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(actual=["a", "b", "c"], ground_truth=["a", "b", "c"]),
    )
    assert result.precision.value == 1.0
    assert result.recall.value == 1.0
    assert result.precision.details["tp"] == 3
    assert result.precision.details["fp"] == 0
    assert result.precision.details["fn"] == 0


def test_set_comparison_partial_match():
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(actual=["a", "b", "x"], ground_truth=["a", "b", "c"]),
    )
    assert result.precision.value == pytest.approx(2 / 3)
    assert result.recall.value == pytest.approx(2 / 3)
    assert result.precision.details["tp"] == 2
    assert result.precision.details["fp"] == 1
    assert result.precision.details["fn"] == 1
    assert sorted(result.precision.details["true_positives"]) == ["a", "b"]
    assert result.precision.details["false_positives"] == ["x"]
    assert result.precision.details["false_negatives"] == ["c"]


def test_set_comparison_no_match():
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config, metric.Input(actual=["x", "y"], ground_truth=["a", "b"])
    )
    assert result.precision.value == 0.0
    assert result.recall.value == 0.0


def test_set_comparison_empty_actual():
    """Empty actual: no FP so precision=1.0, but all GT are FN so recall=0.0."""
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config, metric.Input(actual=[], ground_truth=["a", "b"])
    )
    assert result.precision.value == 1.0
    assert result.recall.value == 0.0


def test_set_comparison_empty_ground_truth():
    """Empty GT: no FN so recall=1.0, but all actual are FP so precision=0.0."""
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config, metric.Input(actual=["a", "b"], ground_truth=[])
    )
    assert result.precision.value == 0.0
    assert result.recall.value == 1.0


def test_set_comparison_both_empty():
    """Both empty: vacuously perfect."""
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(config, metric.Input(actual=[], ground_truth=[]))
    assert result.precision.value == 1.0
    assert result.recall.value == 1.0


def test_set_comparison_duplicates_treated_as_set():
    """Duplicate entries in input should be deduplicated."""
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(actual=["a", "a", "b"], ground_truth=["a", "b", "b"]),
    )
    assert result.precision.value == 1.0
    assert result.recall.value == 1.0
    assert result.precision.details["tp"] == 2


def test_set_comparison_details_shared_between_precision_and_recall():
    """Both output fields carry identical details."""
    metric = registry.get_metric_instance("set_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config, metric.Input(actual=["a", "x"], ground_truth=["a", "b"])
    )
    assert result.precision.details == result.recall.details
