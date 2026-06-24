"""Test metric registration system."""

import pytest

from aidial_admin_evaluation_metrics.metrics.common.registry import (
    MetricsRegistry,
)
from aidial_admin_evaluation_metrics.metrics.simple.exact_match import (
    ExactMatchMetric,
)
from aidial_admin_evaluation_metrics.metrics.simple.regex_match import (
    RegexMatchMetric,
)


def test_empty_registry_initialization():
    """Test that a new registry starts empty."""
    registry = MetricsRegistry()
    assert len(registry) == 0
    assert registry.get_all_metrics() == {}
    assert registry.get_all_descriptions() == []
    assert registry.get_metric_names() == set()


def test_register_single_metric():
    """Test registering a single metric."""
    registry = MetricsRegistry([ExactMatchMetric()])

    assert len(registry) == 1
    assert "exact_match" in registry.get_metric_names()


def test_register_multiple_metrics():
    """Test registering multiple metrics."""
    registry = MetricsRegistry([ExactMatchMetric(), RegexMatchMetric()])

    assert len(registry) == 2
    assert registry.get_metric_names() == {"exact_match", "regex_match"}


def test_create_metric_by_name():
    """Test retrieving a metric instance by name."""
    registry = MetricsRegistry([ExactMatchMetric()])

    metric = registry.get_metric_instance("exact_match")
    assert metric is not None
    assert metric.name == "exact_match"


def test_get_all_metrics():
    """Test getting all metric instances as a dictionary."""
    registry = MetricsRegistry([ExactMatchMetric(), RegexMatchMetric()])

    all_metrics = registry.get_all_metrics()
    assert len(all_metrics) == 2
    assert "exact_match" in all_metrics
    assert "regex_match" in all_metrics
    assert all_metrics["exact_match"].name == "exact_match"
    assert all_metrics["regex_match"].name == "regex_match"


def test_get_all_metrics_returns_copy():
    """Test that get_all_metrics returns a copy, not the internal dict."""
    registry = MetricsRegistry([ExactMatchMetric()])

    metrics1 = registry.get_all_metrics()
    metrics2 = registry.get_all_metrics()

    assert metrics1 is not metrics2
    assert metrics1 == metrics2


def test_get_all_descriptions():
    """Test getting all metric descriptions."""
    registry = MetricsRegistry([ExactMatchMetric(), RegexMatchMetric()])

    descriptions = registry.get_all_descriptions()
    assert len(descriptions) == 2
    description_names = {desc.name for desc in descriptions}
    assert description_names == {"exact_match", "regex_match"}


def test_get_all_descriptions_returns_copy():
    """Test that get_all_descriptions returns a copy, not the internal list."""
    registry = MetricsRegistry([ExactMatchMetric()])

    desc1 = registry.get_all_descriptions()
    desc2 = registry.get_all_descriptions()

    assert desc1 is not desc2
    assert desc1 == desc2


def test_get_metric_names():
    """Test getting all metric names as a set."""
    registry = MetricsRegistry([ExactMatchMetric(), RegexMatchMetric()])

    names = registry.get_metric_names()
    assert isinstance(names, set)
    assert names == {"exact_match", "regex_match"}


def test_register_returns_instance():
    """Test that _register stores the instance and it is retrievable."""
    registry = MetricsRegistry()
    instance = ExactMatchMetric()

    registry._register(instance)
    assert registry.get_metric_instance("exact_match") is instance


def test_register_duplicate_metric_raises_error():
    """Test that registering a metric with duplicate name raises ValueError."""
    # Try to register the same metric twice in the constructor
    with pytest.raises(
        ValueError,
        match=r"Metric with name 'exact_match' is already registered\. "
        r"Cannot register ExactMatchMetric\.",
    ):
        MetricsRegistry([ExactMatchMetric(), ExactMatchMetric()])
