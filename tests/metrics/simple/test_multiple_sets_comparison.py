"""Tests for the multiple_sets_comparison metric."""

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


# ---------------------------------------------------------------------------
# precision / recall
# ---------------------------------------------------------------------------


def test_multiple_sets_comparison_perfect_match():
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"fruits": ["apple", "banana"], "colors": ["red"]},
            ground_truth={"fruits": ["apple", "banana"], "colors": ["red"]},
        ),
    )
    assert result.precision.value == 1.0
    assert result.recall.value == 1.0
    assert result.precision.details["by_key"]["fruits"]["precision"] == 1.0
    assert result.precision.details["by_key"]["colors"]["recall"] == 1.0


def test_multiple_sets_comparison_partial_match():
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k1": ["a", "b"], "k2": ["x"]},
            ground_truth={"k1": ["a", "c"], "k2": ["x", "y"]},
        ),
    )
    # k1: tp=1, fp=1, fn=1 → prec=0.5, rec=0.5
    # k2: tp=1, fp=0, fn=1 → prec=1.0, rec=0.5
    # avg precision = (0.5 + 1.0) / 2 = 0.75
    # avg recall = (0.5 + 0.5) / 2 = 0.5
    assert result.precision.value == pytest.approx(0.75)
    assert result.recall.value == pytest.approx(0.5)


def test_multiple_sets_comparison_missing_key_in_actual():
    """Key present in ground_truth but not in actual → empty actual set."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={},
            ground_truth={"k": ["a", "b"]},
        ),
    )
    # empty actual: prec=1.0, rec=0.0
    assert result.precision.value == 1.0
    assert result.recall.value == 0.0
    assert result.precision.details["by_key"]["k"]["tp"] == 0
    assert result.precision.details["by_key"]["k"]["fn"] == 2


def test_multiple_sets_comparison_missing_key_in_ground_truth():
    """Key present in actual but not in ground_truth → empty GT set."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k": ["a", "b"]},
            ground_truth={},
        ),
    )
    # empty GT: prec=0.0, rec=1.0
    assert result.precision.value == 0.0
    assert result.recall.value == 1.0


def test_multiple_sets_comparison_both_empty():
    """Both dicts empty: vacuously perfect."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(config, metric.Input(actual={}, ground_truth={}))
    assert result.precision.value == 1.0
    assert result.recall.value == 1.0
    assert result.precision.details == {"by_key": {}}


def test_multiple_sets_comparison_details_shared():
    """precision and recall fields carry identical details."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k": ["a"]},
            ground_truth={"k": ["b"]},
        ),
    )
    assert result.precision.details == result.recall.details


def test_multiple_sets_comparison_averaging():
    """Verify macro-average is correct with 3 keys."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k1": ["a"], "k2": [], "k3": ["a", "b"]},
            ground_truth={"k1": ["a"], "k2": ["x"], "k3": ["a"]},
        ),
    )
    # k1: tp=1, fp=0, fn=0 → prec=1.0, rec=1.0
    # k2: tp=0, fp=0, fn=1 → prec=1.0 (empty actual), rec=0.0
    # k3: tp=1, fp=1, fn=0 → prec=0.5, rec=1.0
    # avg prec = (1.0 + 1.0 + 0.5) / 3 = 5/6
    # avg rec  = (1.0 + 0.0 + 1.0) / 3 = 2/3
    assert result.precision.value == pytest.approx(5 / 6)
    assert result.recall.value == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# include_item_details config option
# ---------------------------------------------------------------------------


def test_multiple_sets_comparison_include_item_details_false():
    """By default, item lists are not included per key."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config(include_item_details=False)
    result = metric._evaluate(
        config,
        metric.Input(actual={"k": ["a"]}, ground_truth={"k": ["b"]}),
    )
    key_entry = result.precision.details["by_key"]["k"]
    assert "true_positives" not in key_entry
    assert "false_positives" not in key_entry
    assert "false_negatives" not in key_entry


def test_multiple_sets_comparison_include_item_details_true():
    """When include_item_details=True, item lists appear per key."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config(include_item_details=True)
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k": ["a", "x"]},
            ground_truth={"k": ["a", "b"]},
        ),
    )
    key_entry = result.precision.details["by_key"]["k"]
    assert key_entry["true_positives"] == ["a"]
    assert key_entry["false_positives"] == ["x"]
    assert key_entry["false_negatives"] == ["b"]


# ---------------------------------------------------------------------------
# precision_soft
# ---------------------------------------------------------------------------


def test_precision_soft_extra_key_in_actual_not_penalized():
    """FPs in keys absent from ground_truth must not count against precision_soft."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    # A: actual={x,z}, GT={x,y}  → tp=1, fp=1 (z, in allowed)
    # C: actual={q},   GT=∅       → extra key, fp=1 but NOT in allowed_keys
    # allowed_keys = {A}
    # tp_total=1, fp_soft=1 (only key A counts) → precision_soft = 1/2 = 0.5
    # precision (macro): A→0.5, C→0.0 → avg=0.25
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"A": ["x", "z"], "C": ["q"]},
            ground_truth={"A": ["x", "y"]},
        ),
    )
    assert result.precision.value == pytest.approx(
        0.25
    )  # macro-avg penalizes C
    assert result.precision_soft.value == pytest.approx(0.5)  # C's FP ignored


def test_precision_soft_definition_example():
    """Reproduces the exact example from the precision_soft definition."""
    # target_map = {A: {x,y}, B: {p}}
    # actual_map = {A: {x,z}, C: {q}}
    # allowed_keys = {A, B}
    # TP_total = |{x}| = 1
    # FP_soft  = |{z}| (key A, allowed) = 1  [C is not in allowed_keys]
    # precision_soft = 1 / (1+1) = 0.5
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"A": ["x", "z"], "C": ["q"]},
            ground_truth={"A": ["x", "y"], "B": ["p"]},
        ),
    )
    assert result.precision_soft.value == pytest.approx(0.5)
    assert result.precision_soft.details["tp_total"] == 1
    assert result.precision_soft.details["fp_soft"] == 1
    assert sorted(result.precision_soft.details["allowed_keys"]) == ["A", "B"]


def test_precision_soft_empty_ground_truth_nonempty_actual():
    """Empty ground_truth + non-empty actual → precision_soft=0.0."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(actual={"k": ["a"]}, ground_truth={}),
    )
    assert result.precision_soft.value == 0.0


def test_precision_soft_empty_ground_truth_empty_actual():
    """Both empty → precision_soft=1.0."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(config, metric.Input(actual={}, ground_truth={}))
    assert result.precision_soft.value == 1.0


def test_precision_soft_actual_only_extra_keys():
    """Actual has only keys not in ground_truth → no penalizable FPs → precision_soft=1.0."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"extra": ["x", "y"]},
            ground_truth={"k": ["a"]},
        ),
    )
    # key "extra" is not in allowed_keys → fp_soft=0; key "k" has tp=0 fn=1
    # tp_total=0, fp_soft=0, allowed_keys={"k"} → precision_soft=1.0 (policy)
    assert result.precision_soft.value == 1.0


def test_precision_soft_perfect_match():
    """Perfect match → precision_soft=1.0 with correct detail fields."""
    metric = registry.get_metric_instance("multiple_sets_comparison")
    assert metric is not None
    config = metric.Config()
    result = metric._evaluate(
        config,
        metric.Input(
            actual={"k": ["a", "b"]},
            ground_truth={"k": ["a", "b"]},
        ),
    )
    assert result.precision_soft.value == 1.0
    assert result.precision_soft.details["tp_total"] == 2
    assert result.precision_soft.details["fp_soft"] == 0
    assert result.precision_soft.details["allowed_keys"] == ["k"]
