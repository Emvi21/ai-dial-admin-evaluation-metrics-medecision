from typing import Annotated

from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)
from aidial_admin_evaluation_metrics.metrics.simple.set_comparison import (
    _compute_set_metrics,
)


def _per_key_entry(
    tp: int,
    fp: int,
    fn: int,
    precision: float,
    recall: float,
    true_positives: list[str],
    false_positives: list[str],
    false_negatives: list[str],
    include_item_details: bool,
) -> dict:
    entry: dict = {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
    if include_item_details:
        entry["true_positives"] = true_positives
        entry["false_positives"] = false_positives
        entry["false_negatives"] = false_negatives
    return entry


class MultipleSetsComparisonMetric(BaseMetric):
    name: str = "multiple_sets_comparison"
    display_name: str = "Multiple Sets Comparison"
    description: str = (
        "Compares actual and ground_truth dicts (key → list of strings) by matching sets "
        "per key and computing macro-averaged precision and recall. "
        "Keys missing in either dict are treated as empty sets. "
        "Precision = TP / (TP + FP), Recall = TP / (TP + FN) per key; "
        "final values are the macro-average across all keys. "
        "precision_soft is a relaxed micro-aggregated precision that excludes false positives "
        "from keys not present in ground_truth (allowed_keys = keys(ground_truth)): "
        "precision_soft = TP_total / (TP_total + FP_soft). "
        "Per-key breakdowns are included in the details of each output field. "
        "Enable include_item_details in config to also include item lists per key."
    )

    class Config(BaseModel):
        include_item_details: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "When True, include true_positives, false_positives, and "
                    "false_negatives item lists in the per-key details."
                ),
            ),
        ] = False

    class Input(BaseModel):
        actual: Annotated[
            dict[str, list[str]],
            Field(description="The actual output items grouped by key"),
        ]
        ground_truth: Annotated[
            dict[str, list[str]],
            Field(description="The expected correct items grouped by key"),
        ]

    class Output(BaseModel):
        precision: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description=(
                    "Macro-averaged precision across all keys. "
                    "Details contain per-key breakdown."
                ),
            ),
        ]
        recall: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description=(
                    "Macro-averaged recall across all keys. "
                    "Details contain per-key breakdown."
                ),
            ),
        ]
        precision_soft: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description=(
                    "Relaxed micro-aggregated precision. "
                    "False positives from keys absent in ground_truth are not penalized. "
                    "precision_soft = TP_total / (TP_total + FP_soft) where "
                    "FP_soft counts only FPs in keys present in ground_truth. "
                    "Details include tp_total, fp_soft, allowed_keys, and per-key breakdown."
                ),
            ),
        ]

    examples = [
        MetricExample(
            name="multiple_sets_comparison perfect match",
            description="Example where all keys match perfectly",
            input=Input(
                actual={
                    "fruits": ["apple", "banana"],
                    "colors": ["red", "blue"],
                },
                ground_truth={
                    "fruits": ["apple", "banana"],
                    "colors": ["red", "blue"],
                },
            ),
            expected_output=Output(
                precision=MetricOutputField(
                    value=1.0,
                    details={
                        "by_key": {
                            "colors": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                            "fruits": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                        }
                    },
                ),
                recall=MetricOutputField(
                    value=1.0,
                    details={
                        "by_key": {
                            "colors": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                            "fruits": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                        }
                    },
                ),
                precision_soft=MetricOutputField(
                    value=1.0,
                    details={
                        "tp_total": 4,
                        "fp_soft": 0,
                        "allowed_keys": ["colors", "fruits"],
                        "by_key": {
                            "colors": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                            "fruits": {
                                "fn": 0,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "tp": 2,
                            },
                        },
                    },
                ),
            ),
        ),
        MetricExample(
            name="multiple_sets_comparison partial match with missing key",
            description="Example with partial match and a key missing in actual",
            input=Input(
                actual={"fruits": ["apple", "cherry"]},
                ground_truth={"fruits": ["apple", "banana"], "colors": ["red"]},
            ),
            expected_output=Output(
                precision=MetricOutputField(
                    value=0.75,
                    details={
                        "by_key": {
                            "colors": {
                                "fn": 1,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 0.0,
                                "tp": 0,
                            },
                            "fruits": {
                                "fn": 1,
                                "fp": 1,
                                "precision": 0.5,
                                "recall": 0.5,
                                "tp": 1,
                            },
                        }
                    },
                ),
                recall=MetricOutputField(
                    value=0.25,
                    details={
                        "by_key": {
                            "colors": {
                                "fn": 1,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 0.0,
                                "tp": 0,
                            },
                            "fruits": {
                                "fn": 1,
                                "fp": 1,
                                "precision": 0.5,
                                "recall": 0.5,
                                "tp": 1,
                            },
                        }
                    },
                ),
                precision_soft=MetricOutputField(
                    value=0.5,
                    details={
                        "tp_total": 1,
                        "fp_soft": 1,
                        "allowed_keys": ["colors", "fruits"],
                        "by_key": {
                            "colors": {
                                "fn": 1,
                                "fp": 0,
                                "precision": 1.0,
                                "recall": 0.0,
                                "tp": 0,
                            },
                            "fruits": {
                                "fn": 1,
                                "fp": 1,
                                "precision": 0.5,
                                "recall": 0.5,
                                "tp": 1,
                            },
                        },
                    },
                ),
            ),
        ),
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        all_keys = sorted(set(input.actual) | set(input.ground_truth))
        allowed_keys = set(input.ground_truth.keys())

        if not all_keys:
            # Both dicts empty: vacuously perfect
            empty_soft_details: dict = {
                "tp_total": 0,
                "fp_soft": 0,
                "allowed_keys": [],
                "by_key": {},
            }
            return self.Output(
                precision=MetricOutputField(value=1.0, details={"by_key": {}}),
                recall=MetricOutputField(value=1.0, details={"by_key": {}}),
                precision_soft=MetricOutputField(
                    value=1.0, details=empty_soft_details
                ),
            )

        by_key: dict[str, dict] = {}
        precision_sum = 0.0
        recall_sum = 0.0
        tp_total = 0
        fp_soft = 0

        for key in all_keys:
            actual_items = input.actual.get(key, [])
            gt_items = input.ground_truth.get(key, [])
            prec, rec, cm = _compute_set_metrics(actual_items, gt_items)

            by_key[key] = _per_key_entry(
                tp=cm["tp"],
                fp=cm["fp"],
                fn=cm["fn"],
                precision=prec,
                recall=rec,
                true_positives=cm["true_positives"],
                false_positives=cm["false_positives"],
                false_negatives=cm["false_negatives"],
                include_item_details=config.include_item_details,
            )
            precision_sum += prec
            recall_sum += rec
            tp_total += cm["tp"]
            if key in allowed_keys:
                fp_soft += cm["fp"]

        n = len(all_keys)
        avg_precision = precision_sum / n
        avg_recall = recall_sum / n
        details = {"by_key": by_key}

        if tp_total + fp_soft == 0:
            # No allowed-key FPs and no TPs: either ground_truth is empty
            # (agent selected items when none expected → 0.0) or actual only
            # has items in extra keys (no penalizable FPs → 1.0).
            prec_soft_value = 0.0 if not allowed_keys else 1.0
        else:
            prec_soft_value = tp_total / (tp_total + fp_soft)

        precision_soft_details: dict = {
            "tp_total": tp_total,
            "fp_soft": fp_soft,
            "allowed_keys": sorted(allowed_keys),
            "by_key": by_key,
        }

        return self.Output(
            precision=MetricOutputField(value=avg_precision, details=details),
            recall=MetricOutputField(value=avg_recall, details=details),
            precision_soft=MetricOutputField(
                value=prec_soft_value, details=precision_soft_details
            ),
        )
