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


def _compute_set_metrics(
    actual: list[str], ground_truth: list[str]
) -> tuple[float, float, dict]:
    """Compute precision and recall for two lists treated as sets.

    Returns (precision, recall, details).

    Edge cases:
    - actual empty, ground_truth non-empty: precision=1.0, recall=0.0
    - actual non-empty, ground_truth empty: precision=0.0, recall=1.0
    - both empty: precision=1.0, recall=1.0
    """
    actual_set = set(actual)
    gt_set = set(ground_truth)

    true_positives = sorted(actual_set & gt_set)
    false_positives = sorted(actual_set - gt_set)
    false_negatives = sorted(gt_set - actual_set)

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    details = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    return precision, recall, details


class SetComparisonMetric(BaseMetric):
    name: str = "set_comparison"
    display_name: str = "Set Comparison"
    description: str = (
        "Compares the actual and ground_truth lists as sets and calculates "
        "precision and recall. "
        "Precision = TP / (TP + FP), Recall = TP / (TP + FN). "
        "When the actual set is empty, precision is 1.0; "
        "when the ground truth set is empty, recall is 1.0. "
        "Both fields carry the same confusion matrix in their details."
    )

    class Config(BaseModel):
        pass

    class Input(BaseModel):
        actual: Annotated[
            list[str],
            Field(description="The actual output items to evaluate"),
        ]
        ground_truth: Annotated[
            list[str],
            Field(description="The expected correct items"),
        ]

    class Output(BaseModel):
        precision: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="Precision of the set comparison",
            ),
        ]
        recall: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="Recall of the set comparison",
            ),
        ]

    examples = [
        MetricExample(
            name="set_comparison perfect match",
            description="Example where actual exactly matches ground truth",
            input=Input(
                actual=["a", "b", "c"],
                ground_truth=["a", "b", "c"],
            ),
            expected_output=Output(
                precision=MetricOutputField(
                    value=1.0,
                    details={
                        "tp": 3,
                        "fp": 0,
                        "fn": 0,
                        "true_positives": ["a", "b", "c"],
                        "false_positives": [],
                        "false_negatives": [],
                    },
                ),
                recall=MetricOutputField(
                    value=1.0,
                    details={
                        "tp": 3,
                        "fp": 0,
                        "fn": 0,
                        "true_positives": ["a", "b", "c"],
                        "false_positives": [],
                        "false_negatives": [],
                    },
                ),
            ),
        ),
        MetricExample(
            name="set_comparison partial match",
            description="Example where actual partially overlaps with ground truth",
            input=Input(
                actual=["a", "b", "x"],
                ground_truth=["a", "b", "c"],
            ),
            expected_output=Output(
                precision=MetricOutputField(
                    value=2 / 3,
                    details={
                        "tp": 2,
                        "fp": 1,
                        "fn": 1,
                        "true_positives": ["a", "b"],
                        "false_positives": ["x"],
                        "false_negatives": ["c"],
                    },
                ),
                recall=MetricOutputField(
                    value=2 / 3,
                    details={
                        "tp": 2,
                        "fp": 1,
                        "fn": 1,
                        "true_positives": ["a", "b"],
                        "false_positives": ["x"],
                        "false_negatives": ["c"],
                    },
                ),
            ),
        ),
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        precision, recall, details = _compute_set_metrics(
            input.actual, input.ground_truth
        )
        return self.Output(
            precision=MetricOutputField(value=precision, details=details),
            recall=MetricOutputField(value=recall, details=details),
        )
