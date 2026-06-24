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


class ExactMatchMetric(BaseMetric):
    name: str = "exact_match"
    display_name: str = "Exact Match"
    description: str = "Checks if the actual exactly matches the ground_truth."

    class Config(BaseModel):
        pass

    class Input(BaseModel):
        actual: str = Field(..., description="The actual output to evaluate")
        ground_truth: str = Field(
            ..., description="The expected correct output"
        )

    class Output(BaseModel):
        exact_match: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="The result of the exact match evaluation",
            ),
        ]

    examples = [
        MetricExample(
            name="exact_match matches",
            description="Example where actual matches ground truth",
            input=Input(actual="test answer", ground_truth="test answer"),
            expected_output=Output(exact_match=MetricOutputField(value=1.0)),
        ),
        MetricExample(
            name="exact_match does not match",
            description="Example where actual does not match ground truth",
            input=Input(actual="wrong answer", ground_truth="test answer"),
            expected_output=Output(exact_match=MetricOutputField(value=0.0)),
        ),
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        is_match = float(input.actual == input.ground_truth)
        return self.Output(
            exact_match=MetricOutputField(
                value=is_match,
            )
        )
