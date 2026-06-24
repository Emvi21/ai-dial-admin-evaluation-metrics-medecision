from typing import Annotated, Literal

from deepdiff import DeepDiff
from pydantic import BaseModel, Field, model_validator

from aidial_admin_evaluation_metrics.metrics.common.base_metric import (
    BaseMetric,
    MetricExample,
)
from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)

_JsonValue = str | bool | int | float | list | dict | None


class DeepDiffMetric(BaseMetric):
    name: str = "deepdiff"
    display_name: str = "DeepDiff"
    description: str = (
        "Compares actual and ground_truth using DeepDiff. "
        "Supports strings, booleans, numbers, lists, and nested objects. "
        "Returns 1.0 when the values are equal, 0.0 otherwise. "
        "Mismatch details are included in the output."
    )

    class Config(BaseModel):
        ignore_order: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "If True, the order of items in lists is ignored during comparison."
                ),
            ),
        ] = False
        significant_digits: Annotated[
            int | None,
            Field(
                default=None,
                ge=0,
                description=(
                    "Number of digits to use when comparing numbers. "
                    "When number_format_notation is 'f' (default), counts digits AFTER the decimal point "
                    "(e.g., significant_digits=2 means 1.113 ≈ 1.114). "
                    "When 'e', counts digits in scientific notation, enabling approximate matching of large integers "
                    "(e.g., significant_digits=2 with 'e' means 1024 ≈ 1020, since both round to 1.02e+03). "
                    "If None, numbers are compared exactly."
                ),
            ),
        ] = None
        number_format_notation: Annotated[
            Literal["f", "e"],
            Field(
                default="f",
                description=(
                    "Controls the meaning of significant_digits. "
                    "'f' (fixed point, default): counts digits after the decimal point. "
                    "'e' (scientific notation): counts digits in exponent form, enabling approximate matching "
                    "of large integers (e.g., 1024 ≈ 1020 with significant_digits=2)."
                ),
            ),
        ] = "f"
        math_epsilon: Annotated[
            float | None,
            Field(
                default=None,
                ge=0,
                description=(
                    "Absolute tolerance for numeric comparison using Python's math.isclose(). "
                    "Numbers within this tolerance are considered equal "
                    "(e.g., math_epsilon=20 means values within 20 units match). "
                    "Cannot be combined with ignore_order=True. "
                    "If None, no epsilon tolerance is applied."
                ),
            ),
        ] = None

        @model_validator(mode="after")
        def _check_epsilon_and_ignore_order(self) -> "DeepDiffMetric.Config":
            if self.math_epsilon is not None and self.ignore_order:
                raise ValueError(
                    "math_epsilon cannot be used with ignore_order=True"
                )
            return self

    class Input(BaseModel):
        actual: Annotated[
            _JsonValue,
            Field(description="The actual output to evaluate"),
        ]
        ground_truth: Annotated[
            _JsonValue,
            Field(description="The expected correct output"),
        ]

    class Output(BaseModel):
        match: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="1.0 if actual equals ground_truth, 0.0 otherwise",
            ),
        ]

    examples = [
        MetricExample(
            name="deepdiff matches strings",
            description="Example where actual matches ground_truth",
            input=Input(actual="test answer", ground_truth="test answer"),
            expected_output=Output(
                match=MetricOutputField(value=1.0),
            ),
        ),
        MetricExample(
            name="deepdiff does not match strings",
            description="Example where actual does not match ground_truth",
            input=Input(actual="wrong answer", ground_truth="test answer"),
            expected_output=Output(
                match=MetricOutputField(
                    value=0.0,
                    details={
                        "diff": {
                            "values_changed": {
                                "root": {
                                    "new_value": "wrong answer",
                                    "old_value": "test answer",
                                }
                            }
                        },
                        "pretty": 'Value of root changed from "test answer" to "wrong answer".',
                    },
                ),
            ),
        ),
        MetricExample(
            name="deepdiff matches lists ignoring order",
            description="Example where lists match when order is ignored",
            config=Config(ignore_order=True),
            input=Input(actual=[3, 1, 2], ground_truth=[1, 2, 3]),
            expected_output=Output(
                match=MetricOutputField(value=1.0),
            ),
        ),
        MetricExample(
            name="deepdiff matches numbers within epsilon",
            description="Example where numbers match within an absolute tolerance",
            config=Config(math_epsilon=20),
            input=Input(
                actual={"bbox": [1436, 1071, 2288, 1715]},
                ground_truth={"bbox": [1435, 1071, 2291, 1714]},
            ),
            expected_output=Output(
                match=MetricOutputField(value=1.0),
            ),
        ),
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        diff = DeepDiff(
            input.ground_truth,
            input.actual,
            ignore_order=config.ignore_order,
            ignore_nan_inequality=True,  # treat NaN values as equal
            significant_digits=config.significant_digits,
            number_format_notation=config.number_format_notation,
            math_epsilon=config.math_epsilon,
        )
        if diff:
            return self.Output(
                match=MetricOutputField(
                    value=0.0,
                    details={"diff": diff.to_dict(), "pretty": diff.pretty()},
                ),
            )
        return self.Output(
            match=MetricOutputField(value=1.0),
        )
