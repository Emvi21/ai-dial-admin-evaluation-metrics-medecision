import re
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


class RegexMatchMetric(BaseMetric):
    name: str = "regex_match"
    display_name: str = "Regex Match"
    description: str = "Checks if the actual matches the regular expression provided in ground_truth."

    class Config(BaseModel):
        pattern: Annotated[
            str,
            Field(
                description="The regular expression pattern to match against the actual output"
            ),
        ]

    class Input(BaseModel):
        actual: Annotated[
            str, Field(description="The actual output to evaluate")
        ]

    class Output(BaseModel):
        regex_match: Annotated[
            MetricOutputField | MetricError,
            Field(
                discriminator="type",
                description="The result of the regex match evaluation",
            ),
        ]

    examples = [
        MetricExample(
            name="regex_match matches",
            description="Example where actual matches the regex pattern",
            config=Config(pattern=r"test\s\w+"),
            input=Input(actual="test answer"),
            expected_output=Output(regex_match=MetricOutputField(value=1.0)),
        ),
        MetricExample(
            name="regex_match error",
            description="Example with invalid regex pattern",
            config=Config(pattern=r"test\s("),
            input=Input(actual="test answer"),
            expected_output=Output(
                regex_match=MetricError(
                    message="Invalid regular expression pattern: missing ), unterminated subpattern at position 6"
                )
            ),
        ),
    ]

    def _evaluate(self, config: Config, input: Input) -> Output:
        try:
            regex = re.compile(config.pattern)
        except re.error as e:
            return self.Output(
                regex_match=MetricError(
                    message=f"Invalid regular expression pattern: {e}"
                )
            )

        is_match = float(regex.fullmatch(input.actual) is not None)
        return self.Output(
            regex_match=MetricOutputField(
                value=is_match,
            )
        )
