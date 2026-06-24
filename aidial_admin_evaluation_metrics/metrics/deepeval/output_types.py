"""Shared output types for deepeval-based metrics."""

from typing import Annotated, TypeAlias, TypedDict

from pydantic import Field

from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)


class DeepEvalDetails(TypedDict, total=False):
    reason: str
    verbose_logs: str


class DeepEvalScoreField(MetricOutputField):
    details: Annotated[
        DeepEvalDetails,
        Field(
            description=(
                "Evaluation details including the reason for the score "
                "and verbose logs from the evaluation steps"
            )
        ),
    ] = {}


ScoreOutput: TypeAlias = Annotated[
    DeepEvalScoreField | MetricError,
    Field(
        discriminator="type",
        description="Metric score (0–1) with evaluation details",
    ),
]
