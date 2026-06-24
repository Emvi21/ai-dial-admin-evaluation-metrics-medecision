"""Shared output types for ragas-backed metrics."""

from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field, model_serializer

from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)


class RagasDetails(BaseModel):
    reason: str | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler) -> dict:
        return {k: v for k, v in handler(self).items() if v is not None}


class RagasScoreField(MetricOutputField):
    details: Annotated[
        RagasDetails,
        Field(
            description=(
                "Evaluation details including the reason for the score "
                "as provided by the ragas metric"
            ),
        ),
    ] = Field(default_factory=RagasDetails)


ScoreOutput: TypeAlias = Annotated[
    RagasScoreField | MetricError,
    Field(
        discriminator="type",
        description="Metric score (0–1) with evaluation details",
    ),
]
