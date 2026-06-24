"""Output types and helper functions for RAG evaluation retrieval metrics."""

import math
from typing import Annotated, Literal, TypeAlias

import pandas as pd
from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)


class RecallDetails(BaseModel):
    """Details about the recall calculation."""

    facts_ranks: list[int] = Field(
        default=[],
        description="Array where each element is the rank of the corresponding fact in the context (-1 if not found)",
    )
    context_relevance: list[int] = Field(
        default=[],
        description="Array where each element is the count of facts found in the corresponding context chunk",
    )
    context_highlight: list[str] = Field(
        default=[],
        description="Array of JSON strings showing highlighted facts in each context chunk",
    )


class RecallField(MetricOutputField):
    """Recall metric output with additional details."""

    details: Annotated[
        RecallDetails,
        Field(
            description="Details about the recall calculation",
        ),
    ] = RecallDetails()


RecallOutput: TypeAlias = Annotated[
    RecallField | MetricError,
    Field(
        discriminator="type",
        description="The recall of the retrieval",
    ),
]

PrecisionOutput: TypeAlias = Annotated[
    MetricOutputField | MetricError,
    Field(
        discriminator="type",
        description="The precision of the retrieval",
    ),
]

F1ScoreOutput: TypeAlias = Annotated[
    MetricOutputField | MetricError,
    Field(
        discriminator="type",
        description="The F1 score of the retrieval",
    ),
]

MrrOutput: TypeAlias = Annotated[
    MetricOutputField | MetricError,
    Field(
        discriminator="type",
        description="The Mean Reciprocal Rank (MRR) of the retrieval",
    ),
]


_INVALID_NUMBER_ERROR = MetricError(
    message="metric result is not a finite number"
)


def _to_finite_float(raw) -> float | MetricError:
    try:
        val = float(raw)
    except Exception:
        return _INVALID_NUMBER_ERROR
    return val if math.isfinite(val) else _INVALID_NUMBER_ERROR


def extract_retrieval_metric(
    row: pd.Series | dict,
    name: Literal["precision", "f1", "mrr"],
) -> MetricOutputField | MetricError:
    raw = row.get(name) if isinstance(row, dict) else row[name]
    val = _to_finite_float(raw)
    if isinstance(val, MetricError):
        return val
    return MetricOutputField(value=val)


def extract_recall(row: pd.Series | dict) -> RecallField | MetricError:
    raw = row.get("recall") if isinstance(row, dict) else row["recall"]
    val = _to_finite_float(raw)
    if isinstance(val, MetricError):
        return val
    return RecallField(
        value=val,
        details=RecallDetails(
            facts_ranks=row["facts_ranks"].tolist(),
            context_relevance=row["context_relevance"].tolist(),
            context_highlight=row["context_highlight"].tolist(),
        ),
    )
