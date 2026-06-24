"""Output types and helper functions for RAG evaluation generation metrics."""

from typing import Annotated, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, Field

from aidial_admin_evaluation_metrics.metrics.common.types import (
    MetricError,
    MetricOutputField,
)


class InferenceDetails(BaseModel):
    """Details about NLI-based inference calculation."""

    json_explanation: Annotated[
        str,
        Field(
            description="String with low level details of the inference calculation in JSON format. Exact format is an implementation detail of the inference calculation and should not be relied on for any downstream use.",
        ),
    ] = ""
    highlight: Annotated[
        str,
        Field(
            description="Highlighted text showing which segments of the text affected the inference score.",
        ),
    ] = ""


class InferenceField(MetricOutputField):
    """Metric output field for NLI-based inference score, including details about the inference calculation."""

    details: Annotated[
        InferenceDetails,
        Field(
            description="Details about the inference calculation containing JSON explanation and highlights",
        ),
    ] = InferenceDetails()


GroundTruthToAnswerField = Annotated[
    InferenceField | MetricError,
    Field(
        discriminator="type",
        description="Inference score (0-1) measuring if answer is entailed from ground truth. Score lower than 1 indicates that the answer has some statements that are not supported by the ground truth: statements that contradict the ground truth or extra statements that are not mentioned in the ground truth.",
    ),
]

AnswerToGroundTruthField = Annotated[
    InferenceField | MetricError,
    Field(
        discriminator="type",
        description="Inference score (0-1) measuring if ground truth is entailed from answer. Score lower than 1 indicates that the ground truth has some statements that are not supported by the answer: statements that contradict the answer or statements that are missing in the answer.",
    ),
]

ContextToAnswerField = Annotated[
    InferenceField | MetricError,
    Field(
        discriminator="type",
        description="Inference score (0-1) measuring if answer is entailed by context. Score lower than 1 indicates that the answer has some statements that are not supported by the context: statements that contradict the context or extra statements that are not mentioned in the context (which may or may not be correct, but are not supported by the context).",
    ),
]


def extract_generation_metric(
    row: pd.Series,
    prefix: Literal["gt_ans", "ans_gt", "ctx_ans"],
) -> InferenceField:
    """Utility to extract an InferenceField from a DataFrame row using the specified prefix."""
    return InferenceField(
        value=float(row[f"{prefix}_inference"]),
        details=InferenceDetails(
            json_explanation=str(row[f"{prefix}_json"]),
            highlight=str(row[f"{prefix}_highlight"]),
        ),
    )


T = TypeVar("T", bound=BaseModel)


def output_error(e: Exception, outputType: type[T]) -> T:
    """Utility to report an exception as error for all output fields"""
    error = MetricError(message=str(e))

    return outputType.model_validate(
        {
            field_name: error.model_dump()
            for field_name, _ in outputType.model_fields.items()
        }
    )


def extract_row(
    result_df: pd.DataFrame,
) -> pd.Series:
    """Utility to extract a single row from the result DataFrame, ensuring it has exactly one row."""
    if result_df.shape[0] != 1:
        raise ValueError(
            f"Expected result DataFrame to have a single row, but got {result_df.shape[0]} rows."
        )
    return result_df.iloc[0]
