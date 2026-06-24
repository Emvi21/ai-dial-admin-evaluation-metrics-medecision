"""Shared types for metrics."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class MetricOutputField(BaseModel):
    type: Literal["value"] = "value"  # Discriminator for metrics value type
    value: Annotated[
        float, Field(description="The numeric value of the metric output")
    ]
    details: Annotated[
        dict,
        Field(description="Additional details about the metric output"),
    ] = {}


class MetricError(BaseModel):
    type: Literal["error"] = "error"  # Discriminator for error type
    message: Annotated[
        str, Field(description="Error message describing the issue")
    ]


class ToolCallInput(BaseModel):
    """A single tool call made by an agent.

    Used as the shared input type for all tool-call evaluation metrics.
    """

    name: str = Field(description="The name of the tool that was called")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments passed to the tool call",
    )


MetricField = Annotated[
    MetricOutputField | MetricError,
    Field(
        discriminator="type",
        description=(
            "The result of a metric field, which can be either an output value"
            " or an error. The 'type' field indicates which one it is."
        ),
    ),
]


MetricResults = Annotated[
    dict[str, MetricField],
    Field(
        description="A dictionary mapping metric output field names to their corresponding results."
    ),
]
