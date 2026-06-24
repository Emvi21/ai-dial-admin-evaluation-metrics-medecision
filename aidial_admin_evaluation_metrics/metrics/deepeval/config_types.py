"""Shared configuration types for deepeval-backed metrics.

These types are used in metric `Config` classes. Note: the `model` value will
be wrapped by the local `DialDeepEvalLLM` adapter before being passed to
DeepEval. Custom `DeepEvalBaseLLM` instances are not accepted via these configs.
"""

from typing import Annotated, TypeAlias

from pydantic import Field

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL

ModelConfig: TypeAlias = Annotated[
    str,
    Field(
        default=DEFAULT_MODEL,
        description=(
            "The LLM deployment name to use for deepeval metric evaluation. "
            "Must be one of the configured DIAL deployment names."
        ),
    ),
]

ThresholdConfig: TypeAlias = Annotated[
    float,
    Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for the metric to pass",
    ),
]

IncludeReasonConfig: TypeAlias = Annotated[
    bool,
    Field(
        default=True,
        description=(
            "When True, instructs DeepEval to include a human-readable reason "
            "for the metric score (may increase latency and token usage)."
        ),
    ),
]

StrictModeConfig: TypeAlias = Annotated[
    bool,
    Field(
        default=False,
        description=(
            "If True, enforces a binary metric: 1 for perfect match, 0 otherwise; "
            "overrides `threshold`."
        ),
    ),
]
