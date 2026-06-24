"""Shared configuration types for ragas-backed metrics."""

from typing import Annotated, TypeAlias

from pydantic import Field

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL

ModelConfig: TypeAlias = Annotated[
    str,
    Field(
        default=DEFAULT_MODEL,
        description=(
            "The LLM deployment name to use for ragas metric evaluation. "
            "Must be one of the configured DIAL deployment names."
        ),
    ),
]

EmbeddingsModelConfig: TypeAlias = Annotated[
    str,
    Field(
        default="text-embedding-ada-002",
        description=(
            "The embeddings deployment name to use for ragas metrics requiring embeddings "
            "(e.g. ragas.answer_relevancy). Must be a DIAL embeddings deployment name."
        ),
    ),
]
