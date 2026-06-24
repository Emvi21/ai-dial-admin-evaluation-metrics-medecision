"""Shared input types for deepeval-backed metrics."""

from typing import Annotated, Optional, TypeAlias

from pydantic import Field

# Primary deepeval / LLMTestCase-aligned TypeAliases
# These types correspond to the keywords expected by `deepeval.test_case.LLMTestCase`.
# Note: the main input alias is named `DeepevalInput` (not `Input`) to avoid
# colliding with Pydantic `Input` models defined inside metric wrappers.
DeepevalInput: TypeAlias = Annotated[
    str,
    Field(description="Primary input text (e.g., question or prompt)"),
]

ActualOutput: TypeAlias = Annotated[
    str,
    Field(description="The actual output produced by the system/LLM"),
]

ExpectedOutput: TypeAlias = Annotated[
    str,
    Field(
        description="The expected or ground-truth output used for evaluation"
    ),
]

RetrievalContext: TypeAlias = Annotated[
    list[str],
    Field(
        description="List of context chunks retrieved by the RAG system (used as evidence)"
    ),
]

OptionalRetrievalContext: TypeAlias = Annotated[
    Optional[list[str]],
    Field(
        default=None,
        description="Optional list of context chunks retrieved by the RAG system",
    ),
]
