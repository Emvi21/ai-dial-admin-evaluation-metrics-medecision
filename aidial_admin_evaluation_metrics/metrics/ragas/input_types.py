"""Shared input types for ragas-backed metrics."""

from typing import Annotated, TypeAlias

from pydantic import Field

UserInput: TypeAlias = Annotated[
    str,
    Field(description="The user query or question"),
]

Response: TypeAlias = Annotated[
    str,
    Field(description="The system response to be evaluated"),
]

RetrievedContexts: TypeAlias = Annotated[
    list[str],
    Field(description="List of context chunks retrieved by the RAG system"),
]

Reference: TypeAlias = Annotated[
    str,
    Field(description="The expected or ground-truth answer"),
]
