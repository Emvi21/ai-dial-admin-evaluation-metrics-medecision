from typing import Annotated, Literal, TypeAlias

from pydantic import Field

from aidial_admin_evaluation_metrics.app_config import DEFAULT_MODEL

MatcherType = Literal["citation", "exact"]


MatcherConfig: TypeAlias = Annotated[
    MatcherType,
    Field(
        default="citation",
        description="The type of matcher to use for matching facts to context for retrieval evaluation.",
        json_schema_extra={
            "oneOf": [
                {
                    "const": "citation",
                    "title": "Citation Matcher",
                    "description": """Fuzzy matching that canonizes text (lowercase, removes punctuation) and finds facts as substrings within context chunks.
                    Use to evaluate text retrieval, when context contains the text chunks from the document and facts contain citations or phrases from the document
                    that should be present in the retrieved chunks to answer the question.
                    """,
                },
                {
                    "const": "exact",
                    "title": "Exact Matcher",
                    "description": """Strict equality matching. Only matches when the entire context chunk exactly equals the fact (not substring matching).
                    Use when facts must match context chunks precisely. For example, if retrieved context and facts are the ids of the chunks or urls of the retrieved pages.
                    """,
                },
            ]
        },
    ),
]


ModelConfig: TypeAlias = Annotated[
    str,
    Field(
        default=DEFAULT_MODEL,
        description=(
            "The LLM deployment name to use for metric evaluation. "
            "Must be one of the configured DIAL deployment names."
        ),
    ),
]
