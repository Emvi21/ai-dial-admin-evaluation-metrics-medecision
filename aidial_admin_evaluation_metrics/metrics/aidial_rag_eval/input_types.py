from typing import Annotated, TypeAlias

import pandas as pd
from aidial_rag_eval.types import AnswerColumns, GroundTruthColumns
from pydantic import BaseModel, Field

QuestionInput: TypeAlias = Annotated[
    str,
    Field(
        description="The question being asked to the RAG system",
    ),
]


GroundTruthAnswerInput: TypeAlias = Annotated[
    str,
    Field(
        description="The expected answer, used as ground truth for evaluation",
    ),
]


GroundTruthFactsInput: TypeAlias = Annotated[
    list[str],
    Field(
        description="A list of the facts that should be retrieved by the RAG system "
        "to be able to produce correct answer. Used as ground truth for retrieval evaluation",
    ),
]


ContextInput: TypeAlias = Annotated[
    list[str],
    Field(
        description="List of text chunks actually retrieved by the RAG system that will be used to evaluate retrieval."
    ),
]


AnswerInput: TypeAlias = Annotated[
    str,
    Field(
        description="The answer produced by the RAG system that will be evaluated against the ground truth answer.",
    ),
]


DocumentsInput: TypeAlias = Annotated[
    list[str] | None,
    Field(
        description="Optional list of document names for this question. "
        "Useful if the dataset contains the same question multiple times with different set of documents "
        "or if the name of the document is required to answer the question.",
        default=None,
    ),
]


def get_ground_truth_dataframe(input: BaseModel) -> pd.DataFrame:
    """Convert input data to a DataFrame format expected by ai-dial-rag-eval
    depending on the fields in Input model
    """
    row = {}
    if hasattr(input, "question"):
        row[GroundTruthColumns.QUESTION] = getattr(input, "question")
    if hasattr(input, "documents"):
        row[GroundTruthColumns.DOCUMENTS] = getattr(input, "documents") or []
    if hasattr(input, "ground_truth_facts"):
        row[GroundTruthColumns.FACTS] = getattr(input, "ground_truth_facts")
    if hasattr(input, "ground_truth_answer"):
        row[GroundTruthColumns.ANSWER] = getattr(input, "ground_truth_answer")
    return pd.DataFrame([row])


def get_answers_dataframe(input: BaseModel) -> pd.DataFrame:
    """Convert input data to a DataFrame format expected by ai-dial-rag-eval
    depending on the fields in Input model
    """
    row = {}
    if hasattr(input, "question"):
        row[AnswerColumns.QUESTION] = getattr(input, "question")
    if hasattr(input, "documents"):
        row[AnswerColumns.DOCUMENTS] = getattr(input, "documents") or []
    if hasattr(input, "context"):
        row[AnswerColumns.CONTEXT] = getattr(input, "context")
    if hasattr(input, "answer"):
        row[AnswerColumns.ANSWER] = getattr(input, "answer")
    return pd.DataFrame([row])
