"""Ragas LLM adapter backed by a DialFactory."""

from typing import Type, TypeVar

from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory

T = TypeVar("T", bound=BaseModel)


class DialRagasLLM(InstructorBaseRagasLLM):
    """Adapts a DIAL deployment for use as a Ragas InstructorBaseRagasLLM.

    Delegates to DialFactory.create_llm_with_schema() so the DIAL-specific
    structured-output encoding (function_calling) is centralised in the factory.
    Only the async path is supported, since the service uses an async event loop.
    """

    def __init__(self, factory: DialFactory, model_name: str) -> None:
        super().__init__()
        self._factory = factory
        self._model_name = model_name

    def generate(self, prompt: str, response_model: Type[T]) -> T:  # type: ignore[override]
        raise NotImplementedError(
            "Sync generate() is not supported. Use ascore() which calls agenerate()."
        )

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:  # type: ignore[override]
        chain = self._factory.create_llm_with_schema(
            self._model_name, response_model
        )
        return await chain.ainvoke(prompt)  # type: ignore[return-value]
