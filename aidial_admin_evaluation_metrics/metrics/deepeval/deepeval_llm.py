"""DeepEval LLM adapter backed by a DialFactory."""

from typing import Any, Union

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from aidial_admin_evaluation_metrics.dial.llm_client import DialFactory


class DialDeepEvalLLM(DeepEvalBaseLLM):
    """Adapts a DIAL deployment for use as a DeepEval LLM.

    Delegates to DialFactory.create_llm_with_schema() so the DIAL-specific
    structured-output encoding (function_calling) is centralised in the factory.
    """

    def __init__(self, factory: DialFactory, model_name: str):
        self._factory = factory
        self._model_name = model_name
        # Pre-create model for load_model() and the no-schema (plain text) path.
        self._model = factory.create_llm(model_name)

        # super().__init__() calls load_model()
        super().__init__(model=model_name)

    def load_model(self):  # type: ignore[override]
        return self._model

    def get_model_name(self) -> str:
        return self._model_name

    def _get_model_with_schema(self, schema=None):
        if schema:
            return self._factory.create_llm_with_schema(
                self._model_name, schema
            )
        return self._model | StrOutputParser()

    # We have to use `type: ignore[override]` here, because the declared DeepEvalBaseLLM
    # does not correspond to the actual implementations.
    # Actual implementations can return not only string, but also BaseModel instances,
    # depending on the presence of a schema in the input, and even cost.
    def generate(self, prompt: str, *args, schema=None, **kwargs) -> Union[dict[str, Any], str, BaseModel]:  # type: ignore[override]
        raise NotImplementedError(
            "Sync generate() is not supported. Use a_measure() which calls a_generate()."
        )

    async def a_generate(self, prompt: str, *args, schema=None, **kwargs) -> Union[dict[str, Any], str, BaseModel]:  # type: ignore[override]
        model = self._get_model_with_schema(schema)
        return await model.ainvoke(prompt)
