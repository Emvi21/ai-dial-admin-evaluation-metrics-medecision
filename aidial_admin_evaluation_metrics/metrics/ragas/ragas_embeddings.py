"""Ragas embeddings adapter wrapping LangChain AzureOpenAIEmbeddings."""

from langchain_openai import AzureOpenAIEmbeddings
from ragas.embeddings.base import BaseRagasEmbedding


class DialRagasEmbeddings(BaseRagasEmbedding):
    """Adapts LangChain AzureOpenAIEmbeddings for use as a Ragas BaseRagasEmbedding.

    Provides both sync and async embedding methods by delegating to LangChain's
    AzureOpenAIEmbeddings, which supports DIAL's Azure-compatible API.
    """

    def __init__(self, embeddings: AzureOpenAIEmbeddings) -> None:
        super().__init__()
        self._embeddings = embeddings

    def embed_text(self, text: str, **kwargs: object) -> list[float]:
        raise NotImplementedError(
            "Sync embed_text() is not supported. Use aembed_text() which is called via ascore()."
        )

    async def aembed_text(self, text: str, **kwargs: object) -> list[float]:
        return await self._embeddings.aembed_query(text)
