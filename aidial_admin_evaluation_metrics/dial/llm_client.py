"""DIAL resources factory — shared httpx transport for all DIAL API calls."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from aidial_admin_evaluation_metrics.app_config import DialConfig

SchemaMethod = Literal["function_calling", "json_mode", "json_schema"]


def _create_chat_model(
    model_name: str,
    dial_config: DialConfig,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> BaseChatModel:
    return AzureChatOpenAI(
        azure_endpoint=dial_config.dial_url,
        azure_deployment=model_name,
        api_key=dial_config.dial_api_key,
        api_version="2023-12-01-preview",
        http_client=http_client,
        http_async_client=http_async_client,
        streaming=True,
        stream_usage=True,
    )


def _create_embeddings_model(
    model_name: str,
    dial_config: DialConfig,
    http_client: httpx.Client | None = None,
    http_async_client: httpx.AsyncClient | None = None,
) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=dial_config.dial_url,
        azure_deployment=model_name,
        api_key=dial_config.dial_api_key,
        api_version="2023-12-01-preview",
        http_client=http_client,
        http_async_client=http_async_client,
    )


class DialFactory:
    """Factory for all DIAL API resources, sharing a single httpx transport.

    Holds the DIAL connection config and (optionally) shared sync/async httpx
    clients so that all model instances created through this factory reuse the
    same TCP/TLS connections to DIAL.

    In production, obtain an instance via ``dial_factory_lifespan`` which
    manages the client lifecycle.  In tests, use ``create_dial_factory`` which
    creates unshared clients suitable for unit/integration tests.
    """

    def __init__(
        self,
        dial_config: DialConfig,
        http_client: httpx.Client | None = None,
        http_async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._dial_config = dial_config
        self._http_client = http_client
        self._http_async_client = http_async_client

    def create_llm(self, model_name: str) -> BaseChatModel:
        """Create a LangChain chat model for the given DIAL deployment."""
        return _create_chat_model(
            model_name,
            self._dial_config,
            self._http_client,
            self._http_async_client,
        )

    def create_embeddings_model(self, model_name: str) -> AzureOpenAIEmbeddings:
        """Create a LangChain embeddings model for the given DIAL deployment."""
        return _create_embeddings_model(
            model_name,
            self._dial_config,
            self._http_client,
            self._http_async_client,
        )

    def create_llm_with_schema(
        self,
        model_name: str,
        schema: type,
        method: SchemaMethod = "function_calling",
    ) -> Any:
        """Create a LangChain chat model bound to a Pydantic schema for structured output."""
        return self.create_llm(model_name).with_structured_output(
            schema, method=method
        )


def create_dial_factory(dial_config: DialConfig) -> DialFactory:
    """Create a DialFactory without shared httpx transport.

    For use in integration tests and scripts.  In production use
    ``dial_factory_lifespan`` to share httpx connections.
    """
    return DialFactory(dial_config)


@asynccontextmanager
async def dial_factory_lifespan(
    dial_config: DialConfig,
) -> AsyncIterator[DialFactory]:
    """Async context manager that provides a shared-transport DialFactory.

    Creates one ``httpx.Client`` and one ``httpx.AsyncClient`` for the
    duration of the context, shared across all model instances produced by
    the factory.  This allows TCP/TLS connections to DIAL to be reused
    across consecutive requests instead of being re-established each time.

    Args:
        dial_config: DIAL connection configuration

    Yields:
        A ``DialFactory`` instance with shared httpx clients

    Example:
        >>> async with dial_factory_lifespan(dial_config) as factory:
        ...     llm = factory.create_llm("gemini-2.5-flash-lite")
        ...     emb = factory.create_embeddings_model("text-embedding-3-small")
    """
    async with httpx.AsyncClient() as async_client:
        with httpx.Client() as sync_client:
            yield DialFactory(dial_config, sync_client, async_client)
