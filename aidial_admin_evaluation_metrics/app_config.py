"""Application configuration."""

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

SUPPORTED_MODELS: list[str] = [
    "gemini-2.5-flash-lite",
    "gpt-5-nano-2025-08-07",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "gemini-3.1-flash-lite-preview",
    "gpt-5-mini-2025-08-07",
    "gpt-5.4-mini-2026-03-17",
]
DEFAULT_MODEL: str = "gemini-3.1-flash-lite-preview"


@dataclass(frozen=True)
class DialConfig:
    """Configuration for DIAL API connection.

    Attributes:
        dial_url: The DIAL API endpoint URL
        dial_api_key: The API key for DIAL authentication (SecretStr for security)
    """

    dial_url: str
    dial_api_key: SecretStr


class AppRuntimeSettings(BaseModel):
    """Runtime settings for the application server."""

    max_concurrent_evaluations: int = 10
    max_queue_backlog: int = 100


class CommonGroupSettings(BaseModel):
    """Common fallback settings for all LLM-backed metric groups.

    When a group does not specify its own ``supported_models`` or
    ``default_model``, the values from this object are used, falling back
    to ``SUPPORTED_MODELS`` / ``DEFAULT_MODEL`` when not explicitly configured.
    """

    supported_models: list[str] | None = None
    default_model: str | None = None


class DeepEvalGroupSettings(CommonGroupSettings):
    """Settings for all deepeval-backed metrics."""

    verbose_mode: bool = False


class AidialRagEvalGroupSettings(CommonGroupSettings):
    """Settings for all aidial-rag-eval-backed metrics."""

    max_concurrency: int = 8


class RagasGroupSettings(CommonGroupSettings):
    """Settings for all ragas-backed metrics."""

    embeddings_model: str | None = None


class MetricsSettings(BaseModel):
    """Settings for all metric groups, passed to the registry at startup."""

    common: CommonGroupSettings = CommonGroupSettings()
    deepeval: DeepEvalGroupSettings = DeepEvalGroupSettings()
    aidial_rag_eval: AidialRagEvalGroupSettings = AidialRagEvalGroupSettings()
    ragas: RagasGroupSettings = RagasGroupSettings()

    @model_validator(mode="after")
    def _resolve_group_fallbacks(self) -> "MetricsSettings":
        """Fill each group's None model fields from common settings, then validate."""
        global_supported = self.common.supported_models or SUPPORTED_MODELS
        global_default = self.common.default_model or DEFAULT_MODEL
        for group in (self.deepeval, self.aidial_rag_eval, self.ragas):
            if group.supported_models is None:
                group.supported_models = list(global_supported)
            if group.default_model is None:
                group.default_model = global_default
            if group.default_model not in group.supported_models:
                logger.warning(
                    "Default model '%s' is not in the allowed models list %s. "
                    "Using first allowed model as default.",
                    group.default_model,
                    group.supported_models,
                )
                group.default_model = group.supported_models[0]
        return self


class AppSettings(BaseSettings):
    """Top-level application settings loaded from environment variables.

    Reads DIAL connection settings, app runtime limits, and per-group LLM model
    configuration. See README for the full list of supported environment variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL__",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        populate_by_name=True,
    )

    dial_url: str = Field(default="", validation_alias="DIAL_URL")
    dial_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="DIAL_API_KEY"
    )
    app: AppRuntimeSettings = AppRuntimeSettings()
    metrics: MetricsSettings = MetricsSettings()

    @property
    def dial_config(self) -> DialConfig:
        """Build a DialConfig from current settings."""
        return DialConfig(
            dial_url=self.dial_url,
            dial_api_key=self.dial_api_key,
        )

    @model_validator(mode="after")
    def _warn_missing_settings(self) -> "AppSettings":
        if not self.dial_url:
            raise ValueError(
                "DIAL_URL environment variable is required. "
                "Please set it to the DIAL API endpoint URL."
            )
        if not self.dial_api_key.get_secret_value():
            logger.warning(
                "DIAL_API_KEY environment variable is not set. "
                "LLM-based metrics will not work without this. "
                "Please set DIAL_API_KEY to your DIAL API key if you intend to use LLM-based metrics."
            )
        return self
