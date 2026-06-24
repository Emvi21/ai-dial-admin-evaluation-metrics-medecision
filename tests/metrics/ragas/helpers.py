"""Shared test helpers for ragas metric unit tests."""

from unittest.mock import MagicMock

from ragas.metrics.result import MetricResult


def mock_ascore(value: float, reason: str = ""):
    """Return an async ``ascore`` method replacement for a ragas metric.

    Bypasses the real LLM call and returns a MetricResult with the given
    value and reason, mirroring what a real ragas metric would return.
    """

    async def _ascore(self, **kwargs):
        return MetricResult(value=value, reason=reason)

    return _ascore


def make_mock_dial_factory():
    """Return a mock DialFactory whose create_llm and create_embeddings_model return MagicMocks."""
    factory = MagicMock()
    factory.create_llm.return_value = MagicMock()
    factory.create_embeddings_model.return_value = MagicMock()
    return factory
