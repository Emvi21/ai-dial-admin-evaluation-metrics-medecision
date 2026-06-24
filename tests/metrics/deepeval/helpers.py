"""Shared test helpers for deepeval metric unit tests."""

from unittest.mock import MagicMock


def mock_a_measure(score: float, reason: str, verbose_logs: str):
    """Return a bound-method replacement for deepeval metric.a_measure().

    Sets score/reason/verbose_logs on the metric instance without calling an LLM.
    """

    async def _a_measure(self, test_case, **kwargs):
        self.score = score
        self.reason = reason
        self.verbose_logs = verbose_logs
        return score

    return _a_measure


def make_mock_dial_factory():
    factory = MagicMock()
    factory.create_llm.return_value = MagicMock()
    return factory
