"""DIAL connection fixtures for tests."""

import os

import pytest
from pydantic import SecretStr

from aidial_admin_evaluation_metrics.app_config import DialConfig

# Normalized URL used in cassettes (same for all developers)
NORMALIZED_DIAL_HOST = "dial.test.example.com"
NORMALIZED_DIAL_URL = f"https://{NORMALIZED_DIAL_HOST}"


@pytest.fixture(scope="session")
def test_dial_config():
    """Provide test DIAL configuration.

    For recording (first run):
        Reads DIAL_URL and DIAL_API_KEY from environment variables.
        Allows testing against any DIAL instance.

    For replay (subsequent runs):
        Uses normalized URL that matches cassettes.
        No real credentials needed.
    """
    dial_url = os.getenv("DIAL_URL", NORMALIZED_DIAL_URL)
    dial_api_key = os.getenv("DIAL_API_KEY", "test-api-key")

    return DialConfig(
        dial_url=dial_url,
        dial_api_key=SecretStr(dial_api_key),
    )
