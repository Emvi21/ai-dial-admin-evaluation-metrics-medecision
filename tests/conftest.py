"""Pytest configuration for setting up test environment."""

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from aidial_admin_evaluation_metrics.app import create_app
from aidial_admin_evaluation_metrics.app_config import AppSettings

pytest_plugins = [
    "tests.fixtures.dial",
    "tests.fixtures.vcr",
    "tests.fixtures.nltk",
]

# Load .env file at test session start (before any fixtures)
# Searches for .env starting from cwd and walking up the tree
load_dotenv()


@pytest.fixture(scope="session")
def test_app(test_dial_config):
    """Create test application instance.

    Creates app with test configuration instead of reading from environment.
    """
    return create_app(
        app_config=AppSettings(
            dial_url=test_dial_config.dial_url,
            dial_api_key=test_dial_config.dial_api_key,
        )
    )


@pytest.fixture
def app_client(test_app):
    """Provide FastAPI test client.

    Creates a fresh test client for each test.
    """
    with TestClient(test_app) as client:
        yield client
