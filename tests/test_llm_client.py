"""Tests for LLM configuration module."""

import logging
import os
from unittest.mock import patch

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from aidial_admin_evaluation_metrics.app_config import (
    SUPPORTED_MODELS,
    AppSettings,
)
from aidial_admin_evaluation_metrics.dial.llm_client import create_dial_factory


def test_create_chat_model_success(test_dial_config):
    """Test successful LLM creation with valid DialConfig."""
    llm = create_dial_factory(test_dial_config).create_llm(
        "gemini-2.5-flash-lite"
    )

    assert llm is not None
    assert isinstance(llm, BaseChatModel)


def test_create_chat_model_all_models(test_dial_config):
    """Test LLM creation with all configured models."""
    factory = create_dial_factory(test_dial_config)
    for model in SUPPORTED_MODELS:
        llm = factory.create_llm(model)
        assert isinstance(llm, BaseChatModel)


def test_create_chat_model_with_string(test_dial_config):
    """Test that create_llm accepts plain string model names."""
    llm = create_dial_factory(test_dial_config).create_llm(
        "gemini-2.5-flash-lite"
    )
    assert isinstance(llm, BaseChatModel)


@patch.dict(
    os.environ,
    {"DIAL_URL": "https://dial.example.com", "DIAL_API_KEY": "test-key"},
)
def test_create_app_settings_from_env_success():
    """Test successful AppSettings creation from environment variables."""
    config = AppSettings()

    assert config.dial_url == "https://dial.example.com"
    assert config.dial_api_key.get_secret_value() == "test-key"


@patch.dict(os.environ, {}, clear=True)
def test_create_app_settings_from_env_missing_dial_url():
    """Test that missing DIAL_URL raises a validation error."""
    os.environ.pop("DIAL_URL", None)
    os.environ["DIAL_API_KEY"] = "test-key"

    with pytest.raises(
        ValidationError, match="DIAL_URL environment variable is required"
    ):
        AppSettings()


@patch.dict(os.environ, {}, clear=True)
def test_create_app_settings_missing_dial_api_key(caplog):
    """Test that missing DIAL_API_KEY logs a warning."""
    os.environ["DIAL_URL"] = "https://dial.example.com"
    os.environ.pop("DIAL_API_KEY", None)

    with caplog.at_level(logging.WARNING):
        AppSettings()
        assert "DIAL_API_KEY environment variable is not set" in caplog.text
