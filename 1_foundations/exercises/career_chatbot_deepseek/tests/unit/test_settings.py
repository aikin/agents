"""Tests for configuration management."""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict
from config.settings import Settings


@pytest.fixture(autouse=True)
def use_test_env(monkeypatch):
    """Automatically use .env.test for all tests in this module."""
    # Change to test directory to ensure .env.test is found
    test_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    monkeypatch.chdir(test_dir)
    
    # Temporarily modify Settings.model_config to use .env.test
    original_config = Settings.model_config
    Settings.model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    yield
    # Restore original config after test
    Settings.model_config = original_config


def test_settings_with_required_fields():
    """Test settings initialization with required fields."""
    settings = Settings(deepseek_api_key="test-key")
    assert settings.deepseek_api_key == "test-key"
    assert settings.agent_name == "Your Name"  # from .env.test


def test_settings_feature_flags():
    """Test feature flags work correctly."""
    settings = Settings(
        enable_rag=False,
        enable_evaluation=False
    )
    assert settings.enable_rag is False
    assert settings.enable_evaluation is False


def test_settings_defaults():
    """Test default values are set correctly."""
    settings = Settings()
    assert settings.deepseek_api_key == "test-deepseek-key"  # from .env.test
    assert settings.primary_model == "deepseek-chat"
    assert settings.chunk_size == 1000
    assert settings.top_k_results == 3


def test_settings_custom_values():
    """Test custom values override defaults."""
    settings = Settings(
        agent_name="Test Agent",
        chunk_size=500
    )
    assert settings.agent_name == "Test Agent"
    assert settings.chunk_size == 500