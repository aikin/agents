import os

import pytest
from pydantic_settings import SettingsConfigDict

from config.settings import Settings


@pytest.fixture(autouse=True)
def use_test_env(monkeypatch):
    test_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    monkeypatch.chdir(test_dir)

    original_config = Settings.model_config
    Settings.model_config = SettingsConfigDict(
        env_file=".env.test",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    yield
    Settings.model_config = original_config


def test_settings_with_required_fields():
    """Test settings initialization with required fields."""
    settings = Settings(deepseek_api_key="test-key")
    assert settings.deepseek_api_key == "test-key"
    assert settings.agent_name == "Your Name"


def test_settings_feature_flags():
    """Test feature flags work correctly."""
    settings = Settings(
        enable_rag=False,
        enable_evaluation=False
    )
    assert settings.enable_rag is False
    assert settings.enable_evaluation is False


def test_settings_defaults():
    settings = Settings()
    assert settings.deepseek_api_key == "sk-your-deepseek-key-here"
    assert settings.primary_model == "deepseek-chat"
    assert settings.chunk_size == 1000
    assert settings.top_k_results == 3


def test_settings_custom_values():
    settings = Settings(
        agent_name="Test Agent",
        chunk_size=500
    )
    assert settings.agent_name == "Test Agent"
    assert settings.chunk_size == 500
