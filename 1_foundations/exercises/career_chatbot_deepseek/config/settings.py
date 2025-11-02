"""Configuration management using Pydantic Settings.

This module provides centralized, type-safe configuration management
for the Career Chatbot application.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # API Keys (Required)
    deepseek_api_key: str
    
    # Optional API Keys
    google_api_key: Optional[str] = None
    pushover_user: Optional[str] = None
    pushover_token: Optional[str] = None
    
    # Agent Configuration
    agent_name: str = "Your Name"
    primary_model: str = "deepseek-chat"
    evaluator_model: str = "gemini-2.0-flash"
    
    # Feature Flags
    enable_rag: bool = True
    enable_evaluation: bool = True
    enable_memory: bool = False  # MVP: Working memory only
    
    # RAG Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3
    
    # Memory Configuration
    working_memory_max_turns: int = 10
    working_memory_max_tokens: int = 4000
    
    # Database Paths
    db_path: str = "data/career_qa.db"
    chroma_path: str = "data/chroma_db"
    kb_path: str = "knowledge_base"
    
    # Performance
    max_retries: int = 1
    timeout_seconds: int = 30


# Global settings instance
# This will raise ValidationError at import time if required config is missing,
# making configuration issues immediately clear rather than causing confusing
# AttributeErrors later when trying to access settings properties.
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance.
    
    This function exists for dependency injection patterns and testing,
    where you might want to override settings dynamically.
    """
    return settings 