"""
Application configuration variables loaded from environment variables and dotenv configurations.
Uses pydantic-settings to validate types and values.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application-wide settings validated at runtime.
    Default values are provided if variables are missing from the environment.
    """
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database URLs
    DATABASE_URL: str = "sqlite:///./data/sqlite.db"

    # Vector Indexes
    CHROMA_DB_PATH: str = "./data/chromadb"

    # Ollama Local Service Configuration (Deprecated)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct-q4_K_M"

    # Groq Service Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # File Storage Configurations
    UPLOAD_DIR: str = "./data/uploads"
    PROFILES_DIR: str = "./data/profiles"
    MAX_FILE_SIZE_MB: int = 50

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings singleton
settings = Settings()

# Ensure target storage directories exist on startup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROFILES_DIR, exist_ok=True)

