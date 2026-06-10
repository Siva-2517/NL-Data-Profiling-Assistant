"""
Application configuration variables loaded from environment variables and dotenv configurations.
Uses pydantic-settings to validate types and values.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Application-wide settings validated at runtime.
    Default values are provided if variables are missing from the environment.
    """
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database URLs
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'data' / 'sqlite.db').as_posix()}"

    # Vector Indexes
    CHROMA_DB_PATH: str = (BASE_DIR / "data" / "chromadb").as_posix()

    # API Configurations
    API_BASE_URL: str = "http://localhost:8000/api/v1"

    # Groq Service Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # File Storage Configurations
    UPLOAD_DIR: str = (BASE_DIR / "data" / "uploads").as_posix()
    PROFILES_DIR: str = (BASE_DIR / "data" / "profiles").as_posix()
    MAX_FILE_SIZE_MB: int = 50

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **values):
        super().__init__(**values)
        
        # Dynamically resolve DATABASE_URL if relative sqlite
        if self.DATABASE_URL.startswith("sqlite:///"):
            db_path_str = self.DATABASE_URL[10:]
            if db_path_str.startswith("./"):
                db_path_str = db_path_str[2:]
            db_path = Path(db_path_str)
            if not db_path.is_absolute():
                self.DATABASE_URL = f"sqlite:///{(BASE_DIR / db_path).resolve().as_posix()}"

        # Dynamically resolve relative CHROMA_DB_PATH
        chroma_path = Path(self.CHROMA_DB_PATH)
        if not chroma_path.is_absolute():
            self.CHROMA_DB_PATH = (BASE_DIR / chroma_path).resolve().as_posix()

        # Dynamically resolve relative UPLOAD_DIR
        upload_path = Path(self.UPLOAD_DIR)
        if not upload_path.is_absolute():
            self.UPLOAD_DIR = (BASE_DIR / upload_path).resolve().as_posix()

        # Dynamically resolve relative PROFILES_DIR
        profiles_path = Path(self.PROFILES_DIR)
        if not profiles_path.is_absolute():
            self.PROFILES_DIR = (BASE_DIR / profiles_path).resolve().as_posix()

# Instantiate settings singleton
settings = Settings()

# Ensure target storage directories exist on startup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROFILES_DIR, exist_ok=True)

