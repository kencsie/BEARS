"""
System configuration module.

Uses Pydantic Settings to load secrets and DB connections from .env.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Secrets
    OPENAI_API_KEY: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str

    # Observability (optional)
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # Paths
    DATA_DIR: Path = Path("data")

    @property
    def corpus_path(self) -> Path:
        return self.DATA_DIR / "corpus.json"

    @property
    def queries_path(self) -> Path:
        return self.DATA_DIR / "queries.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
