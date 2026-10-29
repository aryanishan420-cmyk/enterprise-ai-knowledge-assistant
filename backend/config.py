"""Environment-based configuration. No secrets are hardcoded."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_backend: str = "sentence-transformers"  # or "hashing" (offline, lexical)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./data/vector_store"
    database_url: str = "sqlite:///./data/app.db"
    top_k: int = Field(5, ge=1, le=50)
    chunk_size: int = Field(800, ge=100)
    chunk_overlap: int = Field(120, ge=0)
    similarity_threshold: float = 0.0
    max_upload_mb: int = Field(10, ge=1)
    admin_token: str = ""
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    def require_llm_key(self) -> str:
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Add it to your .env file to enable answer "
                "generation. Search-only mode works without it."
            )
        return self.openai_api_key

    def public_dict(self) -> dict:
        """Settings safe to show in a UI (no credentials)."""
        d = self.model_dump(exclude={"openai_api_key", "admin_token"})
        d["llm_key_configured"] = bool(self.openai_api_key)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()
