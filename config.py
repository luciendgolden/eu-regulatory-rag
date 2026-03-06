"""Application configuration via Pydantic Settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # EUR-Lex SOAP credentials (optional — falls back to REST if not set)
    eurlex_username: str = Field(default="", description="EUR-Lex SOAP username")
    eurlex_password: str = Field(default="", description="EUR-Lex SOAP password")

    # Qdrant
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant REST port")

    # Embedding
    embedding_provider: str = Field(
        default="local",
        description="Embedding provider: 'local' (sentence-transformers) or 'openai'",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name for local embeddings",
    )

    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # LLM
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' or 'ollama'",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model name (e.g. gpt-4o, llama3)",
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )

    # API authentication
    api_key: str = Field(default="", description="API key for /api/v1/* endpoints")
    admin_api_key: str = Field(
        default="",
        description="Admin API key for privileged endpoints (e.g. /ingest). "
                    "Falls back to api_key when not set.",
    )

    @property
    def has_eurlex_credentials(self) -> bool:
        """True when SOAP credentials are configured."""
        return bool(self.eurlex_username and self.eurlex_password)

    @property
    def vector_size(self) -> int:
        """Return embedding dimension based on provider."""
        if self.embedding_provider == "openai":
            return 1536
        return 384


# Singleton — import this everywhere
settings = Settings()
