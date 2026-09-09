"""Application settings, read from environment variables or a .env file.

Every setting has a working default, so the app runs with no configuration at
all. Copy .env.example to .env and change only what you need.

Where values come from, highest priority first:

    1. environment variables   (RAG_TOP_K=8)
    2. the .env file
    3. the defaults below

Usage:

    from rag_app.config import get_settings

    settings = get_settings()
    print(settings.top_k)

See .env.example for the full list of settings and what each one does.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The project folder, i.e. the one containing this package. Relative paths in
# settings are resolved against this, so the app behaves the same no matter
# which directory you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where a local Ollama server listens by default.
OLLAMA_BASE_URL = "http://localhost:11434/v1"

LlmProvider = Literal["none", "ollama", "openai"]
RetrievalMode = Literal["dense", "hybrid", "hybrid+rerank"]


class ConfigError(Exception):
    """Raised when the configuration is unusable, with advice on the fix."""


class Settings(BaseSettings):
    """All settings for the app. Prefix every environment variable with RAG_."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("./data"),
        description="Where your source documents live.",
    )
    store_dir: Path = Field(
        default=Path("./data/vector_store"),
        description="Where the vector store is written.",
    )
    collection_name: str = Field(
        default="documents",
        min_length=1,
        description="Name of the collection holding the embedded chunks.",
    )

    # --- Parsing (Phase 2) ------------------------------------------------
    enable_ocr: bool = Field(
        default=True,
        description="Read text from images and scanned PDFs. Slow but thorough.",
    )
    # NoDecode stops pydantic-settings trying to read this as JSON, which lets
    # the validator below accept a plain comma-separated string instead.
    ocr_languages: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["en"],
        min_length=1,
        description="Language hints for OCR, comma-separated.",
    )

    # --- Chunking (Phase 3) -----------------------------------------------
    chunk_max_tokens: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Maximum tokens per chunk.",
    )

    # --- Embeddings (Phase 4) ---------------------------------------------
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="Any model name supported by fastembed.",
    )

    # --- Retrieval (Phase 5) ----------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="How many chunks to feed the model per question.",
    )
    min_relevance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Discard chunks scoring below this similarity.",
    )

    # --- Hybrid search (Phase 6) ------------------------------------------
    retrieval_mode: RetrievalMode = Field(
        default="dense",
        description="Retrieval strategy: dense, hybrid, or hybrid+rerank.",
    )

    # --- Answer generation (Phase 5) --------------------------------------
    llm_provider: LlmProvider = Field(
        default="none",
        description="Which service answers questions: none, ollama, or openai.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        min_length=1,
        description="Model name as your provider spells it.",
    )
    llm_api_key: SecretStr | None = Field(
        default=None,
        description="API key. Required for openai. Never hard-code this.",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Override for OpenAI-compatible endpoints.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="0.0 keeps answers closest to the source documents.",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=64,
        le=32768,
        description="Maximum length of a generated answer, in tokens.",
    )

    # --- HTTP API (Phase 7) -----------------------------------------------
    api_host: str = Field(
        default="127.0.0.1",
        min_length=1,
        description="Keep on 127.0.0.1; the API has no authentication yet.",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port the API listens on.",
    )

    # --- Tidying up the raw values ----------------------------------------

    @field_validator("llm_api_key", "llm_base_url", mode="before")
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat `RAG_LLM_API_KEY=` as not set rather than as an empty value.

        .env.example ships these keys empty, which would otherwise read as a
        deliberate empty string and cause confusing failures later.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("ocr_languages", mode="before")
    @classmethod
    def _accept_comma_separated(cls, value: object) -> object:
        """Allow `RAG_OCR_LANGUAGES=en,de,fr` as well as a JSON list."""
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("data_dir", "store_dir")
    @classmethod
    def _make_absolute(cls, value: Path) -> Path:
        """Resolve relative paths against the project folder, not the cwd."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    # --- Cross-checks that need more than one field -----------------------

    @model_validator(mode="after")
    def _fill_in_provider_details(self) -> Settings:
        """Apply per-provider defaults, then confirm the result is usable."""
        if self.llm_provider == "openai":
            # Accept the conventional OPENAI_API_KEY too, so an existing shell
            # setup works without duplicating the key under a RAG_ name.
            if self.llm_api_key is None:
                if env_key := os.environ.get("OPENAI_API_KEY", "").strip():
                    self.llm_api_key = SecretStr(env_key)
                else:
                    raise ValueError(
                        "llm_provider is 'openai' but no API key was found. "
                        "Set RAG_LLM_API_KEY or OPENAI_API_KEY in your .env "
                        "file, or use RAG_LLM_PROVIDER=none to skip answer "
                        "generation."
                    )

        elif self.llm_provider == "ollama":
            if self.llm_base_url is None:
                self.llm_base_url = OLLAMA_BASE_URL
            if self.llm_api_key is None:
                # Ollama needs no credential, but the OpenAI client insists on
                # one being present, so supply a harmless placeholder.
                self.llm_api_key = SecretStr("ollama")

        return self

    # --- Convenience ------------------------------------------------------

    def api_key_value(self) -> str | None:
        """Return the API key as plain text, for handing to a client library.

        Kept as an explicit call so the secret is never revealed by accident:
        printing or logging `settings` shows `**********` instead.
        """
        return self.llm_api_key.get_secret_value() if self.llm_api_key else None


_cached: Settings | None = None


def get_settings(reload: bool = False) -> Settings:
    """Return the settings, loading them the first time they are asked for.

    Raises:
        ConfigError: if any setting is invalid, with the offending fields and
            their environment variable names listed.
    """
    global _cached
    if _cached is not None and not reload:
        return _cached

    try:
        _cached = Settings()
    except ValidationError as error:
        raise ConfigError(_explain(error)) from error
    return _cached


def _explain(error: ValidationError) -> str:
    """Turn a pydantic error into something worth showing a person."""
    lines = ["Your configuration could not be loaded:", ""]
    for problem in error.errors():
        field = problem["loc"][0] if problem["loc"] else "settings"
        env_name = f"RAG_{str(field).upper()}"
        lines.append(f"  {env_name}: {problem['msg']}")
    lines += ["", "Check your .env file against .env.example."]
    return "\n".join(lines)
