"""
Configuration loader for Onyx.

Reads environment variables (via python-dotenv + os.environ), validates
them, and exposes a single frozen `Config` object. All other modules
import `CONFIG` from here — no module reads os.environ directly.

Env vars (all optional except OPENAI_API_KEY):

    OPENAI_API_KEY       LLM provider API key (default: "ollama")
    OPENAI_BASE_URL      OpenAI-compatible endpoint
    LLM_MODEL            Chat model name
    EMBED_MODEL          Embedding model name
    TAVILY_API_KEY       Optional; DuckDuckGo used if empty
    DATA_DIR             Where runtime data lives (default: ./data)
    SKILLS_DIR           Where skill DBs live (default: DATA_DIR/skills)
    LOG_LEVEL            DEBUG|INFO|WARNING|ERROR (default: INFO)
    MAX_SOURCES_PER_SKILL
    MAX_PAGES_PER_SOURCE
    REQUEST_TIMEOUT
    LLM_MAX_ATTEMPTS
    CODE_TIMEOUT
    DEDUPE_DISTANCE
    MAX_CHARS_PER_PAGE
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env once at import time. Safe if file is missing.
load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


def _env_opt_str(key: str) -> Optional[str]:
    val = os.getenv(key)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"env var {key} must be an integer, got {raw!r}") from e


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"env var {key} must be a float, got {raw!r}") from e


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration. Build once via `Config.from_env()`."""

    # --- LLM ---
    openai_api_key: str
    openai_base_url: str
    llm_model: str
    embed_model: str

    # --- Search ---
    tavily_api_key: Optional[str]

    # --- Paths ---
    data_dir: Path
    skills_dir: Path
    chroma_dir: Path
    registry_db: Path
    logs_dir: Path

    # --- Behaviour ---
    max_sources_per_skill: int
    max_pages_per_source: int
    request_timeout: int
    llm_max_attempts: int
    code_timeout: int
    dedupe_distance: float
    max_chars_per_page: int
    log_level: str

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from environment variables."""
        # Paths — resolve relative to CWD
        data_dir = Path(_env_str("DATA_DIR", "./data")).expanduser().resolve()
        skills_dir = Path(_env_str("SKILLS_DIR", str(data_dir / "skills"))).expanduser().resolve()
        chroma_dir = data_dir / "chroma"
        logs_dir = data_dir / "logs"
        registry_db = data_dir / "registry.sqlite3"

        for p in (data_dir, skills_dir, chroma_dir, logs_dir):
            p.mkdir(parents=True, exist_ok=True)

        return cls(
            openai_api_key=_env_str("OPENAI_API_KEY", "ollama"),
            openai_base_url=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            llm_model=_env_str("LLM_MODEL", "gpt-4o-mini"),
            embed_model=_env_str("EMBED_MODEL", "text-embedding-3-small"),
            tavily_api_key=_env_opt_str("TAVILY_API_KEY"),
            data_dir=data_dir,
            skills_dir=skills_dir,
            chroma_dir=chroma_dir,
            registry_db=registry_db,
            logs_dir=logs_dir,
            max_sources_per_skill=_env_int("MAX_SOURCES_PER_SKILL", 12),
            max_pages_per_source=_env_int("MAX_PAGES_PER_SOURCE", 1),
            request_timeout=_env_int("REQUEST_TIMEOUT", 20),
            llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 4),
            code_timeout=_env_int("CODE_TIMEOUT", 20),
            dedupe_distance=_env_float("DEDUPE_DISTANCE", 0.12),
            max_chars_per_page=_env_int("MAX_CHARS_PER_PAGE", 12000),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-line summary (never leaks the API key)."""
        key = "set" if self.openai_api_key and self.openai_api_key != "ollama" else "ollama"
        tavily = "set" if self.tavily_api_key else "off (DuckDuckGo)"
        return (
            f"model={self.llm_model} embed={self.embed_model} "
            f"key={key} tavily={tavily} data={self.data_dir}"
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

CONFIG: Config = Config.from_env()


__all__ = ["Config", "CONFIG"]
