"""
Pytest fixtures shared across Onyx tests.

All fixtures are fast, isolated, and do NOT make network calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Yield a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="onyx_test_") as d:
        yield Path(d)


@pytest.fixture
def temp_config(temp_dir):
    """
    Return an Onyx Config pointing entirely at a temporary directory.
    No LLM keys required — tests that need them should mock.
    """
    from onyx.config import Config

    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    skills_dir = data_dir / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    chroma_dir = data_dir / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    registry_db = data_dir / "registry.sqlite3"

    return Config(
        openai_api_key="test-key",
        openai_base_url="https://api.example.com/v1",
        llm_model="test-model",
        embed_model="test-embed",
        tavily_api_key=None,
        data_dir=data_dir,
        skills_dir=skills_dir,
        chroma_dir=chroma_dir,
        registry_db=registry_db,
        logs_dir=logs_dir,
        max_sources_per_skill=3,
        max_pages_per_source=1,
        request_timeout=5,
        llm_max_attempts=1,
        code_timeout=5,
        dedupe_distance=0.12,
        max_chars_per_page=2000,
        log_level="WARNING",
    )
