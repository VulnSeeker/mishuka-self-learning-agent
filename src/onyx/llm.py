"""
LLM client for Onyx.

Thin wrapper around any OpenAI-compatible endpoint (OpenAI, Groq, Together,
Ollama, OpenRouter, vLLM, etc.). Adds:
  - retries with exponential backoff on transient errors
  - JSON extraction from messy responses
  - batch embeddings with a single round trip
  - thread-safe access

Nothing else in Onyx talks to the LLM directly.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Iterable

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from onyx.config import CONFIG, Config

log = logging.getLogger("onyx.llm")


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMResponseError(RuntimeError):
    """Raised when the LLM returns something we cannot parse or use."""


def parse_json_loose(raw: str) -> Any:
    """
    Extract a JSON value from a possibly-messy LLM response.

    Handles:
      - pure JSON
      - ```json fenced blocks
      - prose before/after a JSON object or array
    """
    if not raw or not raw.strip():
        raise LLMResponseError("empty LLM response")

    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First { ... } or [ ... ]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = raw.find(open_c)
        end = raw.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                continue

    snippet = raw[:200].replace("\n", " ")
    raise LLMResponseError(f"could not parse JSON from LLM output: {snippet!r}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Thread-safe OpenAI-compatible LLM client."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            timeout=cfg.request_timeout * 3,  # LLM calls can be slow
            max_retries=0,                     # we handle retries ourselves
        )
        self._lock = threading.Lock()
        log.debug("LLMClient initialised: %s", cfg.summary())

    # ------------------------------------------------------------------
    # Retry policy: only retry on transient errors, never on 4xx.
    # ------------------------------------------------------------------

    def _retry(self):
        return retry(
            stop=stop_after_attempt(self.cfg.llm_max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(
                (APIConnectionError, RateLimitError)
            ),
            reraise=True,
        )

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        """Single-turn chat completion. Returns the assistant message string."""

        @self._retry()
        def _call() -> str:
            with self._lock:
                resp = self._client.chat.completions.create(
                    model=self.cfg.llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            if not resp.choices:
                raise LLMResponseError("LLM returned no choices")
            return (resp.choices[0].message.content or "").strip()

        try:
            return _call()
        except APIError as e:
            log.error("LLM chat failed: %s", e)
            raise

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Any:
        """Like chat() but forces JSON-only responses and parses them."""
        system_json = (
            system.rstrip()
            + "\n\nYou MUST respond with valid JSON only. "
            "No prose, no markdown fences, no commentary."
        )
        raw = self.chat(system_json, user, temperature=temperature, max_tokens=max_tokens)
        return parse_json_loose(raw)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, texts: Iterable[str] | str) -> list[list[float]]:
        """Embed one or more strings. Returns a list of vectors."""
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return []

        @self._retry()
        def _call() -> list[list[float]]:
            with self._lock:
                resp = self._client.embeddings.create(
                    model=self.cfg.embed_model,
                    input=batch,
                )
            return [d.embedding for d in resp.data]

        try:
            return _call()
        except APIError as e:
            log.error("LLM embed failed: %s", e)
            raise

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Returns one vector."""
        vecs = self.embed([text])
        if not vecs:
            raise LLMResponseError("embedding returned no vectors")
        return vecs[0]


__all__ = ["LLMClient", "LLMResponseError", "parse_json_loose"]
