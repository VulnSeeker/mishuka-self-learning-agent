"""
Web search for Onyx.

Uses Tavily if `TAVILY_API_KEY` is set (better for agents — clean results),
otherwise falls back to DuckDuckGo (free, no key required).

Returns a list of `Source` dataclasses. Never raises on network errors —
logs and returns [] so callers can degrade gracefully.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from onyx.config import CONFIG, Config
from onyx.schemas import Source

log = logging.getLogger("onyx.search")

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OnyxAgent/0.1"
)


# ---------------------------------------------------------------------------
# DuckDuckGo (lazy import — optional dependency)
# ---------------------------------------------------------------------------

def _load_ddg():
    try:
        from ddgs import DDGS  # type: ignore
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
            return DDGS
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# SearchClient
# ---------------------------------------------------------------------------

class SearchClient:
    """Tavily with DuckDuckGo fallback."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        })
        self._ddg = _load_ddg()
        if not self._ddg and not cfg.tavily_api_key:
            log.warning(
                "No search backend available. Install duckduckgo-search "
                "or set TAVILY_API_KEY."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 6) -> list[Source]:
        """Return up to `max_results` sources for the query. Never raises."""
        query = (query or "").strip()
        if not query:
            return []

        if self.cfg.tavily_api_key:
            try:
                out = self._tavily(query, max_results)
                if out:
                    return out
                log.debug("Tavily returned 0 results; trying DDG")
            except Exception as e:  # noqa: BLE001
                log.warning("Tavily failed (%s); falling back to DDG", e)

        return self._ddg_search(query, max_results)

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------

    def _tavily(self, query: str, max_results: int) -> list[Source]:
        assert self.cfg.tavily_api_key is not None
        r = self._session.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self.cfg.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_raw_content": False,
                "include_answer": False,
            },
            timeout=self.cfg.request_timeout,
        )
        r.raise_for_status()
        payload = r.json()
        out: list[Source] = []
        for item in payload.get("results", []):
            url = (item.get("url") or "").strip()
            if not url:
                continue
            out.append(
                Source(
                    title=(item.get("title") or "").strip(),
                    url=url,
                    snippet=(item.get("content") or "").strip(),
                )
            )
        return out[:max_results]

    # ------------------------------------------------------------------
    # DuckDuckGo
    # ------------------------------------------------------------------

    def _ddg_search(self, query: str, max_results: int) -> list[Source]:
        if not self._ddg:
            return []

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                with self._ddg() as ddgs:
                    hits = list(ddgs.text(query, max_results=max_results))
                out: list[Source] = []
                for h in hits:
                    url = (h.get("href") or h.get("url") or "").strip()
                    if not url:
                        continue
                    out.append(
                        Source(
                            title=(h.get("title") or "").strip(),
                            url=url,
                            snippet=(h.get("body") or "").strip(),
                        )
                    )
                return out[:max_results]
            except Exception as e:  # noqa: BLE001
                last_err = e
                log.debug("DDG attempt %d failed: %s", attempt + 1, e)
                time.sleep(1.0 * (attempt + 1))

        if last_err:
            log.warning("DDG failed after 3 attempts: %s", last_err)
        return []


__all__ = ["SearchClient"]
