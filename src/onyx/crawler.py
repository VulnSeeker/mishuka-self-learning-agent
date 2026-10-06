"""
Polite HTML → text crawler for Onyx.

Features:
  - honours robots.txt (per-host cache)
  - timeouts, content-type checks
  - strips scripts, styles, nav, footer
  - returns clean whitespace-normalised text
  - never raises on network errors (returns None)
"""

from __future__ import annotations

import logging
import threading
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from onyx.config import CONFIG, Config

log = logging.getLogger("onyx.crawler")

_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OnyxAgent/0.1"
)

# Tags removed before extracting text
_STRIP_TAGS = (
    "script", "style", "nav", "footer", "header", "aside",
    "form", "noscript", "iframe", "svg", "canvas", "button",
)


class Crawler:
    """Fetch and clean a web page. One instance can be reused."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._robots_cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._robots_lock = threading.Lock()

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _robots_allowed(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            root = f"{parsed.scheme}://{parsed.netloc}"

            with self._robots_lock:
                if root in self._robots_cache:
                    rp = self._robots_cache[root]
                else:
                    rp = robotparser.RobotFileParser()
                    rp.set_url(f"{root}/robots.txt")
                    try:
                        rp.read()
                    except Exception:  # noqa: BLE001
                        rp = None
                    self._robots_cache[root] = rp

            if rp is None:
                return True  # no robots → allow by default
            return rp.can_fetch(_USER_AGENT, url)
        except Exception:  # noqa: BLE001
            # On any parsing error, be permissive.
            return True

    # ------------------------------------------------------------------
    # Fetch + clean
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> str | None:
        """Return cleaned text, or None if unavailable/blocked/not HTML."""
        url = (url or "").strip()
        if not url.startswith(("http://", "https://")):
            return None

        if not self._robots_allowed(url):
            log.info("robots.txt disallows: %s", url)
            return None

        try:
            r = self._session.get(
                url,
                timeout=self.cfg.request_timeout,
                allow_redirects=True,
                stream=False,
            )
        except requests.RequestException as e:
            log.debug("crawl error %s: %s", url, e)
            return None

        if r.status_code >= 400:
            log.debug("crawl %s -> HTTP %d", url, r.status_code)
            return None

        ctype = (r.headers.get("content-type") or "").lower()
        if not any(t in ctype for t in ("text/html", "text/plain", "application/xhtml")):
            log.debug("crawl %s -> skip content-type %s", url, ctype)
            return None

        return self._clean_html(r.text)

    @staticmethod
    def _clean_html(html: str) -> str | None:
        if not html or len(html) < 200:
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:  # noqa: BLE001
            log.debug("HTML parse failed: %s", e)
            return None

        # Prefer <article> or <main> if present — less boilerplate
        container = soup.find("article") or soup.find("main") or soup

        for tag in container(list(_STRIP_TAGS)):
            tag.decompose()

        text = container.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        if len(text) < 300:
            return None

        return text[: CONFIG.max_chars_per_page]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "Crawler":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["Crawler"]
