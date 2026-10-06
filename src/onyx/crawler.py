"""
Polite HTML → text crawler for Onyx.

Features:
  - honours robots.txt (per-host cache)
  - timeouts, content-type checks
  - UTF-8 first decoding (fixes non-ASCII mangling on modern sites)
  - strips scripts, styles, nav, footer, comments
  - PRESERVES code blocks: <pre> and inline <code> keep newlines,
    wrapped in markdown fences so downstream extraction can identify them
  - returns clean whitespace-normalised text
  - never raises on network errors (returns None)
"""

from __future__ import annotations

import logging
import threading
import urllib.robotparser as robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, Comment

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

# Page must yield at least this many characters to be considered useful
_MIN_TEXT_LENGTH = 300

# <article> / <main> must hold at least this many characters to be
# preferred over the full page
_MIN_CONTAINER_LENGTH = 500


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

        html = self._decode(r)
        return self._clean_html(html)

    # ------------------------------------------------------------------
    # Decoding — fix non-ASCII mangling
    # ------------------------------------------------------------------

    @staticmethod
    def _decode(r: requests.Response) -> str:
        """
        Decode response body with UTF-8 preference.

        Modern sites (including docs.python.org) serve UTF-8 without a
        charset in the Content-Type header. The requests library then
        falls back to ISO-8859-1, turning 'Ł' into 'Å' + control char.
        We fix this by assuming UTF-8 unless the header says otherwise.
        """
        content_type = (r.headers.get("content-type") or "").lower()
        has_explicit_charset = "charset=" in content_type

        if not has_explicit_charset:
            r.encoding = "utf-8"

        html = r.text

        # If UTF-8 produced replacement characters, retry with detected encoding
        if "\ufffd" in html:
            apparent = r.apparent_encoding
            if apparent and apparent.lower() not in ("utf-8", "ascii"):
                r.encoding = apparent
                html = r.text

        return html

    # ------------------------------------------------------------------
    # HTML → text
    # ------------------------------------------------------------------

    @classmethod
    def _clean_html(cls, html: str) -> str | None:
        if not html or len(html) < 200:
            return None

        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as e:  # noqa: BLE001
            log.debug("HTML parse failed: %s", e)
            return None

        # 1. Drop HTML comments — often contain boilerplate / tracking text
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.extract()

        # 2. Pick best content container
        container = cls._pick_container(soup)

        # 3. Preserve code blocks BEFORE stripping anything else
        code_blocks = cls._extract_code_blocks(container)

        # 4. Strip junk tags
        for tag in container(list(_STRIP_TAGS)):
            tag.decompose()

        # 5. Extract text with normalized whitespace
        text = container.get_text(separator=" ", strip=True)
        text = " ".join(text.split())

        # 6. Restore code blocks with their original formatting
        text = cls._restore_code_blocks(text, code_blocks)

        if len(text) < _MIN_TEXT_LENGTH:
            return None

        return text[: CONFIG.max_chars_per_page]

    @staticmethod
    def _pick_container(soup: BeautifulSoup):
        """
        Prefer <article>/<main> when they hold enough text; otherwise
        return the full soup.
        """
        for tag_name in ("article", "main"):
            candidate = soup.find(tag_name)
            if candidate is None:
                continue
            length = len(candidate.get_text(strip=True))
            if length >= _MIN_CONTAINER_LENGTH:
                return candidate
        return soup

    @staticmethod
    def _extract_code_blocks(container) -> list[tuple[str, str]]:
        """
        Replace <pre> and inline <code> elements with unique markers,
        saving their original text (with newlines intact).

        Returns a list of (marker, code_text) tuples.
        """
        code_blocks: list[tuple[str, str]] = []

        # Fenced blocks: <pre>...</pre>
        for i, pre in enumerate(container.find_all("pre")):
            code = pre.get_text()
            if not code.strip():
                continue
            marker = f"@@ONYX_PRE_{i}@@"
            code_blocks.append((marker, code))
            pre.replace_with(marker)

        # Inline code: <code>...</code> not already inside <pre>
        for j, code_tag in enumerate(container.find_all("code")):
            text = code_tag.get_text()
            if not text.strip():
                continue
            marker = f"@@ONYX_CODE_{j}@@"
            code_blocks.append((marker, text))
            code_tag.replace_with(marker)

        return code_blocks

    @staticmethod
    def _restore_code_blocks(text: str, code_blocks: list[tuple[str, str]]) -> str:
        """Substitute markers back with formatted code."""
        for marker, code in code_blocks:
            code_clean = code.strip()
            if not code_clean:
                text = text.replace(marker, " ")
                continue

            if "\n" in code_clean:
                # Multi-line → markdown fenced block
                replacement = f" ```\n{code_clean}\n``` "
            else:
                # Single line → inline code
                replacement = f" `{code_clean}` "

            text = text.replace(marker, replacement)

        return text

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
