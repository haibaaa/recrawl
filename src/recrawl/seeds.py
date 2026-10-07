"""RSS and homepage seed discovery."""

from __future__ import annotations

from urllib.parse import urlsplit

import feedparser
import requests

from recrawl.config import FETCH_TIMEOUT_SECONDS, HOME_SEEDS, RSS_SEEDS, USER_AGENT
from recrawl.normalize import normalize_url

_ALLOWED_SUFFIXES: tuple[str, ...] = (
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "aljazeera.com",
    "dw.com",
)


def is_allowed_host(url: str) -> bool:
    """Whether *url* belongs to the focused-crawl host allowlist."""
    host = urlsplit(url).netloc.lower().split(":")[0]
    return any(host == suffix or host.endswith("." + suffix) for suffix in _ALLOWED_SUFFIXES)


def collect_seeds(session: requests.Session | None = None) -> list[str]:
    """Normalised seed URLs from RSS feeds and homepages.

    Feed fetch failures are skipped; a session can be injected for tests.
    """
    own_session = session is None
    session = session or requests.Session()
    urls: list[str] = []
    for feed_url in RSS_SEEDS:
        try:
            response = session.get(
                feed_url,
                timeout=FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
            if response.status_code != 200:
                continue
            parsed = feedparser.parse(response.content)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError):
            continue
        for entry in parsed.entries:
            link = entry.get("link", "")
            if link:
                urls.append(link)
    if own_session:
        session.close()

    for home in HOME_SEEDS:
        urls.append(home)

    allowed = [url for url in urls if is_allowed_host(url)]
    return _dedupe_normalize(allowed)


def _dedupe_normalize(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        try:
            normalized = normalize_url(url)
        except ValueError:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result
