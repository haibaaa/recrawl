"""robots.txt handling with per-host caching."""

from __future__ import annotations

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import requests

from recrawl.config import FETCH_TIMEOUT_SECONDS, USER_AGENT


class RobotsCache:
    """Per-host robots.txt cache. A missing or unreadable robots.txt allows all."""

    def __init__(self, session: requests.Session, user_agent: str = USER_AGENT) -> None:
        self._session = session
        self._user_agent = user_agent
        self._cache: dict[str, RobotFileParser | None] = {}
        self._delays: dict[str, float] = {}

    def can_fetch(self, url: str) -> bool:
        """Whether the bot may fetch *url* according to its host's robots.txt."""
        host, robots = self._load(url)
        if robots is None:
            return True
        return robots.can_fetch(self._user_agent, url)

    def crawl_delay(self, url: str) -> float | None:
        """Crawl-delay declared for this bot on *url*'s host, if any."""
        host, robots = self._load(url)
        if robots is None:
            return None
        delay = robots.crawl_delay(self._user_agent)
        return float(delay) if delay is not None else None

    def _load(self, url: str) -> tuple[str, RobotFileParser | None]:
        parts = urlsplit(url)
        host = parts.netloc.lower()
        if host in self._cache:
            return host, self._cache[host]

        robots_url = f"{parts.scheme}://{host}/robots.txt"
        parser: RobotFileParser | None = None
        try:
            response = self._session.get(
                robots_url,
                timeout=FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": self._user_agent},
            )
            if response.status_code == 200:
                parser = RobotFileParser()
                parser.parse(response.text.splitlines())
                delay = parser.crawl_delay(self._user_agent)
                if delay is not None:
                    self._delays[host] = float(delay)
            elif response.status_code in (401, 403):
                parser = RobotFileParser()
                parser.parse(["User-agent: *", "Disallow: /"])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError):
            parser = None

        self._cache[host] = parser
        return host, parser
