"""HTTP fetching with retries and timeouts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from recrawl.config import FETCH_RETRIES, FETCH_TIMEOUT_SECONDS, USER_AGENT


@dataclass
class FetchResult:
    """Outcome of a single HTTP GET."""

    url: str
    status: int | None
    html: str
    final_url: str
    nbytes: int
    error: str | None


def fetch(
    url: str,
    session: requests.Session,
    timeout: float = FETCH_TIMEOUT_SECONDS,
    retries: int = FETCH_RETRIES,
) -> FetchResult:
    """GET *url*, retrying transient network failures up to *retries* times.

    Raises:
        requests.RequestException: after all retries fail, or on non-retryable
            request failures.
    """
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            encoding = response.encoding or "utf-8"
            text = response.content.decode(encoding, errors="replace")
            return FetchResult(
                url=url,
                status=response.status_code,
                html=text if response.status_code == 200 else "",
                final_url=str(response.url),
                nbytes=len(response.content),
                error=None,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise requests.exceptions.ConnectionError(
        f"fetch failed after {retries + 1} attempts: {url}"
    ) from last_error
