"""URL normalisation for the crawl frontier and the content-seen check."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING_PARAMS = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "fbclid",
        "gclid",
        "mc_cid",
        "mc_eid",
        "ref",
        "cmpid",
        "at_campaign",
        "at_medium",
        "traffic_source",
    }
)

_SESSION_PARAM_NAMES = frozenset({"jsessionid", "sid", "phpsessid", "asp_sessionid"})

_DEFAULT_PORTS = {"http": "", "https": ""}
_MAX_URL_LENGTH = 2048


def normalize_url(url: str) -> str:
    """Return the canonical form of *url*: lowercased scheme/host, no fragment,
    no tracking or session parameters, sorted query, no default port, no
    duplicate slashes.

    Raises:
        ValueError: if *url* has no network location component.
    """
    parts = urlsplit(url.strip())
    if not parts.netloc:
        raise ValueError(f"URL has no host: {url!r}")

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported scheme in URL: {url!r}")

    netloc = parts.netloc.lower()
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[: -len(":80")]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[: -len(":443")]

    path = _clean_path(parts.path)
    query = _clean_query(parts.query)
    normalized = urlunsplit((scheme, netloc, path, query, ""))
    if len(normalized) > _MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds {_MAX_URL_LENGTH} characters: {url!r}")
    return normalized


def _clean_path(path: str) -> str:
    while "//" in path:
        path = path.replace("//", "/")
    if path.endswith("/index.html") or path.endswith("/index.htm"):
        path = path[: -len("index.html") if path.endswith(".html") else -len("index.htm")]
    return path or "/"


def _clean_query(query: str) -> str:
    pairs = [
        (key, value)
        for key, value in parse_qsl(query, keep_blank_values=True)
        if key.lower() not in _TRACKING_PARAMS and key.lower() not in _SESSION_PARAM_NAMES
    ]
    pairs.sort()
    return urlencode(pairs)


def host_of(url: str) -> str:
    """Return the lowercased host component of *url*."""
    return urlsplit(url).netloc.lower()
