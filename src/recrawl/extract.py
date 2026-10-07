"""Main-content extraction and link discovery from raw HTML."""

from __future__ import annotations

import html as html_unescape
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import trafilatura

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_HREF_RE = re.compile(r"""href\s*=\s*["']([^"'<> ]+)["']""", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


@dataclass
class Extracted:
    """Main text and title pulled out of a HTML document."""

    title: str
    text: str


def extract(html: str, url: str) -> Extracted:
    """Extract main text and title from *html*.

    Falls back to a crude tag-strip when the boilerplate remover finds no body.
    """
    text = trafilatura.extract(html, url=url, include_comments=False)
    if not text:
        text = _strip_tags(html)
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = (metadata.title or "").strip() if metadata else ""
    return Extracted(title=title, text=_WS_RE.sub(" ", text).strip())


def _strip_tags(html: str) -> str:
    html = _SCRIPT_STYLE_RE.sub(" ", html)
    html = _TAG_RE.sub(" ", html)
    return html_unescape.unescape(html)


def extract_links(html: str, base_url: str) -> list[str]:
    """Absolute URLs of every anchor href in *html*, deduplicated in order."""
    links: list[str] = []
    seen: set[str] = set()
    for href in _HREF_RE.findall(html):
        if href.startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        absolute = urljoin(base_url, href)
        if urlsplit(absolute).scheme not in ("http", "https"):
            continue
        if absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return links
