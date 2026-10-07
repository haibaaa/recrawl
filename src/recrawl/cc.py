"""Common Crawl snapshot labels: an external oracle for page changes.

For a URL captured in two monthly crawls, the payload digests recorded by
Common Crawl say whether the raw payload changed; we additionally run our own
shingle detector on the extracted main text of both snapshots.
"""

from __future__ import annotations

import gzip
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
from warcio.archiveiterator import ArchiveIterator

from recrawl.config import FETCH_TIMEOUT_SECONDS, JACCARD_TAU, SHINGLE_K, USER_AGENT
from recrawl.detect import content_sha, detect_change, strip_volatile
from recrawl.extract import extract
from recrawl.normalize import host_of

INDEX_API = "https://index.commoncrawl.org/{crawl}-index"
DATA_API = "https://data.commoncrawl.org/{filename}"

# Monthly crawls to compare (verified to exist and cover news homepages).
CRAWLS: tuple[str, ...] = ("CC-MAIN-2026-30", "CC-MAIN-2026-34", "CC-MAIN-2026-39")

_INDEX_DELAY_S = 0.15


@dataclass
class Capture:
    """One WARC record reference for a URL in one crawl."""

    crawl: str
    url: str
    timestamp: str
    digest: str
    filename: str
    offset: int
    length: int
    status: int | None


@dataclass
class SnapshotPair:
    """Change labels for one URL observed in two crawls."""

    url: str
    crawl_a: str
    crawl_b: str
    timestamp_a: str
    timestamp_b: str
    digest_changed: bool
    sim: float | None
    pred_changed: bool | None
    stripped_changed: bool | None


def parse_index_lines(text: str, crawl: str) -> list[Capture]:
    """Parse the ND-JSON body of an index query into captures (HTTP 200 only)."""
    captures: list[Capture] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("status") != "200":
            continue
        captures.append(
            Capture(
                crawl=crawl,
                url=record.get("url", ""),
                timestamp=record.get("timestamp", ""),
                digest=record.get("digest", ""),
                filename=record.get("filename", ""),
                offset=int(record.get("offset", 0)),
                length=int(record.get("length", 0)),
                status=200,
            )
        )
    return captures


def query_captures(session: requests.Session, crawl: str, url: str) -> list[Capture]:
    """Fetch index entries for *url* in *crawl*; empty when not captured."""
    delay = _INDEX_DELAY_S
    for _attempt in range(4):
        try:
            response = session.get(
                INDEX_API.format(crawl=crawl),
                params={"url": url, "output": "json"},
                timeout=FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": USER_AGENT},
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError):
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code == 200:
            return parse_index_lines(response.text, crawl)
        if response.status_code == 404:
            return []
        if response.status_code < 500:
            return []
        time.sleep(delay)
        delay *= 2
    return []


def fetch_payload(session: requests.Session, capture: Capture) -> str | None:
    """Download the WARC record for *capture* and return the HTTP body as text."""
    start = capture.offset
    end = capture.offset + capture.length - 1
    for attempt in range(3):
        try:
            response = session.get(
                DATA_API.format(filename=capture.filename),
                headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
                timeout=FETCH_TIMEOUT_SECONDS * 2,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, OSError):
            if attempt < 2:
                time.sleep(2**attempt)
            continue
        if response.status_code == 206:
            return _body_from_warc(response.content)
        if response.status_code in (429, 503):
            time.sleep(2**attempt)
            continue
        return None
    return None


def _body_from_warc(warc_bytes: bytes) -> str | None:
    try:
        for record in ArchiveIterator(io.BytesIO(warc_bytes)):
            if record.rec_type != "response":
                continue
            payload = record.content_stream().read()
            encoding = ""
            if record.http_headers is not None:
                encoding = (record.http_headers.get_header("Content-Encoding") or "").lower()
            if encoding == "gzip":
                payload = gzip.decompress(payload)
            charset = "utf-8"
            if record.http_headers is not None:
                content_type = record.http_headers.get_header("Content-Type") or ""
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
            return payload.decode(charset, errors="replace")
    except (OSError, EOFError, gzip.BadGzipFile):
        return None
    return None


def label_pair(
    url: str,
    first: Capture,
    second: Capture,
    first_html: str,
    second_html: str,
    tau: float = JACCARD_TAU,
) -> SnapshotPair:
    """Compute change labels between two snapshots of *url*."""
    extracted_a = extract(first_html, url).text
    extracted_b = extract(second_html, url).text
    if extracted_a and extracted_b:
        pred_changed, sim = detect_change(extracted_a, extracted_b, tau, SHINGLE_K)
        stripped_changed = content_sha(strip_volatile(extracted_a)) != content_sha(
            strip_volatile(extracted_b)
        )
    else:
        pred_changed = None
        sim = None
        stripped_changed = None
    return SnapshotPair(
        url=url,
        crawl_a=first.crawl,
        crawl_b=second.crawl,
        timestamp_a=first.timestamp,
        timestamp_b=second.timestamp,
        digest_changed=first.digest != second.digest,
        sim=sim,
        pred_changed=pred_changed,
        stripped_changed=stripped_changed,
    )


# Hosts whose pages Common Crawl's recent crawls actually capture; others
# (BBC, Guardian, DW) are largely absent from the index, so they can never
# produce pairs no matter how durable their paths are.
_CC_CAPTURED_HOSTS: frozenset[str] = frozenset({"aljazeera.com", "hindustantimes.com"})


def durable_candidates(
    urls: list[str], limit: int, captured_hosts: frozenset[str] = _CC_CAPTURED_HOSTS
) -> list[str]:
    """Shallow, lean paths first; URLs whose host CC captures get priority."""

    def depth(url: str) -> tuple[int, int]:
        path = url.split("/", 3)
        segments = len(path) - 3 if len(path) > 3 else 0
        return (max(segments, 0), len(url))

    def captured(url: str) -> int:
        return 0 if host_of(url).removeprefix("www.") in captured_hosts else 1

    return sorted(set(urls), key=lambda url: (captured(url), *depth(url)))[:limit]


def run_cc_labels(
    urls: list[str],
    session: requests.Session,
    out_path: Path,
    crawls: tuple[str, ...] = CRAWLS,
) -> list[SnapshotPair]:
    """Label every URL captured in at least two of *crawls*; write JSONL to *out_path*."""
    pairs: list[SnapshotPair] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for url in urls:
            captures: dict[str, Capture] = {}
            for crawl in crawls:
                found = query_captures(session, crawl, url)
                time.sleep(_INDEX_DELAY_S)
                if found:
                    captures[crawl] = found[0]
            if len(captures) < 2:
                continue
            ordered = [captures[crawl] for crawl in crawls if crawl in captures]
            payloads: dict[str, str] = {}
            for capture in ordered:
                html = fetch_payload(session, capture)
                if html is None:
                    break
                payloads[capture.crawl] = html
            if len(payloads) < 2:
                continue
            present = [capture for capture in ordered if capture.crawl in payloads]
            for first, second in zip(present, present[1:], strict=False):
                pair = label_pair(url, first, second, payloads[first.crawl], payloads[second.crawl])
                pairs.append(pair)
                handle.write(json.dumps(asdict(pair)) + "\n")
    return pairs


def load_pairs(path: Path) -> list[SnapshotPair]:
    """Read pairs previously written by :func:`run_cc_labels`."""
    pairs: list[SnapshotPair] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            pairs.append(SnapshotPair(**record))
    return pairs
