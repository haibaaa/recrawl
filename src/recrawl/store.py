"""SQLite storage: page table, fetch log, content versions, backlinks."""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from recrawl.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    discovered_ts REAL NOT NULL,
    first_fetch_ts REAL,
    last_fetch_ts REAL,
    last_change_ts REAL,
    fetch_count INTEGER NOT NULL DEFAULT 0,
    change_count INTEGER NOT NULL DEFAULT 0,
    status INTEGER,
    importance REAL NOT NULL DEFAULT 0.0,
    title TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL DEFAULT '',
    text_sha TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS fetch_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL,
    ts REAL NOT NULL,
    status INTEGER,
    bytes INTEGER NOT NULL DEFAULT 0,
    sha_raw TEXT,
    sha_stripped TEXT,
    truth_changed INTEGER,
    pred_changed INTEGER,
    sim REAL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_fetch_log_url_ts ON fetch_log (url, ts);
CREATE TABLE IF NOT EXISTS versions (
    url TEXT NOT NULL,
    ts REAL NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (url, ts)
);
CREATE TABLE IF NOT EXISTS backlinks (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    PRIMARY KEY (src, dst)
);
CREATE TABLE IF NOT EXISTS passes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    fetched INTEGER NOT NULL DEFAULT 0,
    changed INTEGER NOT NULL DEFAULT 0,
    errors INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class PageRow:
    """A tracked URL and its latest known state."""

    url: str
    host: str
    discovered_ts: float
    first_fetch_ts: float | None
    last_fetch_ts: float | None
    last_change_ts: float | None
    fetch_count: int
    change_count: int
    status: int | None
    importance: float
    title: str
    text: str
    text_sha: str


@dataclass
class FetchLogRow:
    """One observed fetch of a URL in the history."""

    url: str
    ts: float
    status: int | None
    truth_changed: int | None
    pred_changed: int | None
    sim: float | None


class Store:
    """Thin data-access layer over the crawl database."""

    def __init__(self, path: Path = DB_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_page(self, url: str, host: str, ts: float | None = None) -> None:
        """Register *url* for tracking if it is not already tracked."""
        self.conn.execute(
            "INSERT OR IGNORE INTO pages (url, host, discovered_ts) VALUES (?, ?, ?)",
            (url, host, ts if ts is not None else time.time()),
        )
        self.conn.commit()

    def get_page(self, url: str) -> PageRow | None:
        """Return the page row for *url*, or None if untracked."""
        row = self.conn.execute("SELECT * FROM pages WHERE url = ?", (url,)).fetchone()
        return _page_from_row(row) if row else None

    def all_pages(self) -> list[PageRow]:
        """Return every tracked page."""
        rows = self.conn.execute("SELECT * FROM pages").fetchall()
        return [_page_from_row(row) for row in rows]

    def page_count(self) -> int:
        """Number of tracked pages."""
        return int(self.conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0])

    def fetched_pages(self) -> list[PageRow]:
        """Return pages that have been fetched at least once."""
        rows = self.conn.execute("SELECT * FROM pages WHERE last_fetch_ts IS NOT NULL").fetchall()
        return [_page_from_row(row) for row in rows]

    def record_fetch(
        self,
        url: str,
        host: str,
        ts: float,
        status: int | None,
        nbytes: int,
        title: str,
        text: str,
        sha_raw: str,
        sha_stripped: str,
        truth_changed: bool | None,
        pred_changed: bool | None,
        sim: float | None,
        error: str | None,
    ) -> bool:
        """Log a fetch attempt and update page state.

        Returns True when the extracted content changed relative to the
        previously stored version.
        """
        changed = bool(truth_changed)
        self.conn.execute(
            """
            INSERT INTO fetch_log
                (url, ts, status, bytes, sha_raw, sha_stripped,
                 truth_changed, pred_changed, sim, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                ts,
                status,
                nbytes,
                sha_raw,
                sha_stripped,
                None if truth_changed is None else int(truth_changed),
                None if pred_changed is None else int(pred_changed),
                sim,
                error,
            ),
        )
        if status == 200 and text:
            previous_sha = self._text_sha(url)
            if previous_sha != sha_raw:
                self.conn.execute(
                    "INSERT OR REPLACE INTO versions (url, ts, text) VALUES (?, ?, ?)",
                    (url, ts, text),
                )
            self.conn.execute(
                """
                UPDATE pages SET
                    first_fetch_ts = COALESCE(first_fetch_ts, ?),
                    last_fetch_ts = ?,
                    last_change_ts = CASE WHEN ? THEN ? ELSE last_change_ts END,
                    fetch_count = fetch_count + 1,
                    change_count = change_count + CASE WHEN ? THEN 1 ELSE 0 END,
                    status = ?, title = ?, text = ?, text_sha = ?
                WHERE url = ?
                """,
                (
                    ts,
                    ts,
                    int(changed),
                    ts,
                    int(changed),
                    status,
                    title,
                    text,
                    sha_raw,
                    url,
                ),
            )
        else:
            self.conn.execute(
                """
                UPDATE pages SET last_fetch_ts = ?, fetch_count = fetch_count + 1, status = ?
                WHERE url = ?
                """,
                (ts, status, url),
            )
        self.conn.commit()
        return changed

    def _text_sha(self, url: str) -> str | None:
        row = self.conn.execute("SELECT text_sha FROM pages WHERE url = ?", (url,)).fetchone()
        return row["text_sha"] if row else None

    def add_backlink(self, src: str, dst: str) -> None:
        """Record a link from *src* to *dst* (deduplicated)."""
        self.conn.execute("INSERT OR IGNORE INTO backlinks (src, dst) VALUES (?, ?)", (src, dst))
        self.conn.commit()

    def record_pass(self, ts: float, fetched: int, changed: int, errors: int) -> None:
        """Record a completed crawl pass (used as simulation rounds)."""
        self.conn.execute(
            "INSERT INTO passes (ts, fetched, changed, errors) VALUES (?, ?, ?, ?)",
            (ts, fetched, changed, errors),
        )
        self.conn.commit()

    def pass_times(self) -> list[float]:
        """Timestamps of all completed crawl passes, in order."""
        rows = self.conn.execute("SELECT ts FROM passes ORDER BY ts").fetchall()
        return [row["ts"] for row in rows]

    def recompute_importance(self) -> None:
        """Set each page's static quality g(d) from backlink count and host weight.

        g(d) = log(1 + inlinks) * host_weight, min-max normalised over the corpus.
        """
        rows = self.conn.execute(
            """
            SELECT p.url, p.host, COUNT(b.src) AS inlinks
            FROM pages p LEFT JOIN backlinks b ON b.dst = p.url
            GROUP BY p.url
            """
        ).fetchall()
        if not rows:
            return
        hosts = sorted({row["host"] for row in rows})
        host_weight = {host: 1.0 + 0.5 * i for i, host in enumerate(hosts)}
        raw = {
            row["url"]: math.log1p(row["inlinks"] or 0) * host_weight.get(row["host"], 1.0)
            for row in rows
        }
        lo, hi = min(raw.values()), max(raw.values())
        span = (hi - lo) or 1.0
        for url, value in raw.items():
            self.conn.execute(
                "UPDATE pages SET importance = ? WHERE url = ?", ((value - lo) / span, url)
            )
        self.conn.commit()

    def fetch_history(self) -> list[FetchLogRow]:
        """Return the full observed fetch history ordered by time."""
        rows = self.conn.execute(
            "SELECT url, ts, status, truth_changed, pred_changed, sim FROM fetch_log ORDER BY ts"
        ).fetchall()
        return [
            FetchLogRow(
                url=row["url"],
                ts=row["ts"],
                status=row["status"],
                truth_changed=row["truth_changed"],
                pred_changed=row["pred_changed"],
                sim=row["sim"],
            )
            for row in rows
        ]

    def version_text(self, url: str, at_ts: float) -> str | None:
        """Return the stored content version of *url* valid at *at_ts*."""
        row = self.conn.execute(
            "SELECT text FROM versions WHERE url = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
            (url, at_ts),
        ).fetchone()
        return row["text"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a metadata entry."""
        self.conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        """Return a metadata value, or None if absent."""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def _page_from_row(row: sqlite3.Row) -> PageRow:
    return PageRow(
        url=row["url"],
        host=row["host"],
        discovered_ts=row["discovered_ts"],
        first_fetch_ts=row["first_fetch_ts"],
        last_fetch_ts=row["last_fetch_ts"],
        last_change_ts=row["last_change_ts"],
        fetch_count=row["fetch_count"],
        change_count=row["change_count"],
        status=row["status"],
        importance=row["importance"],
        title=row["title"],
        text=row["text"],
        text_sha=row["text_sha"],
    )
