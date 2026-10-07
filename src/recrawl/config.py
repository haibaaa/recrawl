"""Shared configuration: paths, politeness settings, detection thresholds."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RECRAWL_DATA", PROJECT_ROOT / "data"))

DB_PATH = DATA_DIR / "recrawl.db"
INDEX_PATH = DATA_DIR / "index.pkl"
CC_DIR = DATA_DIR / "cc"
CC_LABELS_PATH = CC_DIR / "labels.jsonl"
REPORTS_DIR = DATA_DIR / "reports"

BOT_NAME = "recrawl-research-bot"
# Identifying UA with contact, per robots.txt etiquette and the assignment's data-ethics rule.
USER_AGENT = f"{BOT_NAME}/0.1 (CSD358 IR hackathon; +mailto:9700samarth@gmail.com)"

HOST_DELAY_SECONDS = 1.5
FETCH_TIMEOUT_SECONDS = 15
FETCH_RETRIES = 2
CRAWL_PASS_INTERVAL_SECONDS = 3600
DEFAULT_BUDGET_PER_PASS = 400

# Change detection: predict "changed" when shingle similarity falls below this.
JACCARD_TAU = 0.85
SHINGLE_K = 5

# RSS seeds (international, robots-friendly; Reuters excluded as bot-hostile,
# NPR dropped: it refuses connections from this network).
RSS_SEEDS: tuple[str, ...] = (
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/technology/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://rss.dw.com/rdf/rss-en-all",
)

HOME_SEEDS: tuple[str, ...] = (
    "https://www.bbc.com/news",
    "https://www.theguardian.com/world",
    "https://www.aljazeera.com/news/",
    "https://www.dw.com/en/news/s-100940",
)
