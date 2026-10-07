"""Command-line interface for the recrawl pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

from recrawl.config import (
    CC_LABELS_PATH,
    CRAWL_PASS_INTERVAL_SECONDS,
    DATA_DIR,
    DB_PATH,
    DEFAULT_BUDGET_PER_PASS,
    INDEX_PATH,
    REPORTS_DIR,
    USER_AGENT,
)
from recrawl.crawl import run_loop, run_pass
from recrawl.normalize import host_of
from recrawl.robots import RobotsCache
from recrawl.search import DEFAULT_ALPHA, DEFAULT_BETA
from recrawl.seeds import collect_seeds
from recrawl.store import Store


def main(argv: list[str] | None = None) -> int:
    """Parse *argv* (default: sys.argv) and dispatch a subcommand. Returns an exit code."""
    parser = argparse.ArgumentParser(prog="recrawl", description="Freshness-aware recrawl system")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seeds", help="Discover seed URLs from RSS feeds and homepages")

    crawl_p = sub.add_parser("crawl", help="Run a single crawl pass")
    crawl_p.add_argument("--budget", type=int, default=DEFAULT_BUDGET_PER_PASS)
    crawl_p.add_argument("--policy", default="proposed", choices=["rr", "importance", "proposed"])
    crawl_p.add_argument("--cap", type=int, default=500)

    loop_p = sub.add_parser("loop", help="Run crawl passes repeatedly")
    loop_p.add_argument("--budget", type=int, default=DEFAULT_BUDGET_PER_PASS)
    loop_p.add_argument("--policy", default="proposed", choices=["rr", "importance", "proposed"])
    loop_p.add_argument("--cap", type=int, default=500)
    loop_p.add_argument("--interval", type=float, default=CRAWL_PASS_INTERVAL_SECONDS)
    loop_p.add_argument(
        "--passes", type=int, default=None, help="Stop after N passes (default: forever)"
    )

    sub.add_parser("status", help="Print crawl statistics")

    cc_p = sub.add_parser("cc-label", help="Build change labels from Common Crawl snapshots")
    cc_p.add_argument("--limit", type=int, default=120, help="Max candidate URLs to query")

    sub.add_parser("detect-eval", help="Detector P/R: live truth + CC digests")

    sim_p = sub.add_parser("simulate", help="Freshness-vs-budget curves on live history")
    sim_p.add_argument("--budgets", default="2,5,10,20,40")

    sub.add_parser("index", help="Build the search index from tracked pages")

    search_p = sub.add_parser("search", help="Run a ranked query against the index")
    search_p.add_argument("query", nargs="+")
    search_p.add_argument("--k", type=int, default=10)
    search_p.add_argument(
        "--bare", action="store_true", help="Cosine only: no zones in g(d) mixing"
    )
    search_p.add_argument("--bm25", action="store_true", help="Also print the BM25 ranking")

    eval_p = sub.add_parser("eval-search", help="P@k ablation over a judged JSONL query set")
    eval_p.add_argument("queryset", type=Path, help="Path to judged queries (JSONL)")
    eval_p.add_argument("--k", type=int, default=5, help="Rank cutoff")

    inspect_p = sub.add_parser("inspect", help="Inspect intermediate IR postings and weights")
    inspect_p.add_argument("term", help="Term to inspect in inverted index")
    inspect_p.add_argument("--k", type=int, default=10, help="Max postings to display")

    args = parser.parse_args(argv)
    if args.command == "seeds":
        return _cmd_seeds()
    if args.command == "crawl":
        return _cmd_crawl(args)
    if args.command == "loop":
        return _cmd_loop(args)
    if args.command == "status":
        return _cmd_status()
    if args.command == "cc-label":
        return _cmd_cc_label(args)
    if args.command == "detect-eval":
        return _cmd_detect_eval()
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "index":
        return _cmd_index()
    if args.command == "search":
        return _cmd_search(args)
    if args.command == "eval-search":
        return _cmd_eval_search(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _cmd_seeds() -> int:
    urls = collect_seeds()
    store = Store()
    for url in urls:
        store.add_page(url, host_of(url))
    print(f"discovered {len(urls)} seed URLs; database now tracks {store.page_count()} pages")
    store.close()
    return 0


def _cmd_crawl(args: argparse.Namespace) -> int:
    store = Store()
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        robots = RobotsCache(session)
        stats = run_pass(store, session, robots, args.budget, args.policy, args.cap)
    print(
        f"pass: fetched={stats.fetched} changed={stats.changed} errors={stats.errors} "
        f"robots_blocked={stats.robots_blocked} discovered={stats.discovered} "
        f"duration={stats.duration_s:.1f}s"
    )
    store.close()
    return 0


def _cmd_loop(args: argparse.Namespace) -> int:
    store = Store()
    print(
        f"starting loop: interval={args.interval:.0f}s passes={args.passes} "
        f"budget={args.budget} policy={args.policy}",
        flush=True,
    )
    results = run_loop(store, args.budget, args.policy, args.cap, args.interval, args.passes)
    for i, stats in enumerate(results, start=1):
        print(
            f"pass {i}: fetched={stats.fetched} changed={stats.changed} "
            f"errors={stats.errors} blocked={stats.robots_blocked} "
            f"discovered={stats.discovered} duration={stats.duration_s:.1f}s",
            flush=True,
        )
    store.close()
    return 0


def _cmd_cc_label(args: argparse.Namespace) -> int:
    from recrawl.cc import durable_candidates, run_cc_labels

    store = Store()
    urls = [page.url for page in store.all_pages() if page.last_fetch_ts is not None]
    store.close()
    candidates = durable_candidates(urls, args.limit)
    print(f"querying Common Crawl for {len(candidates)} candidate URLs ...", flush=True)
    pairs = run_cc_labels(candidates, requests.Session(), CC_LABELS_PATH)
    changed = sum(1 for pair in pairs if pair.digest_changed)
    print(f"labelled {len(pairs)} snapshot pairs ({changed} with payload changes)")
    print(f"written to {CC_LABELS_PATH}")
    return 0


def _cmd_detect_eval() -> int:
    from recrawl.cc import load_pairs
    from recrawl.evaluation import cc_summary, live_detector_scores, write_csv

    store = Store()
    live = live_detector_scores(store)
    store.close()
    print("detector vs live volatile-stripped truth")
    rows: list[dict[str, object]] = []
    for method, detector in live.items():
        print(
            f"  {method:12s} n={detector.n:4d} P={detector.precision:.3f} "
            f"R={detector.recall:.3f} F1={detector.f1:.3f} acc={detector.accuracy:.3f}"
        )
        rows.append(
            {
                "source": "live",
                "method": method,
                "n": detector.n,
                "precision": detector.precision,
                "recall": detector.recall,
                "f1": detector.f1,
                "accuracy": detector.accuracy,
            }
        )
    if CC_LABELS_PATH.exists():
        summary = cc_summary(load_pairs(CC_LABELS_PATH))
        print(
            f"detector vs CC payload digests "
            f"(pairs={summary.total_pairs} digest_changed={summary.digest_changed} "
            f"payload_only={summary.payload_only_changes})"
        )
        for row in summary.rows:
            print(
                f"  {row['method']:12s} n={row['n']:4d} P={row['precision']:.3f} "
                f"R={row['recall']:.3f} F1={row['f1']:.3f} acc={row['accuracy']:.3f}"
            )
            rows.append({"source": "cc", **row})
    else:
        print(f"no CC labels yet at {CC_LABELS_PATH} (run cc-label first)")
    write_csv(rows, REPORTS_DIR / "detect.csv")
    print(f"written to {REPORTS_DIR / 'detect.csv'}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    from recrawl.evaluation import freshness_curves, write_csv

    budgets = [int(part) for part in args.budgets.split(",") if part.strip()]
    store = Store()
    rows = freshness_curves(store, budgets)
    store.close()
    cols = (
        ("policy", 10, ""),
        ("budg/pass", 9, ""),
        ("fresh%", 7, ""),
        ("missed", 7, ""),
        ("age-hrs", 9, ""),
        ("fetched", 8, ""),
    )
    header = " ".join(f"{label:>{width}s}" for label, width, _ in cols)
    print(header)
    for row in rows:
        print(
            f"{row['policy']:10s} {row['budget']:>9} {100 * float(row['fresh_fraction']):>6.1f}% "
            f"{row['missed_changes']:>7} {row['stale_change_hours']:>9.0f} "
            f"{row['fetches']:>8}"
        )
    write_csv(rows, REPORTS_DIR / "freshness.csv")
    print(f"written to {REPORTS_DIR / 'freshness.csv'}")
    return 0


def _cmd_index() -> int:
    from recrawl.indexer import build_index, save_index

    store = Store()
    index = build_index(store)
    store.close()
    save_index(index, INDEX_PATH)
    print(f"indexed {index.doc_count} documents, {len(index.postings)} distinct terms")
    print(f"written to {INDEX_PATH}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from recrawl.indexer import load_index
    from recrawl.search import parse_query, search_bm25, search_lnc_ltc

    raw = " ".join(args.query)
    query = parse_query(raw)
    index = load_index(INDEX_PATH)
    alpha, beta = (0.0, 0.0) if args.bare else (DEFAULT_ALPHA, DEFAULT_BETA)
    hits = search_lnc_ltc(index, query, args.k, alpha=alpha, beta=beta)
    print(f"query: {raw}  (docs={index.doc_count} terms={len(index.postings)})")
    if args.bare:
        print(f"{'#':>2s} {'cos':>6s} {'url':<60s} title")
    else:
        print(f"{'#':>2s} {'cos':>6s} {'net':>6s} {'url':<60s} title")
    for rank, hit in enumerate(hits, start=1):
        if args.bare:
            print(f"{rank:>2d} {hit.cosine:.4f} {hit.url:<60.60s} {hit.title[:60]}")
        else:
            print(f"{rank:>2d} {hit.cosine:.4f} {hit.net:.4f} {hit.url:<60.60s} {hit.title[:60]}")
    if args.bm25:
        print("\nBM25 baseline:")
        for rank, hit in enumerate(search_bm25(index, query, args.k), start=1):
            print(f"{rank:>2d} {hit.bm25:.4f} {hit.url:<60.60s} {hit.title[:60]}")
    return 0


def _cmd_eval_search(args: argparse.Namespace) -> int:
    from recrawl.evaluation import search_eval, write_csv
    from recrawl.indexer import load_index

    index = load_index(INDEX_PATH)
    result = search_eval(index, args.queryset, k=args.k)
    header = f"{'method':12s} {'mean P@1':>9s} {'mean P@3':>9s} {'mean P@5':>9s}"
    print(header)
    for row in result.summary:
        print(f"{row['method']:12s} {row['mean_p@1']:>9} {row['mean_p@3']:>9} {row['mean_p@5']:>9}")
    write_csv(result.summary, REPORTS_DIR / "search_eval_summary.csv")
    write_csv(result.per_query, REPORTS_DIR / "search_eval_per_query.csv")
    print(f"written to {REPORTS_DIR / 'search_eval_summary.csv'}")
    return 0


def _cmd_status() -> int:
    store = Store()
    pages = store.all_pages()
    fetched = [page for page in pages if page.last_fetch_ts is not None]
    changed_total = sum(page.change_count for page in pages)
    passes = store.pass_times()
    print(f"data dir       : {DATA_DIR}")
    print(f"database       : {DB_PATH}")
    print(f"pages tracked  : {len(pages)}")
    print(f"pages fetched  : {len(fetched)}")
    print(f"content changes: {changed_total}")
    print(f"passes run     : {len(passes)}")
    if passes:
        print(f"last pass      : {passes[-1]:.0f} (epoch)")
    store.close()
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    import math

    from recrawl.indexer import TITLE_ZONE_WEIGHT, load_index
    from recrawl.textproc import analyze, tokenize

    if not INDEX_PATH.exists():
        print(f"Index not found at {INDEX_PATH}. Run 'recrawl index' first.")
        return 1

    index = load_index(INDEX_PATH)
    raw_term = args.term.strip()
    analyzed = analyze(tokenize(raw_term))
    if not analyzed:
        print(f"Term '{raw_term}' was filtered out (empty, stopword, or non-alphabetic).")
        return 0

    stemmed = analyzed[0]
    postings = index.postings.get(stemmed, [])
    df = len(postings)
    n_docs = index.doc_count
    idf = math.log(1.0 + n_docs / (df + 0.5)) if df else 0.0

    print(f"Raw term              : {raw_term}")
    print(f"Stemmed form (Porter) : {stemmed}")
    print(f"Corpus docs count (N) : {n_docs}")
    print(f"Document freq (df)    : {df} docs ({df / max(1, n_docs) * 100:.1f}%)")
    print(f"Inverse doc freq (idf): {idf:.4f}")

    if not postings:
        print(f"\nNo postings found for '{stemmed}'.")
        return 0

    print(f"\nPostings sample (showing {min(args.k, len(postings))} of {len(postings)}):")
    print(
        f"{'doc_id':>6s}  {'tf_body':>7s}  {'tf_title':>8s}  {'weight_d':>8s}  {'norm_d':>7s}  url"
    )
    for posting in postings[: args.k]:
        doc = index.docs[posting.doc_id]
        weight_d = 1.0 + math.log(posting.tf_body + TITLE_ZONE_WEIGHT * posting.tf_title)
        norm_d = index.doc_norms[TITLE_ZONE_WEIGHT][posting.doc_id]
        print(
            f"{posting.doc_id:6d}  {posting.tf_body:7.1f}  {posting.tf_title:8.1f}  "
            f"{weight_d:8.4f}  {norm_d:7.3f}  {doc.url[:60]}"
        )
    return 0
