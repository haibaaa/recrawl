"""Generate publication-quality figures for the IR Hackathon report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
OUTPUT_DIR = PROJECT_ROOT.parent / "report" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.style.use(
    "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)


def plot_freshness_curves() -> None:
    freshness_csv = REPORTS_DIR / "freshness.csv"
    if not freshness_csv.exists():
        print(f"Skipping freshness plot: {freshness_csv} not found")
        return

    data: dict[str, dict[str, list[float]]] = {}
    with freshness_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row["policy"]
            if policy not in data:
                data[policy] = {"budgets": [], "stale_hours": [], "missed": [], "fresh_pct": []}
            data[policy]["budgets"].append(float(row["budget"]))
            data[policy]["stale_hours"].append(float(row["stale_change_hours"]))
            data[policy]["missed"].append(float(row["missed_changes"]))
            data[policy]["fresh_pct"].append(float(row["fresh_fraction"]) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    styles = {
        "proposed": ("#2b5c8f", "o-", "Proposed (λ̂ × staleness × imp)", 2.5),
        "rr": ("#e66101", "s--", "Round-Robin (Oldest first)", 1.8),
        "random": ("#999999", "^:", "Random", 1.5),
        "importance": ("#ca0020", "d-.", "Static Importance", 1.8),
        "oracle": ("#2ca25f", "*--", "Oracle (Upper bound)", 2.0),
    }

    for policy, (color, marker_style, label, lw) in styles.items():
        if policy in data:
            ax1.plot(
                data[policy]["budgets"],
                data[policy]["stale_hours"],
                marker_style,
                color=color,
                label=label,
                linewidth=lw,
                markersize=6,
            )
            ax2.plot(
                data[policy]["budgets"],
                data[policy]["missed"],
                marker_style,
                color=color,
                label=label,
                linewidth=lw,
                markersize=6,
            )

    ax1.set_xlabel("Crawl Budget (Fetches per Pass)")
    ax1.set_ylabel("Total Stale Change Hours (Lower is better)")
    ax1.set_title("Content Staleness vs Fetch Budget")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_xlabel("Crawl Budget (Fetches per Pass)")
    ax2.set_ylabel("Missed Content Changes (Lower is better)")
    ax2.set_title("Missed Changes at Horizon vs Budget")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "freshness_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved freshness curves to {out_path}")


def plot_detector_comparison() -> None:
    detect_csv = REPORTS_DIR / "detect.csv"
    if not detect_csv.exists():
        print(f"Skipping detector plot: {detect_csv} not found")
        return

    methods = []
    p_vals = []
    r_vals = []
    f1_vals = []

    with detect_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] == "live":
                name = (
                    "Shingle Jaccard (Ours)\nk=5, τ=0.85"
                    if row["method"] == "shingle"
                    else "Raw Byte-Hash\n(Baseline)"
                )
                methods.append(name)
                p_vals.append(float(row["precision"]))
                r_vals.append(float(row["recall"]))
                f1_vals.append(float(row["f1"]))

    if not methods:
        return

    import numpy as np

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars1 = ax.bar(x - width, p_vals, width, label="Precision", color="#3182bd")
    bars2 = ax.bar(x, r_vals, width, label="Recall", color="#fd8d3c")
    bars3 = ax.bar(x + width, f1_vals, width, label="F1-Score", color="#31a354")

    ax.set_ylabel("Score [0.0 - 1.0]")
    ax.set_title("Change Detector Performance (n = 29,579 Live Observations)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.15)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right", frameon=True)

    # Annotate values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "detector_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved detector comparison to {out_path}")


def plot_search_ablation() -> None:
    search_csv = REPORTS_DIR / "search_eval_summary.csv"
    if not search_csv.exists():
        print(f"Skipping search ablation plot: {search_csv} not found")
        return

    labels = []
    p1 = []
    p3 = []
    p5 = []

    label_map = {
        "tfidf": "TF-IDF (lnc.ltc)",
        "tfidf+zones": "TF-IDF + Title Zones",
        "+g(d)": "+ g(d) Net Quality",
        "bm25": "Robertson BM25",
    }

    with search_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            labels.append(label_map.get(method, method))
            p1.append(float(row["mean_p@1"]))
            p3.append(float(row["mean_p@3"]))
            p5.append(float(row["mean_p@5"]))

    import numpy as np

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bars1 = ax.bar(x - width, p1, width, label="Mean P@1", color="#4393c3")
    bars2 = ax.bar(x, p3, width, label="Mean P@3", color="#92c5de")
    bars3 = ax.bar(x + width, p5, width, label="Mean P@5", color="#d1e5f0", edgecolor="#2166ac")

    ax.set_ylabel("Mean Precision@k")
    ax.set_title("Search Retrieval Ablation (24 Judged Queries)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(loc="upper right", frameon=True)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "search_ablation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved search ablation to {out_path}")


def main() -> None:
    print(f"Generating evaluation plots from {REPORTS_DIR} to {OUTPUT_DIR}...")
    plot_freshness_curves()
    plot_detector_comparison()
    plot_search_ablation()
    print("Done!")


if __name__ == "__main__":
    main()
