"""
Pairwise comparison of ONE ground-truth transcript against ONE
LLM-generated transcript.

Unlike batch_analysis.py this does NOT aggregate across multiple
samples -- it produces only per-turn scores and individual plots.

Usage (from the project root):
    python -m analysis.pairwise_compare --gt <gt.json> --llm <llm.json>

Optional:
    --output <dir>   Output directory (default: results/pairwise)
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.similarity import compute_all_metrics, METRIC_NAMES
from analysis.transcript_loader import (
    load_ground_truth,
    load_llm_transcript,
    common_turn_ids,
)

DEFAULT_OUTPUT = _PROJECT_ROOT / "results" / "pairwise"

METRIC_LABELS = {
    "tfidf_cosine": "TF-IDF Cosine",
    "bleu": "BLEU",
    "rouge_l": "ROUGE-L",
    "semantic_cosine": "Semantic Cosine",
}


def _load_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformer model (all-MiniLM-L6-v2) ...")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("[WARN] sentence-transformers not installed; "
              "semantic_cosine will be omitted.")
        return None


def _available(df):
    return [m for m in METRIC_NAMES if m in df.columns]


def _label(m):
    return METRIC_LABELS.get(m, m)


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Core comparison
# ------------------------------------------------------------------

def compare_pair(gt_path, llm_path, model=None):
    gt_turns = load_ground_truth(gt_path)
    llm_turns = load_llm_transcript(llm_path)

    rows = []
    for tid in common_turn_ids(gt_turns, llm_turns):
        ref = gt_turns[tid]
        if not ref:
            continue
        hyp = llm_turns[tid]
        metrics = compute_all_metrics(ref, hyp, model)
        metrics["turn_id"] = tid
        metrics["reference"] = ref
        metrics["hypothesis"] = hyp
        rows.append(metrics)

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Plots -- one figure per metric, no aggregation
# ------------------------------------------------------------------

def plot_per_turn_bars(df, out):
    """Individual bar chart for each metric showing score per turn."""
    metrics = _available(df)
    turns = df["turn_id"].values

    for m in metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.45), 4))
        ax.bar(range(len(turns)), df[m].values, color="steelblue",
               edgecolor="black", linewidth=0.5, width=0.7)
        ax.set_xticks(range(len(turns)))
        ax.set_xticklabels(turns.astype(int), fontsize=7)
        ax.set_xlabel("Turn ID")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-Turn {_label(m)}: Ground Truth vs LLM")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        _savefig(fig, out / f"pairwise_bars_{m}.png")


def plot_trend_lines(df, out):
    """Individual line chart for each metric showing trend across turns."""
    metrics = _available(df)

    for m in metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.4), 4))
        ax.plot(df["turn_id"], df[m], marker="o", markersize=4,
                color="steelblue", linewidth=1.2)
        ax.set_xlabel("Turn ID")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{_label(m)} Trend Across Turns")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        _savefig(fig, out / f"pairwise_trend_{m}.png")


def plot_all(df, out):
    plot_per_turn_bars(df, out)
    plot_trend_lines(df, out)
    print(f"Plots saved to {out}/")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pairwise ground-truth vs LLM transcript comparison")
    p.add_argument("--gt", required=True,
                   help="Ground-truth JSON file")
    p.add_argument("--llm", required=True,
                   help="LLM-generated transcript JSON file")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Output directory (default: results/pairwise)")
    return p.parse_args(argv)


def run(gt_path=None, llm_path=None, output_dir=None):
    out = Path(output_dir or DEFAULT_OUTPUT)
    out.mkdir(parents=True, exist_ok=True)

    gt_path, llm_path = Path(gt_path), Path(llm_path)
    for p, label in [(gt_path, "Ground truth"), (llm_path, "LLM transcript")]:
        if not p.exists():
            print(f"[ERROR] {label} not found: {p}")
            sys.exit(1)

    model = _load_semantic_model()
    df = compare_pair(gt_path, llm_path, model)

    if df.empty:
        print("[ERROR] No matching turns with non-empty ground truth.")
        sys.exit(1)

    csv_path = out / "pairwise_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Scores: {csv_path}  ({len(df)} turns)")

    display_cols = ["turn_id"] + _available(df)
    print("\n" + df[display_cols].to_string(index=False, float_format="%.4f"))

    plot_all(df, out)


if __name__ == "__main__":
    args = parse_args()
    run(gt_path=args.gt, llm_path=args.llm, output_dir=args.output)
