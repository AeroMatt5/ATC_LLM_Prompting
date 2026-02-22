"""
Batch comparison of LLM-generated ATC transcripts (C1-C4) against the
ground-truth transcript (C0).

For each condition the script:
  1. Computes per-turn similarity metrics (TF-IDF cosine, BLEU, ROUGE-L,
     and optionally semantic cosine) for every sample.
  2. Aggregates scores across the 10 samples and reports descriptive
     statistics (mean, std, median, 95 % CI).
  3. Generates individual matplotlib figures for each metric:
       - mean bar chart with 95 % CI error bars
       - box plot
       - per-turn heatmap

Usage (from the project root):
    python -m analysis.batch_analysis
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.similarity import compute_all_metrics, METRIC_NAMES
from analysis.transcript_loader import (
    load_ground_truth,
    load_all_transcripts,
    load_llm_transcript,
    common_turn_ids,
)

TRANSCRIPTS_DIR = _PROJECT_ROOT / "Transcripts"
RESULTS_DIR = _PROJECT_ROOT / "results"
CONDITIONS = ["C1", "C2", "C3", "C4", "C5"]

METRIC_LABELS = {
    "tfidf_cosine": "TF-IDF Cosine",
    "bleu": "BLEU",
    "rouge_l": "ROUGE-L",
    "semantic_cosine": "Semantic Cosine",
}

COLORS = {
    "C1": "#4c72b0",
    "C2": "#dd8452",
    "C3": "#55a868",
    "C4": "#c44e52",
    "C5": "#8172b3",
}


# ------------------------------------------------------------------
# Optional heavy dependency
# ------------------------------------------------------------------

def _load_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading sentence-transformer model (all-MiniLM-L6-v2) ...")
        return SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("[WARN] sentence-transformers not installed; "
              "semantic_cosine metric will be omitted.")
        return None


# ------------------------------------------------------------------
# Core comparison
# ------------------------------------------------------------------

def compare_one_transcript(gt_turns, llm_turns, model=None):
    rows = []
    for tid in common_turn_ids(gt_turns, llm_turns):
        ref = gt_turns[tid]
        if not ref:
            continue
        hyp = llm_turns[tid]
        metrics = compute_all_metrics(ref, hyp, model)
        metrics["turn_id"] = tid
        rows.append(metrics)
    return rows


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def _available(df):
    return [m for m in METRIC_NAMES if m in df.columns]


def _label(m):
    return METRIC_LABELS.get(m, m)


def _ci95(series):
    """95 % confidence interval half-width (t-based)."""
    n = series.count()
    if n < 2:
        return np.nan
    se = series.std() / np.sqrt(n)
    return se * sp_stats.t.ppf(0.975, n - 1)


def summary_table(df):
    """Return a DataFrame with descriptive statistics per condition."""
    metrics = _available(df)
    records = []
    for cond in CONDITIONS:
        sub = df.loc[df["condition"] == cond]
        if sub.empty:
            continue
        for m in metrics:
            s = sub[m]
            records.append({
                "Condition": cond,
                "Metric": _label(m),
                "N": int(s.count()),
                "Mean": s.mean(),
                "Std": s.std(),
                "Median": s.median(),
                "CI_95": _ci95(s),
                "Min": s.min(),
                "Max": s.max(),
            })
    return pd.DataFrame(records)


def print_statistics(df):
    tbl = summary_table(df)
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS  (per condition, across all samples and turns)")
    print("=" * 80)
    print(tbl.to_string(index=False, float_format="%.4f"))
    return tbl


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Mean bar charts (one per metric)
# ------------------------------------------------------------------

def plot_mean_bars(df, out):
    metrics = _available(df)
    for m in metrics:
        grp = df.groupby("condition")[m]
        means = grp.mean().reindex(CONDITIONS)
        cis = grp.apply(_ci95).reindex(CONDITIONS)
        colors = [COLORS[c] for c in means.index]

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(means.index, means.values, yerr=cis.values,
                      capsize=5, color=colors, edgecolor="black",
                      linewidth=0.7, width=0.55)
        ax.set_ylabel("Score")
        ax.set_title(f"Mean {_label(m)}")
        ax.set_ylim(0, min(1.05, means.max() + cis.max() + 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        _savefig(fig, out / f"mean_bars_{m}.png")


# ------------------------------------------------------------------
# Box plots (one per metric)
# ------------------------------------------------------------------

def plot_boxplots(df, out):
    metrics = _available(df)
    for m in metrics:
        data_by_cond = [df.loc[df["condition"] == c, m].values
                        for c in CONDITIONS]
        colors = [COLORS[c] for c in CONDITIONS]

        fig, ax = plt.subplots(figsize=(5, 4))
        bp = ax.boxplot(data_by_cond, tick_labels=CONDITIONS, patch_artist=True,
                        widths=0.5, showfliers=True,
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="gray"))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Score")
        ax.set_title(f"{_label(m)} Distribution")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        _savefig(fig, out / f"boxplot_{m}.png")


# ------------------------------------------------------------------
# Per-turn heatmaps (one per metric, pure matplotlib)
# ------------------------------------------------------------------

def plot_heatmaps(df, out):
    metrics = _available(df)
    for m in metrics:
        present = [c for c in CONDITIONS if c in df["condition"].unique()]
        pivot = df.pivot_table(
            index="turn_id", columns="condition", values=m,
            aggfunc="mean"
        ).reindex(columns=present)
        if pivot.empty:
            continue

        data = pivot.values
        turn_ids = pivot.index.astype(int).tolist()
        conds = pivot.columns.tolist()

        fig, ax = plt.subplots(figsize=(5, max(6, len(turn_ids) * 0.35)))
        im = ax.imshow(data, aspect="auto", cmap="YlGnBu",
                        vmin=0, vmax=1)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds)
        ax.set_yticks(range(len(turn_ids)))
        ax.set_yticklabels(turn_ids, fontsize=7)
        ax.set_ylabel("Turn ID")
        ax.set_title(f"Per-Turn Mean {_label(m)}")
        fig.colorbar(im, ax=ax, label="Score", shrink=0.8)

        for i in range(len(turn_ids)):
            for j in range(len(conds)):
                val = data[i, j]
                if not np.isnan(val):
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        fig.tight_layout()
        _savefig(fig, out / f"heatmap_{m}.png")


# ------------------------------------------------------------------
# Summary table (pure matplotlib)
# ------------------------------------------------------------------

def plot_summary_table(stats_df, out):
    if stats_df.empty:
        return
    display = stats_df.copy()
    for col in ["Mean", "Std", "Median", "CI_95", "Min", "Max"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display["N"] = display["N"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 0.35 * len(display) + 1.2))
    ax.axis("off")
    tbl = ax.table(cellText=display.values, colLabels=display.columns,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    ax.set_title("Summary Statistics", fontsize=12, pad=12)
    fig.tight_layout()
    _savefig(fig, out / "summary_table.png")


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------

def plot_all(df, stats_df, out):
    plot_mean_bars(df, out)
    plot_boxplots(df, out)
    plot_heatmaps(df, out)
    plot_summary_table(stats_df, out)
    print(f"\nFigures saved to {out}/")


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gt_dir = TRANSCRIPTS_DIR / "C0"
    gt_files = sorted(gt_dir.glob("*.json"))
    if not gt_files:
        print(f"[ERROR] No ground-truth file in {gt_dir}")
        sys.exit(1)

    gt_turns = load_ground_truth(gt_files[0])
    non_empty = sum(1 for v in gt_turns.values() if v)
    print(f"Ground truth: {gt_files[0].name}  "
          f"({len(gt_turns)} turns, {non_empty} with ATC text)")

    model = _load_semantic_model()

    all_rows = []
    for cond in CONDITIONS:
        cond_dir = TRANSCRIPTS_DIR / cond
        if not cond_dir.exists():
            print(f"  [WARN] {cond_dir} missing -- skipping {cond}")
            continue
        transcripts = load_all_transcripts(cond_dir)
        if not transcripts:
            print(f"  [WARN] {cond_dir} is empty  -- skipping {cond}")
            continue

        print(f"  {cond}: {len(transcripts)} sample(s)")
        for i, entry in enumerate(transcripts, start=1):
            rows = compare_one_transcript(gt_turns, entry["turns"], model)
            for r in rows:
                r["condition"] = cond
                r["sample_idx"] = i
                r["source_file"] = Path(entry["filepath"]).name
            all_rows.extend(rows)

    if not all_rows:
        print("\n[ERROR] No data produced.  Place transcripts in C1-C4.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "similarity_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nScores written to {csv_path}  "
          f"({len(df)} rows, {df['condition'].nunique()} conditions)")

    stats_df = print_statistics(df)
    stats_df.to_csv(RESULTS_DIR / "summary_statistics.csv", index=False)

    plot_all(df, stats_df, RESULTS_DIR)


if __name__ == "__main__":
    run()
