#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_NEUTRAL = "#95a5a6"
ERROR_BAR_SCALE = 1.0


# -----------------------------
# Data loading
# -----------------------------
def load_reviews_from_dir(dir_path: Path, condition_label: str) -> pd.DataFrame:
    """
    Load all review JSONs from a directory.

    Assumes:
      - Each file is <paper_id>.json
      - JSON has numeric field: score
    """
    records = []
    for json_file in sorted(dir_path.glob("*.json")):
        paper_id = json_file.stem
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            record = {
                "paper_id": paper_id,
                "condition": condition_label,
                "score": data.get("score"),
            }
            records.append(record)
        except Exception as e:
            print(f"Warning: failed to read {json_file}: {e}")
    return pd.DataFrame(records)


# -----------------------------
# Stats helpers
# -----------------------------
def pooled_standard_deviation(treatment: np.ndarray, control: np.ndarray):
    n_t = len(treatment)
    n_c = len(control)
    if n_t <= 1 or n_c <= 1:
        return None

    var_t = np.var(treatment, ddof=1)
    var_c = np.var(control, ddof=1)
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    return np.sqrt(pooled_var) if pooled_var > 0 else 0.0


def cohen_d_standard_error(cohen_d: float, n_t: int, n_c: int):
    if n_t <= 1 or n_c <= 1:
        return None
    return np.sqrt((n_t + n_c) / (n_t * n_c) + (cohen_d**2) / (2 * (n_t + n_c - 2)))


def evaluate_expectation(cohen_d: float, expected: str, tolerance: float = 0.1):
    """
    expected in {"positive", "negative", "zero"}.
    """
    if cohen_d is None or np.isnan(cohen_d) or expected is None:
        return None
    if expected == "positive":
        return cohen_d > tolerance
    if expected == "negative":
        return cohen_d < -tolerance
    if expected == "zero":
        return abs(cohen_d) <= tolerance
    return None


def cohen_d_to_overlap_percent(cohen_d: float):
    """
    Convert Cohen's d to an approximate percentage overlap
    between the two underlying normal distributions.

    Formula (from standard approximations):
        overlap = 1 - (2 * |d|) / sqrt(4 + d^2)

    We then return overlap * 100 as a percentage.
    """
    if cohen_d is None or not np.isfinite(cohen_d):
        return None
    overlap = 1.0 - (2.0 * abs(cohen_d)) / np.sqrt(4.0 + cohen_d**2)
    overlap = max(0.0, min(1.0, overlap))  # clamp just in case of tiny numeric noise
    return float(overlap * 100.0)


# -----------------------------
# Effect size computation
# -----------------------------
def compute_effects(
    df: pd.DataFrame,
    comparisons: List[Dict],
    metrics: List[str],
) -> Dict[str, List[Dict]]:
    """
    df columns:
      paper_id, condition, score

    comparisons: list of dicts with keys:
      - label
      - treatment_condition
      - control_condition
      - expected_direction ("positive", "negative", "zero", or None)
    """
    results = {m: [] for m in metrics}

    for comp in comparisons:
        t_label = comp["treatment_condition"]
        c_label = comp["control_condition"]
        expected = comp.get("expected_direction")

        for metric in metrics:
            t_df = df[df["condition"] == t_label][["paper_id", metric]]
            t_df = t_df.rename(columns={metric: "treat"})
            c_df = df[df["condition"] == c_label][["paper_id", metric]]
            c_df = c_df.rename(columns={metric: "control"})

            merged = t_df.merge(c_df, on="paper_id", how="inner").dropna()
            if merged.empty:
                continue

            t_scores = merged["treat"].astype(float).values
            c_scores = merged["control"].astype(float).values

            mean_diff = float(np.mean(t_scores - c_scores))
            pooled_sd = pooled_standard_deviation(t_scores, c_scores)

            if pooled_sd is None:
                cohen_d = np.nan
            elif pooled_sd == 0:
                cohen_d = float(np.sign(mean_diff)) * np.inf if mean_diff != 0 else 0.0
            else:
                cohen_d = float(mean_diff / pooled_sd)

            if np.isfinite(cohen_d):
                se = cohen_d_standard_error(cohen_d, len(t_scores), len(c_scores))
            else:
                se = None

            if se is not None:
                ci_low = float(cohen_d - 1.96 * se)
                ci_high = float(cohen_d + 1.96 * se)
            else:
                ci_low = None
                ci_high = None

            matches = (
                evaluate_expectation(cohen_d, expected) if np.isfinite(cohen_d) else None
            )

            # Add percent overlap for interpretability
            overlap_percent = cohen_d_to_overlap_percent(cohen_d)

            results[metric].append(
                {
                    "comparison": comp["label"],
                    "treatment_label": t_label,
                    "control_label": c_label,
                    "expected_direction": expected,
                    "matches_expectation": matches,
                    "n_pairs": int(len(merged)),
                    "treatment_mean": float(np.mean(t_scores)),
                    "control_mean": float(np.mean(c_scores)),
                    "mean_difference": mean_diff,
                    "cohen_d": float(cohen_d) if np.isfinite(cohen_d) else None,
                    "standard_error": float(se) if se is not None else None,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "overlap_percent": overlap_percent,
                }
            )

    return results


# -----------------------------
# Plotting helpers
# -----------------------------
def _plot_metric_on_axis(ax, effects_for_metric: List[Dict], metric: str):
    """
    Draw bar chart with Cohen's d for a single metric on a given axis.
    Error bars correspond to ~95% CI around d (d ± 1.96 * SE or CI bounds).
    """
    cohen_ds = []
    yerr = [[], []]
    colors = []
    labels = []
    plotted_effects = []

    for effect in effects_for_metric:
        cohen_d = effect["cohen_d"]
        if cohen_d is None:
            continue

        labels.append(effect["comparison"])
        cohen_ds.append(cohen_d)
        plotted_effects.append(effect)

        if effect["ci_low"] is not None and effect["ci_high"] is not None:
            # Distance from the point estimate (d) to the CI bounds
            lower = cohen_d - effect["ci_low"]
            upper = effect["ci_high"] - cohen_d
        elif effect["standard_error"] is not None:
            # Approximate 95% CI using ± 1.96 * SE
            lower = upper = 1.96 * effect["standard_error"]
        else:
            lower = upper = 0.0

        lower *= ERROR_BAR_SCALE
        upper *= ERROR_BAR_SCALE

        yerr[0].append(lower)
        yerr[1].append(upper)

        matches = effect["matches_expectation"]
        if matches is None:
            colors.append(COLOR_NEUTRAL)
        elif matches:
            colors.append(COLOR_MATCH)
        else:
            colors.append(COLOR_MISMATCH)

    if not cohen_ds:
        ax.text(0.5, 0.5, "No finite Cohen's d", ha="center", va="center")
        return

    x_positions = np.arange(len(cohen_ds))
    bars = ax.bar(
        x_positions,
        cohen_ds,
        yerr=yerr,
        capsize=8,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )

    for bar, effect in zip(bars, plotted_effects):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"d={effect['cohen_d']:.2f}",
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=11,
            fontweight="bold",
            color="black",
        )
        expectation = effect["expected_direction"]
        if expectation:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -0.1,
                f"Expected: {expectation}",
                ha="center",
                va="top",
                fontsize=10,
                rotation=45,
                color="dimgray",
                transform=ax.get_xaxis_transform(),
            )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Score", fontsize=14, fontweight="bold")


def create_effect_size_plots(
    effect_results: Dict[str, List[Dict]],
    output_dir: Path,
    pattern_name: str = "Abstract Manipulations",
    output_prefix: str = "abstract",
):
    """
    Create a single-panel figure:
      - Cohen's d for score (bars)
    """
    score_effects = effect_results.get("score", [])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"{pattern_name}", fontsize=16, fontweight="bold")

    _plot_metric_on_axis(ax, score_effects, "score")

    # Legend for score expectation indicators
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MATCH, ec="black", label="Matches expectation"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MISMATCH, ec="black", label="Contradicts expectation"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, ec="black", label="No expectation"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11)

    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plot_path = output_dir / f"cohens_d_{output_prefix}.png"
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {plot_path}")


# -----------------------------
# Empirical reference scale
# -----------------------------
def compute_empirical_reference_scale(effect_results: Dict[str, List[Dict]], metric: str, output_dir: Path):
    """
    Build an empirical reference scale for Cohen's d based on the
    current experiment: min, median, max |d|, plus a small summary JSON.

    This lets you interpret future experiments relative to:
      - "typical small change" vs
      - "extreme manipulation" in your setup.
    """
    metric_effects = effect_results.get(metric, [])
    abs_ds = [abs(e["cohen_d"]) for e in metric_effects if e.get("cohen_d") is not None]

    if not abs_ds:
        print("No finite Cohen's d for reference scale.")
        return

    abs_ds = np.array(abs_ds)
    ref = {
        "metric": metric,
        "min_abs_d": float(np.min(abs_ds)),
        "median_abs_d": float(np.median(abs_ds)),
        "max_abs_d": float(np.max(abs_ds)),
    }

    # Print to console for quick inspection
    print("\nEMPIRICAL REFERENCE SCALE (Cohen's d, abs values)")
    print(f"  min |d|    = {ref['min_abs_d']:.3f}")
    print(f"  median |d| = {ref['median_abs_d']:.3f}")
    print(f"  max |d|    = {ref['max_abs_d']:.3f}")
    print("You can treat 'max |d|' as your extreme manipulation anchor.")

    # Save to JSON for later reuse
    ref_path = output_dir / f"reference_scale_{metric}.json"
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)
    print(f"Saved reference scale to: {ref_path}")


# -----------------------------
# Score distribution plots
# -----------------------------
def plot_score_distributions(df: pd.DataFrame, output_dir: Path):
    """
    Plot grouped bar chart of score distributions (1–9) by condition
    (e.g., control / abstract_sham / abstract_flaw).

    - x-axis: score (1 to 9)
    - y-axis: percentage of reviews with that score
    - bars: grouped by score, one bar per condition
    """
    # Only keep valid scores
    df = df.dropna(subset=["score"]).copy()
    df["score"] = df["score"].astype(int)

    # Score range 1–9
    scores = list(range(1, 10))

    conditions = sorted(df["condition"].unique())

    # Count per (condition, score)
    counts = {
        cond: [int((df[(df["condition"] == cond) & (df["score"] == s)].shape[0])) for s in scores]
        for cond in conditions
    }

    # Convert to percentages within each condition
    percentages = {}
    for cond in conditions:
        total = sum(counts[cond])
        if total == 0:
            percentages[cond] = [0.0] * len(scores)
        else:
            percentages[cond] = [100.0 * c / total for c in counts[cond]]

    x = np.arange(len(scores))  # positions for scores
    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)  # total bar width ≈ 0.8 split across conditions

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, cond in enumerate(conditions):
        offsets = x + (i - (n_cond - 1) / 2) * width
        ax.bar(
            offsets,
            percentages[cond],
            width=width,
            label=cond,
            edgecolor="black",
            alpha=0.85,
        )

    ax.set_title("Score distribution by condition (1–9 scale)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Percentage of reviews (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(scores)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Condition")

    out_path = output_dir / "score_distribution_grouped.png"
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved grouped score distribution plot: {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate abstract manipulations (control vs flaw vs sham) on new review format."
    )
    parser.add_argument("--control_dir", type=str, required=True)
    parser.add_argument("--flaw_dir", type=str, required=True)
    parser.add_argument("--sham_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Plot customisation
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Abstract Manipulations",
        help="Title to use for the figure.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="abstract",
        help="Prefix for the output plot filename (cohens_d_<prefix>.png).",
    )

    args = parser.parse_args()

    control_dir = Path(args.control_dir)
    flaw_dir = Path(args.flaw_dir)
    sham_dir = Path(args.sham_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading reviews...")
    df_control = load_reviews_from_dir(control_dir, "control")
    df_flaw = load_reviews_from_dir(flaw_dir, "abstract_flaw")
    df_sham = load_reviews_from_dir(sham_dir, "abstract_sham")

    df = pd.concat([df_control, df_flaw, df_sham], ignore_index=True)
    print(f"Total rows: {len(df)} (papers: {df['paper_id'].nunique()})")

    # Only score now
    metrics = ["score"]

    comparisons = [
        {
            "label": "Abstract sham vs Control",
            "treatment_condition": "abstract_sham",
            "control_condition": "control",
            "expected_direction": "negative",
        },
        {
            "label": "Abstract flaw vs Control",
            "treatment_condition": "abstract_flaw",
            "control_condition": "control",
            "expected_direction": "negative",
        },
        {
            "label": "Abstract flaw vs Sham",
            "treatment_condition": "abstract_flaw",
            "control_condition": "abstract_sham",
            "expected_direction": "negative",
        },
    ]

    print("Computing effect sizes...")
    effects = compute_effects(df, comparisons, metrics)

    # Print quick summary
    print("\nEFFECT SIZE SUMMARY")
    for metric in metrics:
        metric_effects = effects.get(metric, [])
        print(f"\n{metric.upper()}:")
        if not metric_effects:
            print("  No data.")
            continue
        for e in metric_effects:
            d_val = e["cohen_d"]
            d_str = "nan" if d_val is None else f"{d_val:.3f}"
            overlap_str = (
                "nan"
                if e.get("overlap_percent") is None
                else f"{e['overlap_percent']:.1f}%"
            )
            match_icon = (
                "✅"
                if e["matches_expectation"] is True
                else ("❌" if e["matches_expectation"] is False else "⚪️")
            )
            exp = e["expected_direction"] or "none"
            print(
                f"  {match_icon} {e['comparison']} (expected {exp}) "
                f"| n={e['n_pairs']} | d={d_str} "
                f"(Δ={e['mean_difference']:.3f}) | overlap≈{overlap_str}"
            )

    # Save stats + data
    summary_rows = []
    for metric, metric_effects in effects.items():
        for e in metric_effects:
            row = {"metric": metric}
            row.update(e)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary stats to: {summary_csv}")

    # Empirical reference scale from THIS experiment
    compute_empirical_reference_scale(effects, metric="score", output_dir=output_dir)

    print("\nGenerating plots...")
    create_effect_size_plots(
        effects,
        output_dir,
        pattern_name=args.plot_title,
        output_prefix=args.output_prefix,
    )

    plot_score_distributions(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
