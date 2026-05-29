"""
Generate thesis plots from existing results.
Saves to results/plots/ at 300 DPI in both PDF and PNG.

Plots generated:
  1. bias_by_model.pdf/png   — cohort bias (%) grouped by model, coloured by dataset
  2. cdnow_cumulative.pdf/png — CDNOW holdout cumulative transactions: LSTM vs Pareto/NBD vs actual
  3. seed_variance.pdf/png   — CDNOW bias across 3 seeds (violin / strip plot per model)

Run:
  python generate_thesis_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_CSV = Path("results/kaggle_notebook_results_v7/tables/comparison_all.csv")
CDNOW_LSTM_ARRAYS = Path("results/kaggle_notebook_results_v7/tables/lstm_base_cdnow_v6_seed42_sample_arrays.npz")
CDNOW_PARETO_ARRAYS = Path("results/tables/pareto_nbd_cdnow_arrays.npz")

# Colour palette: one colour per dataset, consistent across all plots
DATASET_COLORS = {
    "cdnow":     "#4C72B0",
    "dunnhumby": "#DD8452",
    "tafeng":    "#55A868",
    "uci":       "#C44E52",
}

# Human-readable model labels
MODEL_LABELS = {
    "pareto_nbd":         "Pareto/NBD",
    "bgnbd_gg":           "BG/NBD+GG",
    "pareto_ggg":         "Pareto/GGG",
    "gamma_poisson":      "Gamma-Poisson",
    "lstm_base":          "Base LSTM",
    "lstm_joint":         "Joint LSTM",
    "transformer_joint":  "Joint Transformer",
    "extension3_lstm":    "Ext-3 LSTM",
    "extension3_transformer": "Ext-3 Transformer",
}

MODEL_ORDER = [
    "pareto_nbd", "bgnbd_gg", "pareto_ggg",
    "lstm_base", "lstm_joint", "transformer_joint",
]

DATASET_ORDER = ["cdnow", "tafeng", "uci", "dunnhumby"]

sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.0)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "figure.dpi": 150,
})


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Per-model cohort bias, grouped by dataset
# ──────────────────────────────────────────────────────────────────────────────

def plot_bias_by_model() -> None:
    df = pd.read_csv(COMPARISON_CSV)

    # Use per-seed rows but aggregate to (model, version, dataset) mean/std
    agg = (
        df[df["model"].isin(MODEL_ORDER + list(MODEL_LABELS.keys()))]
        .groupby(["model", "dataset"])["bias_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg = agg[agg["model"].isin(MODEL_ORDER)]
    agg = agg[agg["dataset"].isin(DATASET_ORDER)]

    n_models = len(MODEL_ORDER)
    n_datasets = len(DATASET_ORDER)
    bar_width = 0.7 / n_datasets
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, ds in enumerate(DATASET_ORDER):
        sub = agg[agg["dataset"] == ds]
        vals, errs = [], []
        for m in MODEL_ORDER:
            row = sub[sub["model"] == m]
            if row.empty:
                vals.append(np.nan)
                errs.append(0.0)
            else:
                vals.append(float(row["mean"].values[0]))
                # sem = std/sqrt(n); show 1 SE error bar
                n = float(row["count"].values[0])
                std = float(row["std"].values[0]) if not np.isnan(row["std"].values[0]) else 0
                errs.append(std / np.sqrt(max(n, 1)))

        offset = i * bar_width - (n_datasets - 1) * bar_width / 2
        bars = ax.bar(
            x + offset, vals, bar_width,
            color=DATASET_COLORS.get(ds, "#888888"),
            yerr=errs, capsize=3, error_kw={"elinewidth": 0.8},
            label=ds.capitalize() if ds != "uci" else "UCI",
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in MODEL_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Cohort bias (%)")
    ax.set_title("Holdout cohort purchase bias by model and dataset\n(mean ± 1 SE across 3 seeds; benchmarks: 1 run)")
    ax.legend(title="Dataset", loc="upper right", framealpha=0.7, fontsize=9)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)

    fig.tight_layout()
    _save(fig, "bias_by_model")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: CDNOW holdout cumulative transactions — LSTM vs Pareto/NBD vs actual
# ──────────────────────────────────────────────────────────────────────────────

def plot_cdnow_cumulative() -> None:
    if not CDNOW_LSTM_ARRAYS.exists():
        print(f"  SKIP: {CDNOW_LSTM_ARRAYS} not found.")
        return
    if not CDNOW_PARETO_ARRAYS.exists():
        print(f"  SKIP: {CDNOW_PARETO_ARRAYS} not found.")
        return

    lstm_data = np.load(CDNOW_LSTM_ARRAYS)
    pareto_data = np.load(CDNOW_PARETO_ARRAYS)

    # LSTM: (N_customers, 26) → sum over customers → (26,) per-week cohort count
    actual_weekly = lstm_data["per_week_true_freq"].sum(axis=0).astype(float)
    lstm_weekly   = lstm_data["per_week_pred_freq"].sum(axis=0).astype(float)

    # Cumulative
    actual_cum = np.cumsum(actual_weekly)
    lstm_cum   = np.cumsum(lstm_weekly)

    # Pareto/NBD: per-customer totals only → single endpoint
    pareto_total = pareto_data["per_customer_pred_freq"].sum()
    actual_total_pareto = pareto_data["per_customer_true_freq"].sum()

    weeks = np.arange(1, len(actual_cum) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(weeks, actual_cum, color="black", linewidth=1.8, label="Actual", zorder=4)
    ax.plot(weeks, lstm_cum,   color=DATASET_COLORS["cdnow"], linewidth=1.6,
            linestyle="-", label="Base LSTM (seed 42)", zorder=3)

    # Pareto/NBD as a 26-week total point with a dashed line from origin
    pareto_cum_approx = np.linspace(0, pareto_total, len(weeks))
    ax.plot(weeks, pareto_cum_approx, color="#888888", linewidth=1.2,
            linestyle="--", label="Pareto/NBD (total, linear approx.)", zorder=2)

    ax.set_xlabel("Holdout week")
    ax.set_ylabel("Cumulative transactions (cohort)")
    ax.set_title("CDNOW holdout cumulative transactions — 26-week forecast\n(Base LSTM v6 seed 42 vs Pareto/NBD vs actual)")
    ax.legend(loc="upper left", framealpha=0.7, fontsize=9)
    ax.yaxis.grid(True, alpha=0.35, zorder=0)

    # Annotate final bias
    bias_lstm = (lstm_cum[-1] - actual_cum[-1]) / actual_cum[-1] * 100
    bias_pareto = (pareto_total - actual_total_pareto) / actual_total_pareto * 100
    ax.annotate(f"LSTM bias: {bias_lstm:+.1f}%", xy=(weeks[-1], lstm_cum[-1]),
                xytext=(-60, 6), textcoords="offset points",
                fontsize=8.5, color=DATASET_COLORS["cdnow"],
                arrowprops=dict(arrowstyle="-", color=DATASET_COLORS["cdnow"], lw=0.8))
    ax.annotate(f"Pareto/NBD bias: {bias_pareto:+.1f}%", xy=(weeks[-1], pareto_total),
                xytext=(-90, -14), textcoords="offset points",
                fontsize=8.5, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))

    fig.tight_layout()
    _save(fig, "cdnow_cumulative")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Seed variance — CDNOW bias across 3 seeds (strip + box per model)
# ──────────────────────────────────────────────────────────────────────────────

def plot_seed_variance() -> None:
    df = pd.read_csv(COMPARISON_CSV)

    # Only CDNOW rows with multiple seeds
    cdnow = df[(df["dataset"] == "cdnow") & (df["model"].isin(MODEL_ORDER))].copy()

    # Keep Tier A + B only (tier C diverged runs distort the y-axis)
    cdnow = cdnow[cdnow["quality_tier"].isin(["A", "B"])].copy()

    if cdnow.empty:
        print("  SKIP: no CDNOW rows for seed variance plot.")
        return

    # Order models as they appear in canonical order, keeping only those present
    present_models = [m for m in MODEL_ORDER if m in cdnow["model"].values]
    cdnow["model_label"] = cdnow["model"].map(lambda m: MODEL_LABELS.get(m, m))
    model_label_order = [MODEL_LABELS.get(m, m) for m in present_models]

    fig, ax = plt.subplots(figsize=(7, 4))

    sns.stripplot(
        data=cdnow, x="model_label", y="bias_pct",
        order=model_label_order,
        jitter=0.08, size=8, ax=ax, zorder=3,
        hue="quality_tier", hue_order=["A", "B"],
        palette={"A": "#4C72B0", "B": "#DD8452"},
    )
    sns.boxplot(
        data=cdnow, x="model_label", y="bias_pct",
        order=model_label_order,
        width=0.35, color="white", linewidth=1.2,
        fliersize=0, ax=ax, zorder=2,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(15, color="#aaa", linewidth=0.6, linestyle=":", zorder=1)
    ax.axhline(-15, color="#aaa", linewidth=0.6, linestyle=":", zorder=1)
    ax.text(ax.get_xlim()[1], 15.5, "Tier A/B threshold", fontsize=7.5, color="#888", ha="right")
    ax.text(ax.get_xlim()[1], -14.5, "Tier A/B threshold", fontsize=7.5, color="#888", ha="right")

    ax.set_xlabel("")
    ax.set_ylabel("Cohort bias (%)")
    ax.set_title("CDNOW holdout bias across 3 seeds — model stability\n(dots = individual seeds; box = IQR)")
    ax.yaxis.grid(True, alpha=0.35, zorder=0)
    ax.tick_params(axis="x", rotation=15)

    # Replace default hue legend
    handles = [
        mpatches.Patch(color="#4C72B0", label="Tier A (publishable)"),
        mpatches.Patch(color="#DD8452", label="Tier B (marginal)"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper right", framealpha=0.7)

    fig.tight_layout()
    _save(fig, "seed_variance_cdnow")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating thesis plots → results/plots/")
    print("\n[1/3] Bias by model bar chart...")
    plot_bias_by_model()

    print("\n[2/3] CDNOW cumulative transactions curve...")
    plot_cdnow_cumulative()

    print("\n[3/3] Seed variance (CDNOW)...")
    plot_seed_variance()

    print("\nDone. Files in results/plots/:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
