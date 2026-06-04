"""
figures_ch3_data.py
-------------------
Generate all tables and figures for Section 3.2 (Datasets and Pipeline)
of the methodology chapter.

Outputs are saved to:
    thesis-full/chapters/3_methodology/figures/

Artefacts produced:
    table_3_1_dataset_summary.tex   – Table 3.1: cross-dataset summary statistics
    figure_3_1_raster.pdf           – Figure 3.1: customer activity raster (2x2 grid)
    figure_3_2_freq_spend.pdf       – Figure 3.2: frequency + spend distributions (4x2)
    figure_3_3_timelines.pdf        – Figure 3.3: weekly aggregate activity timelines
    figure_3_4_dunnhumby_cov.pdf    – Figure 3.4: Dunnhumby covariate profile

Run from the thesis-code/ directory:
    python scripts/figures_ch3_data.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
FIGURES_DIR  = REPO_ROOT.parent / "thesis-full" / "chapters" / "3_methodology" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR      = REPO_ROOT / "data" / "raw"
DUNNH_DIR    = RAW_DIR / "Dunnhumby datasets"

RNG = np.random.default_rng(42)

# ── visual style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "serif",
    "font.size":           8,
    "axes.labelsize":      8,
    "axes.titlesize":      9,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           False,
})

C_CALIB     = "#2d2d2d"   # charcoal  – calibration marks / bars
C_HOLDOUT   = "#2a9d8f"   # teal      – holdout marks
C_SPLIT     = "#e76f51"   # orange    – split line
C_FREQ      = "#264653"   # dark teal – frequency histograms
C_SPEND_RAW = "#adb5bd"   # grey      – raw spend
C_SPEND_LOG = "#e9c46a"   # gold      – log1p spend
C_ACTIVE    = "#264653"   # dark teal – active customers
C_REVENUE   = "#2a9d8f"   # teal      – revenue area

# ── dataset config ────────────────────────────────────────────────────────────
KEYS    = ["cdnow", "uci", "tafeng", "dunnhumby"]
LABELS  = {
    "cdnow":     "CDNOW",
    "uci":       "UCI Online Retail II",
    "tafeng":    "Ta-Feng",
    "dunnhumby": "Dunnhumby",
}
DOMAINS = {
    "cdnow":     "Music retail, USA",
    "uci":       "E-commerce, UK",
    "tafeng":    "Grocery retail, Taiwan",
    "dunnhumby": "Grocery + coupon, USA",
}
CURRENCIES = {
    "cdnow":     "USD",
    "uci":       "GBP",
    "tafeng":    "TWD",
    "dunnhumby": "USD",
}
CALIB_WEEKS   = {"cdnow": 39, "uci": 78, "tafeng": 12, "dunnhumby": 80}
HOLDOUT_WEEKS = {"cdnow": 39, "uci": 26, "tafeng":  5, "dunnhumby": 22}

INCOME_ORDER = [
    "Under 15K", "15-24K", "25-34K", "35-49K",
    "50-74K", "75-99K", "100-124K", "125-149K",
    "150-174K", "175-199K", "200-249K", "250K+",
]
HSIZE_ORDER = ["1", "2", "3", "4", "5+"]

# ── data loaders ──────────────────────────────────────────────────────────────

def load_cdnow() -> pd.DataFrame:
    """Clean transaction-level DataFrame with columns [customer_id, date, transaction_amount]."""
    path = RAW_DIR / "CDNOW_master.txt"
    if not path.exists():
        path = RAW_DIR / "CDNOW_sample.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] == 5:
        df.columns = ["master_id", "customer_id", "date", "num_cds", "amount"]
    else:
        df.columns = ["customer_id", "date", "num_cds", "amount"]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date", "customer_id", "amount"])
    df = df[df["amount"] > 0]
    df["customer_id"] = df["customer_id"].astype(int)
    return df[["customer_id", "date", "amount"]].rename(columns={"amount": "transaction_amount"})


def load_uci() -> pd.DataFrame:
    """Clean invoice-level DataFrame with columns [customer_id, date, transaction_amount]."""
    excel = RAW_DIR / "online_retail_II.xlsx"
    csv   = RAW_DIR / "Online Retail.csv"
    if excel.exists():
        print("  Loading UCI Excel (~30 s)…")
        sheets = pd.read_excel(excel, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    elif csv.exists():
        raw = pd.read_csv(csv, sep=None, engine="python", encoding="utf-8-sig")
    else:
        raise FileNotFoundError(f"No UCI data found in {RAW_DIR}")

    col_map = {}
    for col in raw.columns:
        k = str(col).strip().lower().replace(" ", "").replace("_", "")
        if k in {"invoice", "invoiceno"}:      col_map[col] = "invoice"
        elif k == "invoicedate":               col_map[col] = "invoice_date"
        elif k in {"customerid", "customer"}:  col_map[col] = "cid_raw"
        elif k in {"price", "unitprice"}:      col_map[col] = "price"
        elif k == "quantity":                  col_map[col] = "quantity"
    raw = raw.rename(columns=col_map)

    raw = raw.dropna(subset=["cid_raw"])
    raw = raw[~raw["invoice"].astype(str).str.startswith("C")]
    raw["price"]    = pd.to_numeric(raw["price"].astype(str).str.replace(",", "."), errors="coerce")
    raw["quantity"] = pd.to_numeric(raw["quantity"].astype(str).str.replace(",", "."), errors="coerce")
    raw = raw[(raw["quantity"] > 0) & (raw["price"] > 0)]
    raw["transaction_amount"] = raw["quantity"] * raw["price"]
    raw["customer_id"] = pd.to_numeric(raw["cid_raw"], errors="coerce")
    raw = raw.dropna(subset=["customer_id"])
    raw["customer_id"] = raw["customer_id"].astype(int)
    raw["date"] = pd.to_datetime(raw["invoice_date"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["date"])

    return (
        raw.groupby(["customer_id", "invoice", "date"])
        .agg(transaction_amount=("transaction_amount", "sum"))
        .reset_index()[["customer_id", "date", "transaction_amount"]]
    )


def load_tafeng() -> pd.DataFrame:
    """Clean trip-level DataFrame with columns [customer_id, date, transaction_amount]."""
    path = RAW_DIR / "ta_feng_all_months_merged.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿").upper() for c in df.columns]
    df = df.dropna(subset=["CUSTOMER_ID", "SALES_PRICE"])
    df["SALES_PRICE"] = pd.to_numeric(df["SALES_PRICE"], errors="coerce")
    df = df[df["SALES_PRICE"] > 0]
    df["customer_id"] = df["CUSTOMER_ID"].astype(int)
    df["date"] = pd.to_datetime(df["TRANSACTION_DT"], errors="coerce")
    df = df.dropna(subset=["date"])
    return (
        df.groupby(["customer_id", "date"])
        .agg(transaction_amount=("SALES_PRICE", "sum"))
        .reset_index()[["customer_id", "date", "transaction_amount"]]
    )


def load_dunnhumby() -> pd.DataFrame:
    """Clean basket-level DataFrame with columns [customer_id, date, week, transaction_amount]."""
    path = DUNNH_DIR / "transaction_data.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["household_key", "SALES_VALUE"])
    df = df[df["SALES_VALUE"] > 0]
    df["customer_id"] = df["household_key"].astype(int)
    df["transaction_amount"] = df["SALES_VALUE"].astype(float)
    # Pseudo-date from DAY offset (actual calendar dates are proprietary)
    df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["DAY"].astype(int), unit="D")
    df["week"] = (df["DAY"].astype(int) - 1) // 7
    return (
        df.groupby(["customer_id", "BASKET_ID", "date", "week"])
        .agg(transaction_amount=("transaction_amount", "sum"))
        .reset_index()[["customer_id", "date", "week", "transaction_amount"]]
    )

# ── weekly aggregation ────────────────────────────────────────────────────────

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a weekly-aggregate DataFrame: [customer_id, week, weekly_freq, weekly_spend].
    Uses elapsed-week index from the earliest transaction date.
    If df already has a 'week' column (Dunnhumby), reuse it.
    """
    df = df.copy()
    if "week" not in df.columns:
        min_date = df["date"].min()
        df["week"] = ((df["date"] - min_date).dt.days // 7).astype(int)
    return (
        df.groupby(["customer_id", "week"])
        .agg(weekly_freq=("transaction_amount", "count"),
             weekly_spend=("transaction_amount", "sum"))
        .reset_index()
    )

# ── summary statistics ────────────────────────────────────────────────────────

def compute_summary(key: str, raw_df: pd.DataFrame, calib_w: pd.DataFrame) -> dict:
    """Compute per-dataset statistics for Table 3.1."""
    n_cust  = raw_df["customer_id"].nunique()
    n_trans = len(raw_df)
    period  = (f"{raw_df['date'].min().strftime('%b %Y')}"
               f" – {raw_df['date'].max().strftime('%b %Y')}")

    calib_per_cust = calib_w.groupby("customer_id")["weekly_freq"].sum()
    mean_purch     = calib_per_cust.mean()
    pct_one_time   = (calib_per_cust == 1).mean() * 100

    med_spend = raw_df["transaction_amount"].median()

    nonzero_weekly = calib_w.loc[calib_w["weekly_freq"] > 0, "weekly_freq"]
    if key == "cdnow":
        cap_val = int(calib_w["weekly_freq"].max())
        cap_str = f"{cap_val} (observed)"
    else:
        cap_val = int(np.ceil(np.quantile(nonzero_weekly, 0.99))) if len(nonzero_weekly) > 0 else "?"
        cap_str = f"{cap_val} (99th pctl)"

    return {
        "key":            key,
        "Dataset":        LABELS[key],
        "Domain":         DOMAINS[key],
        "N":              n_cust,
        "Transactions":   n_trans,
        "Period":         period,
        "Tc":             CALIB_WEEKS[key],
        "Th":             HOLDOUT_WEEKS[key],
        "MeanCalib":      mean_purch,
        "PctOneTime":     pct_one_time,
        "MedianSpend":    med_spend,
        "Currency":       CURRENCIES[key],
        "FreqCap":        cap_str,
    }

# ── Table 3.1 ─────────────────────────────────────────────────────────────────

def generate_table_3_1(stats: list[dict]) -> None:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Cross-dataset summary statistics. \textit{N} = number of")
    lines.append(r"    customers after pipeline filtering; $T_c$ and $T_h$ = calibration")
    lines.append(r"    and holdout window lengths; mean calibration purchases = average")
    lines.append(r"    transaction count per customer in calibration; median spend = median")
    lines.append(r"    per-basket value in local currency. Frequency cap: CDNOW uses the")
    lines.append(r"    observed maximum following \citet{valendin_et_al_2022}; all other")
    lines.append(r"    datasets use the 99th percentile of non-zero weekly counts.}")
    lines.append(r"  \label{tab:data:summary}")
    lines.append(r"  \begin{tabular}{lllrrrrrrl}")
    lines.append(r"    \toprule")
    lines.append(
        r"    Dataset & Domain & Period & $N$ & Trans. & "
        r"$T_c$ & $T_h$ & \makecell{Mean\\calib.} & "
        r"\makecell{\% one-\\time} & \makecell{Median\\spend} \\"
    )
    lines.append(r"    \midrule")
    for s in stats:
        med = f"{s['Currency']}\\,{s['MedianSpend']:.2f}"
        lines.append(
            f"    {s['Dataset']} & {s['Domain']} & {s['Period']} & "
            f"{s['N']:,} & {s['Transactions']:,} & "
            f"{s['Tc']} & {s['Th']} & "
            f"{s['MeanCalib']:.1f} & "
            f"{s['PctOneTime']:.0f}\\% & "
            f"{med} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    out_path = FIGURES_DIR / "table_3_1_dataset_summary.tex"
    out_path.write_text("\n".join(lines))
    print(f"  Saved {out_path.name}")

# ── Figure 3.1: customer activity raster ─────────────────────────────────────

def build_raster(weekly: pd.DataFrame, calib_weeks: int, holdout_weeks: int,
                 n_show: int = 30) -> tuple[np.ndarray, list[int]]:
    """
    Build a (n_show, T_total) integer matrix for the activity raster.
    Values: 0 = no purchase, 1 = calibration purchase, 2 = holdout purchase.
    Customers sorted by total calibration purchases (most active at row 0).
    """
    T = calib_weeks + holdout_weeks
    calib_totals = (
        weekly[weekly["week"] < calib_weeks]
        .groupby("customer_id")["weekly_freq"].sum()
    )
    # Sample 30 from customers with ≥1 calibration purchase, sorted desc by total
    eligible = calib_totals[calib_totals > 0].sort_values(ascending=False)
    n_sample = min(n_show, len(eligible))
    # stratified sample: take top 10, middle 10, bottom 10 to show heterogeneity
    if n_sample >= 30:
        top_n    = min(10, n_sample)
        bot_n    = min(10, n_sample - top_n)
        mid_pool = eligible.iloc[top_n: n_sample - bot_n]
        mid_n    = min(10, len(mid_pool))
        mid_idx  = RNG.choice(len(mid_pool), size=mid_n, replace=False) if mid_n > 0 else []
        chosen = pd.concat([
            eligible.iloc[:top_n],
            mid_pool.iloc[mid_idx] if len(mid_idx) > 0 else pd.Series(dtype=float),
            eligible.iloc[n_sample - bot_n:n_sample],
        ])
    else:
        chosen = eligible.iloc[:n_sample]

    chosen_ids = chosen.index.tolist()
    # Sort by calibration total (most active = row 0)
    chosen_ids.sort(key=lambda c: -calib_totals.get(c, 0))

    matrix = np.zeros((len(chosen_ids), T), dtype=np.int8)
    for row_i, cid in enumerate(chosen_ids):
        cust_w = weekly[weekly["customer_id"] == cid]
        for _, row in cust_w.iterrows():
            w = int(row["week"])
            if 0 <= w < T:
                matrix[row_i, w] = 1 if w < calib_weeks else 2

    return matrix, chosen_ids


def generate_figure_3_1(all_weekly: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.15})
    cmap = ListedColormap(["#f8f9fa", C_CALIB, C_HOLDOUT])  # 0=white, 1=calib, 2=holdout

    for ax, key in zip(axes.flat, KEYS):
        weekly = all_weekly[key]
        tc     = CALIB_WEEKS[key]
        th     = HOLDOUT_WEEKS[key]
        mat, _ = build_raster(weekly, tc, th, n_show=30)
        n_rows  = mat.shape[0]
        T_total = mat.shape[1]

        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="none", origin="upper",
                  extent=[-0.5, T_total - 0.5, n_rows - 0.5, -0.5])

        # Split line
        ax.axvline(tc - 0.5, color=C_SPLIT, lw=1.2, linestyle="--")
        ax.text(tc - 0.5, -1.5, "Calibration | Holdout",
                ha="center", va="bottom", fontsize=6, color=C_SPLIT)

        n_cust = weekly["customer_id"].nunique()
        ax.set_title(f"{LABELS[key]}\n($N={n_cust:,}$, $T_c={tc}$, $T_h={th}$ wks)",
                     fontsize=8, pad=4)
        ax.set_xlabel("Week", fontsize=7)
        ax.set_ylabel("Customer (sorted by activity)", fontsize=7)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=C_CALIB,   label="Calibration purchase"),
        mpatches.Patch(facecolor=C_HOLDOUT, label="Holdout purchase"),
        mpatches.Patch(facecolor="#f8f9fa", edgecolor="#cccccc", label="No purchase"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out = FIGURES_DIR / "figure_3_1_raster.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── Figure 3.2: frequency + spend distributions ───────────────────────────────

def kde_safe(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with fallback if bandwidth estimation fails."""
    vals = values[np.isfinite(values)]
    if len(vals) < 5 or vals.std() < 1e-9:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(vals, bw_method="silverman")
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)


def generate_figure_3_2(all_weekly: dict, all_raw: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(8, 10),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})

    for row_i, key in enumerate(KEYS):
        weekly = all_weekly[key]
        raw_df = all_raw[key]
        tc     = CALIB_WEEKS[key]

        # ── left panel: calibration purchase frequency per customer ───────────
        ax_l = axes[row_i, 0]
        calib_per_cust = (
            weekly[weekly["week"] < tc]
            .groupby("customer_id")["weekly_freq"].sum()
        )
        p99 = np.percentile(calib_per_cust, 99)
        cap = min(int(np.ceil(p99)), int(calib_per_cust.max()))
        # Bin: 1, 2, ..., cap-1, cap+
        bins  = np.arange(0.5, cap + 1.5)
        clipped = np.minimum(calib_per_cust.values, cap)
        counts, edges = np.histogram(clipped, bins=bins)

        x_pos  = np.arange(1, len(counts) + 1)
        colors = [C_CALIB] * (len(counts) - 1) + ["#6c757d"]  # last bar (top bin) slightly lighter
        ax_l.bar(x_pos, counts, width=0.8, color=colors, alpha=0.85, linewidth=0)
        ax_l.set_yscale("log")
        ax_l.set_xlabel(f"Calibration purchases per customer", fontsize=7)
        ax_l.set_ylabel("Customers (log scale)", fontsize=7)
        ax_l.set_title(LABELS[key], fontsize=8)
        ax_l.set_xlim(0.5, len(counts) + 0.5)
        # Annotate the top bin
        ax_l.text(len(counts), counts[-1] * 1.5, f"{cap}+",
                  ha="center", va="bottom", fontsize=6, color="#6c757d")
        ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{int(v):,}" if v >= 1 else ""
        ))

        # ── right panel: per-transaction spend (raw vs log1p) ────────────────
        ax_r  = axes[row_i, 1]
        spend = raw_df["transaction_amount"].values
        spend = spend[spend > 0]
        p99s  = np.percentile(spend, 99)
        spend_clipped = spend[spend <= p99s]

        # Raw spend KDE on bottom x-axis (grey, linear scale capped at p99)
        x_raw = np.linspace(0, p99s, 300)
        y_raw = kde_safe(spend_clipped, x_raw)
        ax_r.fill_between(x_raw, y_raw, alpha=0.45, color=C_SPEND_RAW, label="Raw spend")
        ax_r.plot(x_raw, y_raw, color=C_SPEND_RAW, lw=0.8)

        # log1p spend KDE on twin top x-axis (gold)
        ax_top = ax_r.twiny()
        log_spend = np.log1p(spend)
        x_log = np.linspace(0, log_spend.max(), 300)
        y_log = kde_safe(log_spend, x_log)
        # Normalise both to the same peak height so shapes are comparable
        if y_raw.max() > 0:
            y_log_scaled = y_log * (y_raw.max() / y_log.max()) if y_log.max() > 0 else y_log
        else:
            y_log_scaled = y_log
        ax_top.fill_between(x_log, y_log_scaled, alpha=0.5, color=C_SPEND_LOG, label="log1p spend")
        ax_top.plot(x_log, y_log_scaled, color=C_SPEND_LOG, lw=0.8)
        ax_top.set_xlabel("log$_{1p}$(spend)", fontsize=6.5, color=C_SPEND_LOG, labelpad=2)
        ax_top.tick_params(axis="x", labelsize=6, colors=C_SPEND_LOG)

        ax_r.set_xlabel(f"Spend ({CURRENCIES[key]}, clipped at 99th pctl)", fontsize=6.5,
                        color=C_SPEND_RAW, labelpad=2)
        ax_r.set_ylabel("Density", fontsize=7)
        ax_r.tick_params(axis="x", labelsize=6, colors=C_SPEND_RAW)
        ax_r.set_title(LABELS[key], fontsize=8)
        ax_r.set_ylim(bottom=0)

        # Legend in first row only
        if row_i == 0:
            from matplotlib.patches import Patch
            handles = [
                Patch(facecolor=C_SPEND_RAW, alpha=0.6, label="Raw spend (bottom axis)"),
                Patch(facecolor=C_SPEND_LOG, alpha=0.7, label="log$_{1p}$ spend (top axis)"),
            ]
            ax_r.legend(handles=handles, fontsize=6, loc="upper right", frameon=False)

    out = FIGURES_DIR / "figure_3_2_freq_spend.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── Figure 3.3: weekly aggregate activity timelines ───────────────────────────

def weekly_timelines(weekly: pd.DataFrame, tc: int, th: int) -> pd.DataFrame:
    """Return weekly aggregate: [week, n_active, total_spend]."""
    T = tc + th
    wdf = weekly[weekly["week"] < T].copy()
    agg = (
        wdf[wdf["weekly_freq"] > 0]
        .groupby("week")
        .agg(n_active=("customer_id", "nunique"),
             total_spend=("weekly_spend", "sum"))
        .reset_index()
    )
    # Fill missing weeks with 0
    all_weeks = pd.DataFrame({"week": np.arange(T)})
    return all_weeks.merge(agg, on="week", how="left").fillna(0)


def generate_figure_3_3(all_weekly: dict) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8, 9),
                             gridspec_kw={"hspace": 0.6})

    annotations = {
        "cdnow":     {52: "Christmas 1997"},
        "uci":       {52: "Dec 2010 peak"},
        "tafeng":    {},
        "dunnhumby": {},
    }

    for ax, key in zip(axes, KEYS):
        tc  = CALIB_WEEKS[key]
        th  = HOLDOUT_WEEKS[key]
        tl  = weekly_timelines(all_weekly[key], tc, th)

        ax2 = ax.twinx()
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        # Normalize both series to [0,1] for visual comparison
        max_active  = tl["n_active"].max()
        max_revenue = tl["total_spend"].max()
        norm_active  = tl["n_active"]  / max(max_active,  1)
        norm_revenue = tl["total_spend"] / max(max_revenue, 1)

        ax.fill_between(tl["week"], norm_active, alpha=0.3, color=C_ACTIVE, step="mid")
        ax.plot(tl["week"], norm_active, color=C_ACTIVE, lw=1.0,
                label=f"Active customers (max {max_active:,.0f})")

        ax2.fill_between(tl["week"], norm_revenue, alpha=0.25, color=C_REVENUE, step="mid")
        ax2.plot(tl["week"], norm_revenue, color=C_REVENUE, lw=1.0, linestyle="--",
                 label=f"Weekly revenue (max {CURRENCIES[key]} {max_revenue:,.0f})")

        ax.axvline(tc, color=C_SPLIT, lw=1.2, linestyle="--")
        ax.text(tc + 0.3, 0.95, "Holdout →", fontsize=6, color=C_SPLIT, va="top")

        for week_num, label in annotations.get(key, {}).items():
            if week_num < tc + th:
                ax.annotate(label, xy=(week_num, norm_active.iloc[min(week_num, len(norm_active)-1)]),
                            xytext=(week_num, 0.75), fontsize=6, color="#555",
                            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6),
                            ha="center")

        ax.set_ylabel("Normalised\nactive customers", fontsize=7, color=C_ACTIVE)
        ax2.set_ylabel("Normalised\nweekly revenue", fontsize=7, color=C_REVENUE)
        ax.tick_params(axis="y", colors=C_ACTIVE, labelsize=6)
        ax2.tick_params(axis="y", colors=C_REVENUE, labelsize=6)
        ax.tick_params(axis="x", labelsize=6)
        ax.set_xlabel("Week index", fontsize=7)
        ax.set_title(LABELS[key], fontsize=8)
        ax.set_xlim(-0.5, tc + th - 0.5)
        ax.set_ylim(0, 1.15)
        ax2.set_ylim(0, 1.15)
        ax2.set_yticks([0, 0.5, 1.0])

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=6, loc="upper left", frameon=False, ncol=2)

    out = FIGURES_DIR / "figure_3_3_timelines.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── Figure 3.4: Dunnhumby covariate profile ───────────────────────────────────

def generate_figure_3_4() -> None:
    # ── Panel A: income distribution ──────────────────────────────────────────
    demo = pd.read_csv(DUNNH_DIR / "hh_demographic.csv")
    income_counts = demo["INCOME_DESC"].value_counts()
    # Use known order, fill missing with 0
    income_vals = [income_counts.get(band, 0) for band in INCOME_ORDER]
    # Shorten labels
    short_income = [b.replace("Under ", "<") for b in INCOME_ORDER]

    # ── Panel B: household size distribution ──────────────────────────────────
    # Dunnhumby encodes as "1", "2", "3", "4", "5+"
    hh = demo["HOUSEHOLD_SIZE_DESC"].fillna("Unknown")
    hh_counts = hh.value_counts()
    hsize_vals = [hh_counts.get(s, 0) for s in HSIZE_ORDER]

    # ── Panel C: weekly coupon redemptions + campaign exposure ────────────────
    camp_desc  = pd.read_csv(DUNNH_DIR / "campaign_desc.csv")
    camp_table = pd.read_csv(DUNNH_DIR / "campaign_table.csv")
    redempt    = pd.read_csv(DUNNH_DIR / "coupon_redempt.csv")

    # Campaign exposure per week: count distinct households with an active campaign
    camp = camp_table.merge(
        camp_desc[["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN", how="left"
    ).dropna(subset=["START_DAY", "END_DAY"])
    camp["start_week"] = (camp["START_DAY"].astype(int) - 1) // 7
    camp["end_week"]   = (camp["END_DAY"].astype(int) - 1) // 7

    # Vectorized: for each campaign-household assignment, mark all active weeks
    exposure_records = []
    for _, row in camp.iterrows():
        sw, ew = int(row["start_week"]), int(row["end_week"])
        hh_key = row["household_key"]
        for w in range(sw, min(ew + 1, 102)):
            exposure_records.append((hh_key, w))

    exp_df = pd.DataFrame(exposure_records, columns=["household_key", "week"])
    weekly_exposure = exp_df.groupby("week")["household_key"].nunique().reset_index()
    weekly_exposure.columns = ["week", "n_exposed"]

    # Coupon redemptions per week
    redempt["week"] = (redempt["DAY"].astype(int) - 1) // 7
    weekly_redempt  = redempt.groupby("week").size().reset_index(name="n_redemptions")

    # Merge
    weeks_all = pd.DataFrame({"week": np.arange(102)})
    cov_weekly = (
        weeks_all
        .merge(weekly_exposure, on="week", how="left")
        .merge(weekly_redempt,  on="week", how="left")
        .fillna(0)
    )

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2),
                             gridspec_kw={"wspace": 0.45})

    # Panel A
    ax_a = axes[0]
    bars_a = ax_a.barh(short_income, income_vals, color=C_CALIB, alpha=0.8, height=0.7)
    ax_a.set_xlabel("Number of households", fontsize=7)
    ax_a.set_title("(A) Income band", fontsize=8)
    ax_a.tick_params(axis="both", labelsize=6.5)
    ax_a.set_xlim(0, max(income_vals) * 1.25)
    # Annotate counts
    total = demo["INCOME_DESC"].count()
    for bar, val in zip(bars_a, income_vals):
        pct = val / total * 100 if total > 0 else 0
        ax_a.text(val + max(income_vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{pct:.0f}%", va="center", fontsize=5.5, color="#555")

    # Panel B
    ax_b = axes[1]
    bars_b = ax_b.bar(HSIZE_ORDER, hsize_vals, color=C_CALIB, alpha=0.8, width=0.7)
    ax_b.set_xlabel("Household size", fontsize=7)
    ax_b.set_ylabel("Households", fontsize=7)
    ax_b.set_title("(B) Household size", fontsize=8)
    ax_b.tick_params(axis="both", labelsize=6.5)
    total_b = sum(hsize_vals)
    for bar, val in zip(bars_b, hsize_vals):
        pct = val / total_b * 100 if total_b > 0 else 0
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + total_b * 0.005,
                  f"{pct:.0f}%", ha="center", fontsize=6, color="#555")

    # Panel C
    ax_c  = axes[2]
    ax_c2 = ax_c.twinx()
    tc = CALIB_WEEKS["dunnhumby"]

    ax_c.bar(cov_weekly["week"], cov_weekly["n_exposed"], color=C_SPEND_RAW,
             alpha=0.7, width=1.0, label="Exposed households")
    ax_c2.plot(cov_weekly["week"], cov_weekly["n_redemptions"],
               color=C_HOLDOUT, lw=1.2, label="Coupon redemptions")

    ax_c.axvline(tc, color=C_SPLIT, lw=1.0, linestyle="--")
    y_top = cov_weekly["n_exposed"].max() * 0.85 if cov_weekly["n_exposed"].max() > 0 else 1
    ax_c.text(tc + 0.5, y_top, "Holdout", fontsize=6, color=C_SPLIT, va="top")

    ax_c.set_xlabel("Week index", fontsize=7)
    ax_c.set_ylabel("Households exposed", fontsize=7, color=C_SPEND_RAW)
    ax_c2.set_ylabel("Coupon redemptions", fontsize=7, color=C_HOLDOUT)
    ax_c.set_title("(C) Campaign + coupon activity", fontsize=8)
    ax_c.tick_params(axis="both", labelsize=6.5)
    ax_c2.tick_params(axis="y", labelsize=6.5, colors=C_HOLDOUT)
    ax_c.tick_params(axis="y", colors=C_SPEND_RAW)
    ax_c.spines["right"].set_visible(False)
    ax_c2.spines["top"].set_visible(False)

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2,
                fontsize=6, loc="upper left", frameon=False)

    out = FIGURES_DIR / "figure_3_4_dunnhumby_cov.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading datasets…")
    loaders = {
        "cdnow":     load_cdnow,
        "uci":       load_uci,
        "tafeng":    load_tafeng,
        "dunnhumby": load_dunnhumby,
    }
    all_raw    = {}
    all_weekly = {}
    all_stats  = []

    for key in KEYS:
        print(f"  [{key}]")
        raw    = loaders[key]()
        weekly = to_weekly(raw)
        tc     = CALIB_WEEKS[key]
        th     = HOLDOUT_WEEKS[key]
        calib_w = weekly[weekly["week"] < tc]
        stats   = compute_summary(key, raw, calib_w)

        all_raw[key]    = raw
        all_weekly[key] = weekly
        all_stats.append(stats)

        n = raw["customer_id"].nunique()
        print(f"      {n:,} customers, {len(raw):,} transactions, "
              f"{raw['date'].min().date()} – {raw['date'].max().date()}")

    print("\nGenerating Table 3.1…")
    generate_table_3_1(all_stats)

    print("Generating Figure 3.1 (raster)…")
    generate_figure_3_1(all_weekly)

    print("Generating Figure 3.2 (freq + spend)…")
    generate_figure_3_2(all_weekly, all_raw)

    print("Generating Figure 3.3 (timelines)…")
    generate_figure_3_3(all_weekly)

    print("Generating Figure 3.4 (Dunnhumby covariates)…")
    generate_figure_3_4()

    print(f"\nAll outputs saved to:\n  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
