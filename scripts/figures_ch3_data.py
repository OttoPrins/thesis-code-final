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

Alternative / Extra Analytical Visualisations:
    figure_3_alt_joint_dist.pdf     - Joint density of frequency and spend (P1 Proof)
    figure_3_alt_ipt.pdf            - Inter-Purchase Time (IPT) distributions (Poisson Critique)
    figure_3_alt_covariate_box.pdf  - Target-covariate boxplots for Dunnhumby (P3 Proof)
    figure_3_alt_silent_attrition.pdf - Empirical Recency-Frequency Silent Attrition Map
    figure_3_alt_hurdle_justification.pdf - Zero-Inflation vs. Active Log-Spend Distribution

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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
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
    "font.size":           9,
    "axes.labelsize":      9,
    "axes.titlesize":      10,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
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
    path = RAW_DIR / "ta_feng_all_months_merged.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("").upper() for c in df.columns]
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
    path = DUNNH_DIR / "transaction_data.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["household_key", "SALES_VALUE"])
    df = df[df["SALES_VALUE"] > 0]
    df["customer_id"] = df["household_key"].astype(int)
    df["transaction_amount"] = df["SALES_VALUE"].astype(float)
    df["date"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(df["DAY"].astype(int), unit="D")
    df["week"] = (df["DAY"].astype(int) - 1) // 7
    return (
        df.groupby(["customer_id", "BASKET_ID", "date", "week"])
        .agg(transaction_amount=("transaction_amount", "sum"))
        .reset_index()[["customer_id", "date", "week", "transaction_amount"]]
    )

# ── weekly aggregation ────────────────────────────────────────────────────────

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
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
    n_cust  = raw_df["customer_id"].nunique()
    n_trans = len(raw_df)
    period  = (f"{raw_df['date'].min().strftime('%b %Y')}"
               f"--{raw_df['date'].max().strftime('%b %Y')}")

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
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage{makecell}")
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \small")
    lines.append(r"  \caption{Cross-dataset summary statistics.}")
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

# ── ORIGINAL FIGURES ──────────────────────────────────────────────────────────

def build_raster(weekly: pd.DataFrame, calib_weeks: int, holdout_weeks: int,
                 n_show: int = 200) -> tuple[np.ndarray, list[int]]:
    T = calib_weeks + holdout_weeks
    calib_totals = (
        weekly[weekly["week"] < calib_weeks]
        .groupby("customer_id")["weekly_freq"].sum()
    )
    all_ids  = weekly["customer_id"].unique()
    n_sample = min(n_show, len(all_ids))
    chosen_ids = RNG.choice(all_ids, size=n_sample, replace=False).tolist()
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
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), gridspec_kw={"hspace": 0.48, "wspace": 0.15})
    cmap = ListedColormap(["#f8f9fa", C_CALIB, C_HOLDOUT])

    for idx, (ax, key) in enumerate(zip(axes.flat, KEYS)):
        weekly = all_weekly[key]
        tc, th = CALIB_WEEKS[key], HOLDOUT_WEEKS[key]
        mat, _ = build_raster(weekly, tc, th, n_show=200)
        n_rows, T_total  = mat.shape[0], mat.shape[1]

        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="none", origin="upper",
                  extent=[-0.5, T_total - 0.5, n_rows - 0.5, -0.5])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cccccc")
            spine.set_linewidth(0.5)

        ax.axvline(tc - 0.5, color=C_SPLIT, lw=1.2, linestyle="--")

        _bbox = dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.65, edgecolor="none")
        ax.text(tc * 0.5 - 0.5, 2, "← Calibration", ha="center", va="top", fontsize=5.5, color="#333", style="italic", bbox=_bbox)
        ax.text(tc + th * 0.5 - 0.5, 2, "Holdout →", ha="center", va="top", fontsize=5.5, color="#333", style="italic", bbox=_bbox)

        n_cust = weekly["customer_id"].nunique()
        ax.set_title(f"{LABELS[key]}\n$N$={n_cust:,},  $T_c$={tc},  $T_h$={th} wks", fontsize=10, pad=12)
        ax.text(-0.12, 1.12, chr(ord('a') + idx), transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        ax.set_xlabel("Week", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("Customer (sorted by activity)", fontsize=9)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    handles = [
        mpatches.Patch(facecolor=C_CALIB, label="Calibration purchase"),
        mpatches.Patch(facecolor=C_HOLDOUT, label="Holdout purchase"),
        mpatches.Patch(facecolor="#f8f9fa", edgecolor="#cccccc", label="No purchase"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    out = FIGURES_DIR / "figure_3_1_raster.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def kde_safe(values: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if len(vals) < 5 or vals.std() < 1e-9:
        return np.zeros_like(x_grid)
    try:
        kde = gaussian_kde(vals, bw_method="silverman")
        return kde(x_grid)
    except Exception:
        return np.zeros_like(x_grid)

def generate_figure_3_2(all_weekly: dict, all_raw: dict) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(8, 11), gridspec_kw={"hspace": 1.2, "wspace": 0.42})

    for row_i, key in enumerate(KEYS):
        weekly, raw_df, tc = all_weekly[key], all_raw[key], CALIB_WEEKS[key]

        # ── left panel ──
        ax_l = axes[row_i, 0]
        calib_per_cust = weekly[weekly["week"] < tc].groupby("customer_id")["weekly_freq"].sum()
        p99 = np.percentile(calib_per_cust, 99)
        cap = min(int(np.ceil(p99)), int(calib_per_cust.max()))

        bin_width = max(1, int(np.ceil(cap / 25)))
        bin_edges = np.arange(0.5, cap + bin_width + 0.5, bin_width)
        clipped   = np.minimum(calib_per_cust.values, cap)
        counts, _ = np.histogram(clipped, bins=bin_edges)
        x_pos     = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_w     = bin_width * 0.85

        colors = [C_CALIB] * (len(counts) - 1) + ["#6c757d"]
        ax_l.bar(x_pos, counts, width=bar_w, color=colors, alpha=0.85, linewidth=0)
        ax_l.set_yscale("log")
        ax_l.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='all', numticks=10))
        ax_l.tick_params(axis='y', which='minor', length=2, color='#aaa')
        ax_l.set_xlabel("Calibration purchases per customer", fontsize=9)
        ax_l.set_ylabel("Customers (log scale)", fontsize=9)
        ax_l.set_title(LABELS[key], fontsize=10, pad=4)
        ax_l.set_xlim(bin_edges[0], bin_edges[-1])
        ax_l.text(x_pos[-1], counts[-1] * 2.2, f"{cap}+", ha="center", va="bottom", fontsize=7, color="#6c757d")
        ax_l.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v >= 1 else ""))
        ax_l.text(-0.18, 1.14, chr(ord('a') + row_i * 2), transform=ax_l.transAxes, fontsize=10, fontweight='bold', va='top')

        # ── right panel ──
        ax_r  = axes[row_i, 1]
        spend = raw_df["transaction_amount"].values
        spend = spend[spend > 0]
        p99s  = np.percentile(spend, 99)
        spend_clipped = spend[spend <= p99s]

        x_raw = np.linspace(0, p99s, 300)
        y_raw = kde_safe(spend_clipped, x_raw)
        ax_r.fill_between(x_raw, y_raw, alpha=0.45, color=C_SPEND_RAW)
        ax_r.plot(x_raw, y_raw, color=C_SPEND_RAW, lw=0.8)

        ax_top = ax_r.twiny()
        log_spend = np.log1p(spend)
        x_log = np.linspace(0, log_spend.max(), 300)
        y_log = kde_safe(log_spend, x_log)
        y_log_scaled = y_log * (y_raw.max() / y_log.max()) if (y_raw.max() > 0 and y_log.max() > 0) else y_log

        ax_top.fill_between(x_log, y_log_scaled, alpha=0.5, color=C_SPEND_LOG)
        ax_top.plot(x_log, y_log_scaled, color=C_SPEND_LOG, lw=0.8)

        ax_top.set_xlabel(r"$\log(1+\mathrm{spend})$", fontsize=7, color=C_SPEND_LOG, labelpad=6)
        ax_top.tick_params(axis="x", labelsize=6.5, colors=C_SPEND_LOG, pad=2, length=3)

        ax_r.set_xlabel(f"Spend ({CURRENCIES[key]}, clipped at 99th pctl)", fontsize=8, color=C_SPEND_RAW, labelpad=2)
        ax_r.set_ylabel("Density", fontsize=9)
        ax_r.tick_params(axis="x", labelsize=7, colors=C_SPEND_RAW)
        ax_r.set_ylim(bottom=0)
        ax_r.text(-0.18, 1.14, chr(ord('a') + row_i * 2 + 1), transform=ax_r.transAxes, fontsize=10, fontweight='bold', va='top')

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=C_SPEND_RAW, alpha=0.6, label="Raw spend (bottom axis)"),
            mpatches.Patch(facecolor=C_SPEND_LOG, alpha=0.7, label=r"$\log(1+\mathrm{spend})$ (top axis)"),
        ],
        loc="lower center", ncol=2, fontsize=8, frameon=True, framealpha=0.95, edgecolor="#ddd", bbox_to_anchor=(0.72, -0.01),
    )

    out = FIGURES_DIR / "figure_3_2_freq_spend.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def weekly_timelines(weekly: pd.DataFrame, tc: int, th: int) -> pd.DataFrame:
    T = tc + th
    wdf = weekly[weekly["week"] < T].copy()
    agg = (
        wdf[wdf["weekly_freq"] > 0]
        .groupby("week")
        .agg(n_active=("customer_id", "nunique"), total_spend=("weekly_spend", "sum"))
        .reset_index()
    )
    all_weeks = pd.DataFrame({"week": np.arange(T)})
    return all_weeks.merge(agg, on="week", how="left").fillna(0)

def generate_figure_3_3(all_weekly: dict) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), gridspec_kw={"hspace": 0.6})
    annotations = {"cdnow": {}, "uci": {52: "Dec 2010 peak"}, "tafeng": {}, "dunnhumby": {}}

    global_lines, global_labels = [], []

    for idx, (ax, key) in enumerate(zip(axes, KEYS)):
        tc, th = CALIB_WEEKS[key], HOLDOUT_WEEKS[key]
        tl  = weekly_timelines(all_weekly[key], tc, th)

        ax2 = ax.twinx()
        ax.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        max_active, max_revenue = tl["n_active"].max(), tl["total_spend"].max()
        norm_active  = tl["n_active"]  / max(max_active,  1)
        norm_revenue = tl["total_spend"] / max(max_revenue, 1)

        fill1 = ax.fill_between(tl["week"], norm_active, alpha=0.3, color=C_ACTIVE, step="mid")
        line1, = ax.plot(tl["week"], norm_active, color=C_ACTIVE, lw=1.0)
        fill2 = ax2.fill_between(tl["week"], norm_revenue, alpha=0.25, color=C_REVENUE, step="mid")
        line2, = ax2.plot(tl["week"], norm_revenue, color=C_REVENUE, lw=1.0, linestyle="--")

        if idx == 0:
            global_lines.extend([(fill1, line1), (fill2, line2)])
            global_labels.extend(["Active customers", "Weekly revenue"])

        ax.axvline(tc, color=C_SPLIT, lw=1.6, linestyle="--", zorder=5)
        label_h = "Holdout →" if th >= 8 else "H/O"
        ax.text(tc + 0.3, 0.95, label_h, fontsize=7, color=C_SPLIT, va="top", clip_on=False)

        for week_num, label in annotations.get(key, {}).items():
            if week_num < tc + th:
                ax.annotate(label, xy=(week_num, norm_active.iloc[min(week_num, len(norm_active)-1)]),
                            xytext=(week_num, 0.75), fontsize=6, color="#555",
                            arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6), ha="center")

        info_box = (f"1.0 (Active) = {max_active:,.0f}\n1.0 (Rev) = {CURRENCIES[key]} {max_revenue:,.0f}")
        ax.text(0.02, 0.88, info_box, transform=ax.transAxes, fontsize=7, color="#444",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#ddd"), va="top")

        ax.set_ylabel("Normalised\nactive customers", fontsize=9, color=C_ACTIVE)
        ax2.set_ylabel("Normalised\nweekly revenue", fontsize=9, color=C_REVENUE)
        ax.tick_params(axis="y", colors=C_ACTIVE, labelsize=7)
        ax2.tick_params(axis="y", colors=C_REVENUE, labelsize=7)
        ax.tick_params(axis="x", labelsize=7)

        if key == KEYS[-1]:
            ax.set_xlabel("Week index", fontsize=9)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)

        ax.set_title(f"{LABELS[key]}   ($T_c$={tc}, $T_h$={th} wks)", fontsize=10)
        ax.set_xlim(-0.5, tc + th - 0.5)
        ax.set_ylim(0, 1.15)
        ax2.set_ylim(0, 1.15)
        ax2.set_yticks([0, 0.5, 1.0])

    fig.legend(global_lines, global_labels, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.subplots_adjust(bottom=0.1)

    out = FIGURES_DIR / "figure_3_3_timelines.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def generate_figure_3_4() -> None:
    demo = pd.read_csv(DUNNH_DIR / "hh_demographic.csv")
    income_counts = demo["INCOME_DESC"].value_counts()
    income_vals = [income_counts.get(band, 0) for band in INCOME_ORDER]
    short_income = [b.replace("Under ", "<") for b in INCOME_ORDER]

    hh = demo["HOUSEHOLD_SIZE_DESC"].fillna("Unknown")
    hh_counts = hh.value_counts()
    hsize_vals = [hh_counts.get(s, 0) for s in HSIZE_ORDER]

    camp_desc  = pd.read_csv(DUNNH_DIR / "campaign_desc.csv")
    camp_table = pd.read_csv(DUNNH_DIR / "campaign_table.csv")
    redempt    = pd.read_csv(DUNNH_DIR / "coupon_redempt.csv")

    camp = camp_table.merge(camp_desc[["CAMPAIGN", "START_DAY", "END_DAY"]], on="CAMPAIGN", how="left").dropna(subset=["START_DAY", "END_DAY"])
    camp["start_week"] = (camp["START_DAY"].astype(int) - 1) // 7
    camp["end_week"]   = (camp["END_DAY"].astype(int) - 1) // 7

    exposure_records = []
    for _, row in camp.iterrows():
        sw, ew = int(row["start_week"]), int(row["end_week"])
        for w in range(sw, min(ew + 1, 102)):
            exposure_records.append((row["household_key"], w))

    exp_df = pd.DataFrame(exposure_records, columns=["household_key", "week"])
    weekly_exposure = exp_df.groupby("week")["household_key"].nunique().reset_index(name="n_exposed")

    redempt["week"] = (redempt["DAY"].astype(int) - 1) // 7
    weekly_redempt  = redempt.groupby("week").size().reset_index(name="n_redemptions")

    weeks_all = pd.DataFrame({"week": np.arange(102)})
    cov_weekly = weeks_all.merge(weekly_exposure, on="week", how="left").merge(weekly_redempt, on="week", how="left").fillna(0)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), gridspec_kw={"wspace": 0.45})

    ax_a = axes[0]
    bars_a = ax_a.barh(short_income, income_vals, color=C_CALIB, alpha=0.8, height=0.7)
    ax_a.set_xlabel("Number of households", fontsize=9)
    ax_a.set_title("(A) Income band", fontsize=10)
    ax_a.tick_params(axis="both", labelsize=7.5)
    ax_a.set_xlim(0, max(income_vals) * 1.28)
    total = demo["INCOME_DESC"].count()
    for bar, val in zip(bars_a, income_vals):
        pct = val / total * 100 if total > 0 else 0
        ax_a.text(val + max(income_vals) * 0.02, bar.get_y() + bar.get_height() / 2, f"{pct:.0f}%", va="center", fontsize=6.5, color="#555")

    ax_b = axes[1]
    bars_b = ax_b.bar(HSIZE_ORDER, hsize_vals, color=C_CALIB, alpha=0.8, width=0.7)
    ax_b.set_xlabel("Household size", fontsize=9)
    ax_b.set_ylabel("Households", fontsize=9)
    ax_b.set_title("(B) Household size", fontsize=10)
    ax_b.tick_params(axis="both", labelsize=7.5)
    total_b = sum(hsize_vals)
    for bar, val in zip(bars_b, hsize_vals):
        pct = val / total_b * 100 if total_b > 0 else 0
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + max(hsize_vals) * 0.02, f"{pct:.0f}%", ha="center", fontsize=7, color="#555")

    ax_c  = axes[2]
    ax_c2 = ax_c.twinx()
    tc = CALIB_WEEKS["dunnhumby"]

    ax_c.bar(cov_weekly["week"], cov_weekly["n_exposed"], color=C_SPEND_RAW, alpha=0.7, width=1.0, label="Exposed households")
    ax_c2.plot(cov_weekly["week"], cov_weekly["n_redemptions"], color=C_HOLDOUT, lw=1.2, label="Coupon redemptions")

    ax_c.axvline(tc, color=C_SPLIT, lw=1.6, linestyle="--", zorder=5)
    ax_c.text(tc + 0.5, 0.94, "Holdout", fontsize=7, color=C_SPLIT, va="top", transform=ax_c.get_xaxis_transform(), clip_on=False)

    ax_c.set_xlabel("Week index", fontsize=9)
    ax_c.set_ylabel("Households exposed", fontsize=9, color=C_SPEND_RAW)
    ax_c2.set_ylabel("Coupon redemptions", fontsize=9, color=C_HOLDOUT)
    ax_c.set_title("(C) Campaign + coupon activity", fontsize=10)
    ax_c.tick_params(axis="both", labelsize=7.5)
    ax_c2.tick_params(axis="y", labelsize=7.5, colors=C_HOLDOUT)
    ax_c.tick_params(axis="y", colors=C_SPEND_RAW)
    ax_c.spines["right"].set_visible(False)
    ax_c2.spines["top"].set_visible(False)

    lines1, labels1 = ax_c.get_legend_handles_labels()
    lines2, labels2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right", frameon=True, facecolor="white", framealpha=1.0, edgecolor="#dddddd")

    out = FIGURES_DIR / "figure_3_4_dunnhumby_cov.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

# ── EXTRA ANALYTICAL VISUALISATIONS ───────────────────────────────────────────

def generate_figure_alt_joint_dist(all_weekly: dict, all_raw: dict) -> None:
    fig = plt.figure(figsize=(10, 8))
    outer_grid = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.4)

    for idx, key in enumerate(KEYS):
        weekly, raw_df, tc = all_weekly[key], all_raw[key], CALIB_WEEKS[key]

        freq_df = weekly[weekly["week"] < tc].groupby("customer_id")["weekly_freq"].sum()
        calib_raw = raw_df[raw_df["date"] <= raw_df["date"].min() + pd.Timedelta(weeks=tc)]
        spend_df = calib_raw.groupby("customer_id")["transaction_amount"].mean()

        joint_df = pd.concat([freq_df, spend_df], axis=1).dropna()
        joint_df.columns = ["freq", "avg_spend"]

        f_cap = np.percentile(joint_df["freq"], 99)
        s_cap = np.percentile(joint_df["avg_spend"], 99)
        joint_df = joint_df[(joint_df["freq"] <= f_cap) & (joint_df["avg_spend"] <= s_cap)]

        x = joint_df["freq"].values
        y = np.log1p(joint_df["avg_spend"].values)

        inner_grid = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=outer_grid[idx], wspace=0.05, hspace=0.05)

        ax_main = fig.add_subplot(inner_grid[1:4, 0:3])
        ax_top  = fig.add_subplot(inner_grid[0, 0:3], sharex=ax_main)
        ax_right= fig.add_subplot(inner_grid[1:4, 3], sharey=ax_main)

        ax_main.hexbin(x, y, gridsize=20, cmap="YlGnBu", bins='log', mincnt=1, edgecolors='none')
        ax_main.set_xlabel("Calibration Purchases", fontsize=9)
        if idx % 2 == 0:
            ax_main.set_ylabel(r"Avg. Spend ($\log(1+x)$)", fontsize=9)

        corr = np.corrcoef(x, y)[0, 1]
        ax_main.text(0.95, 0.95, f"$r = {corr:.2f}$", transform=ax_main.transAxes,
                     ha="right", va="top", fontsize=9, fontweight="bold", color="#333",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

        ax_top.hist(x, bins=20, color=C_FREQ, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax_top.axis('off')
        ax_top.set_title(LABELS[key], fontsize=10, pad=10)
        ax_top.text(-0.1, 1.2, chr(ord('a') + idx), transform=ax_top.transAxes, fontsize=11, fontweight='bold', va='top')

        ax_right.hist(y, bins=30, color=C_SPEND_LOG, alpha=0.85, orientation='horizontal', edgecolor='white', linewidth=0.5)
        ax_right.axis('off')

        ax_main.spines["top"].set_visible(False)
        ax_main.spines["right"].set_visible(False)
        ax_main.tick_params(axis="both", labelsize=8)

    out = FIGURES_DIR / "figure_3_alt_joint_dist.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def generate_figure_alt_ipt(all_raw: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw={"hspace": 0.45, "wspace": 0.2})

    for idx, (ax, key) in enumerate(zip(axes.flat, KEYS)):
        df = all_raw[key].copy().sort_values(["customer_id", "date"])
        df["ipt"] = df.groupby("customer_id")["date"].diff().dt.days
        ipts = df["ipt"].dropna()
        ipts = ipts[ipts > 0]

        bins = np.arange(1, min(90, ipts.max()), 2)
        ax.hist(ipts, bins=bins, color=C_FREQ, alpha=0.8, density=True, edgecolor='white', linewidth=0.5)

        ax.axvline(7, color=C_SPLIT, linestyle='--', lw=1.2, alpha=0.8)
        ax.text(7.5, ax.get_ylim()[1]*0.9, '7 Days', color=C_SPLIT, fontsize=7, rotation=90, va='top')

        if key in ["dunnhumby", "tafeng"]:
            ax.axvline(14, color=C_SPLIT, linestyle='--', lw=1.2, alpha=0.8)
            ax.text(14.5, ax.get_ylim()[1]*0.9, '14 Days', color=C_SPLIT, fontsize=7, rotation=90, va='top')

        ax.set_title(LABELS[key], fontsize=10)
        ax.set_xlabel("Inter-Purchase Time (Days)", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("Density", fontsize=9)

        ax.text(-0.1, 1.1, chr(ord('a') + idx), transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "figure_3_alt_ipt.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def generate_figure_alt_covariate_boxplots(all_weekly: dict) -> None:
    weekly = all_weekly["dunnhumby"]
    demo = pd.read_csv(DUNNH_DIR / "hh_demographic.csv")

    agg = weekly[weekly["weekly_freq"] > 0].groupby("customer_id").agg(
        avg_spend=("weekly_spend", "mean"), total_freq=("weekly_freq", "sum")
    ).reset_index()

    demo["customer_id"] = demo["household_key"].astype(int)
    merged = agg.merge(demo, on="customer_id", how="inner")
    merged["income_cat"] = pd.Categorical(merged["INCOME_DESC"], categories=INCOME_ORDER, ordered=True)
    merged["size_cat"] = pd.Categorical(merged["HOUSEHOLD_SIZE_DESC"].fillna("Unknown"), categories=HSIZE_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"wspace": 0.25})

    inc_data = [merged[merged["income_cat"] == inc]["avg_spend"].values for inc in INCOME_ORDER]
    inc_data = [np.log1p(d) for d in inc_data if len(d) > 0]

    axes[0].boxplot(inc_data, patch_artist=True, notch=True,
                    boxprops=dict(facecolor=C_CALIB, color=C_CALIB, alpha=0.7),
                    medianprops=dict(color=C_SPLIT, lw=2), flierprops=dict(marker='o', markersize=2, alpha=0.2))
    axes[0].set_xticklabels([inc.replace("Under ", "<") for inc in INCOME_ORDER], rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel(r"Avg Weekly Spend ($\log(1+x)$)", fontsize=9)
    axes[0].set_title("(A) Spend Distribution by Income Band", fontsize=10)

    size_data = [merged[merged["size_cat"] == s]["total_freq"].values for s in HSIZE_ORDER]
    axes[1].boxplot(size_data, patch_artist=True, notch=True,
                    boxprops=dict(facecolor=C_CALIB, color=C_CALIB, alpha=0.7),
                    medianprops=dict(color=C_SPLIT, lw=2), flierprops=dict(marker='o', markersize=2, alpha=0.2))
    axes[1].set_xticklabels(HSIZE_ORDER, fontsize=8)
    axes[1].set_ylabel("Total Purchase Frequency", fontsize=9)
    axes[1].set_title("(B) Frequency Distribution by Household Size", fontsize=10)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "figure_3_alt_covariate_box.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def generate_figure_alt_silent_attrition(all_weekly: dict) -> None:
    """Creates an empirical Recency-Frequency map linking calibration activity directly to holdout retention."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), gridspec_kw={"hspace": 0.38, "wspace": 0.25})

    for idx, key in enumerate(KEYS):
        ax = axes.flat[idx]
        w = all_weekly[key]
        tc = CALIB_WEEKS[key]

        # Calculate Calibration summary stats per customer
        calib = w[w["week"] < tc]
        active_c = calib[calib["weekly_freq"] > 0]

        c_stats = active_c.groupby("customer_id").agg(
            freq=("week", "count"),
            recency=("week", "max")  # Last week with purchase before split
        )

        # Did they buy in holdout window?
        h_active = w[(w["week"] >= tc) & (w["weekly_freq"] > 0)]["customer_id"].unique()
        c_stats["survived"] = c_stats.index.isin(h_active).astype(int)

        # Leverage hexbin computing the mean target profile per density sector
        hb = ax.hexbin(c_stats["recency"], c_stats["freq"], C=c_stats["survived"],
                        reduce_C_function=np.mean, gridsize=15, cmap="YlGnBu", mincnt=1, edgecolors='none')

        ax.set_title(LABELS[key], fontsize=10)
        ax.set_xlabel("Recency (Last Active Calibration Week)", fontsize=9)
        if idx % 2 == 0:
            ax.set_ylabel("Frequency (Active Weeks)", fontsize=9)

        ax.text(-0.1, 1.1, chr(ord('a') + idx), transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        cb = fig.colorbar(hb, ax=ax, orientation='vertical', pad=0.02)
        cb.set_label("Empirical Holdout Purchase Prob.", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = FIGURES_DIR / "figure_3_alt_silent_attrition.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {out.name}")

def generate_figure_alt_hurdle_justification(all_weekly: dict) -> None:
    """Draws side-by-side plots proving panel zero-inflation and log-normal active behavior."""
    fig, axes = plt.subplots(4, 2, figsize=(9, 11), gridspec_kw={"hspace": 0.6, "wspace": 0.3})

    for idx, key in enumerate(KEYS):
        w = all_weekly[key]
        tc = CALIB_WEEKS[key]
        calib_spend = w[w["week"] < tc]["weekly_spend"].values

        ax_l = axes[idx, 0]
        ax_r = axes[idx, 1]

        # Left Panel: Raw distribution highlighting zero dominance
        ax_l.hist(calib_spend, bins=40, color=C_FREQ, alpha=0.8, edgecolor='white', linewidth=0.3)
        ax_l.set_title(f"{LABELS[key]}: Full Spend Sequence (with Zeros)", fontsize=9)
        ax_l.set_ylabel("Observations")
        ax_l.set_xlabel(f"Weekly Value ({CURRENCIES[key]})")

        # Highlight zero spike with callout marker
        zeros = np.sum(calib_spend == 0)
        pct_zero = (zeros / len(calib_spend)) * 100
        ax_l.text(0.4, 0.75, f"{pct_zero:.1f}% Weeks\nAre Exactly 0", transform=ax_l.transAxes,
                  color=C_SPLIT, fontweight="bold", fontsize=7.5, bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.9, edgecolor="#ddd"))

        # Right Panel: Active log distribution demonstrating clean parametric tracking shape
        pos_spend = calib_spend[calib_spend > 0]
        ax_r.hist(np.log1p(pos_spend), bins=30, color=C_SPEND_LOG, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax_r.set_title(f"{LABELS[key]}: Active Weeks Only", fontsize=9)
        ax_r.set_xlabel(r"$\log(1+\mathrm{spend})$")
        ax_r.set_ylabel("Observations")

        for ax in [ax_l, ax_r]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=7.5)

    out = FIGURES_DIR / "figure_3_alt_hurdle_justification.pdf"
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
        tc, th = CALIB_WEEKS[key], HOLDOUT_WEEKS[key]
        calib_w = weekly[weekly["week"] < tc]
        stats   = compute_summary(key, raw, calib_w)

        all_raw[key]    = raw
        all_weekly[key] = weekly
        all_stats.append(stats)

        n = raw["customer_id"].nunique()
        print(f"      {n:,} customers, {len(raw):,} transactions, "
              f"{raw['date'].min().date()} – {raw['date'].max().date()}")

    print("\nGenerating Original Pipeline Visualisations…")
    generate_table_3_1(all_stats)
    generate_figure_3_1(all_weekly)
    generate_figure_3_2(all_weekly, all_raw)
    generate_figure_3_3(all_weekly)
    generate_figure_3_4()

    print("\nGenerating Analytical Alternative Extras…")
    generate_figure_alt_joint_dist(all_weekly, all_raw)
    generate_figure_alt_ipt(all_raw)
    generate_figure_alt_covariate_boxplots(all_weekly)
    generate_figure_alt_silent_attrition(all_weekly)
    generate_figure_alt_hurdle_justification(all_weekly)

    print(f"\nAll outputs saved to:\n  {FIGURES_DIR}")


if __name__ == "__main__":
    main()
