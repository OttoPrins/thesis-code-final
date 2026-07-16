#!/usr/bin/env python3
"""
build_tuned_results.py — Tuned-performance analysis artifacts (T7/T8/T9, F10/F11).

Post-processes the output of kaggle_tuned_runner.ipynb. Run AFTER downloading
the Kaggle archive:

    unzip results_archive_tuned.zip -d results/final_kaggle_tuned
    python build_tuned_results.py

Consumes:
    results/final_kaggle_tuned/   — tuned runs (10 configs x 3 seeds), tuned
                                    Pareto/NBD re-fit, tuned SHAP summary,
                                    archived HPO winner + tuned configs
    results/final_kaggle/         — frozen fixed-parameter baselines (thesis)

Emits into results/thesis_final_v2/ (additive — never rewrites T1–T6/F1–F9):
    tables/T7_tuned_headline.{csv,tex}      fixed vs tuned, Dunnhumby 80/22
    tables/T7b_tuned_significance.{csv,tex} paired customer bootstrap, Holm-adj.
    tables/T8_tuned_ablation.{csv,tex}      covariate ablation under tuned HPs
    tables/T9_tuned_shap.{csv,tex}          dual-head SHAP under tuned HPs
    figures/F10_tuning_effect.{pdf,png}
    figures/F11_tuned_ablation.{pdf,png}
    TUNED_RESULTS_ADDENDUM.md               writing-guide addendum
    TUNED_ARTIFACT_MANIFEST.json            sha256 provenance of the above

Methodology notes baked in here:
  * Monetary/CLV metrics for the 80/22 models are rescored against the same
    raw-holdout truth matrix as the frozen thesis tables (the Pareto/NBD
    arrays.npz aggregation); the fixed and tuned truth matrices are asserted
    byte-identical before any paired comparison.
  * The paired bootstrap follows the T4 protocol exactly (5,000 resamples,
    bootstrap seed 42, seed-averaged deep-model errors, Holm adjustment across
    the displayed contrasts). Pairing across sessions is valid because the
    pipeline, split seeds, and holdout are deterministic and identical.
  * Extension 3 (80/4) rows are reported as stored, mirroring T5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import build_thesis_results as bts
from build_thesis_results import (
    EXTENSION3_LABELS,
    EXTENSION3_VARIANTS,
    MODEL_COLORS,
    SEEDS,
    WEEKLY_DISCOUNT_RATE,
    _format_p,
    _holm_adjust,
    _paired_bootstrap_seed_mean,
    _savefig,
    _savetable,
)
from src.evaluation.compare import aggregate_all_results
from src.evaluation.metrics import compute_all_metrics, per_customer_clv

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
TUNED = Path("results/final_kaggle_tuned")
FIXED = Path("results/final_kaggle")
OUT = bts.OUT

DATASET = "dunnhumby"
MAIN_ARCHS = ["lstm_joint", "transformer_joint"]
ARCH_LABELS = {"lstm_joint": "Joint LSTM", "transformer_joint": "Joint Transformer"}
MONEY_MODELS = {"pareto_nbd", "lstm_joint", "transformer_joint"}

HEADLINE_METRICS = [
    "freq_rmse", "freq_mape", "bias_pct",
    "spend_r2_log", "spend_mae_raw",
    "clv_mae", "clv_spearman", "clv_decile_lift",
]
METRIC_HEADERS = {
    "freq_rmse": "Freq RMSE", "freq_mape": r"Freq MAPE\,\%", "bias_pct": r"Bias\,\%",
    "spend_r2_log": r"Spend $R^2$", "spend_mae_raw": "Spend MAE",
    "clv_mae": "CLV MAE", "clv_spearman": r"CLV $\rho$", "clv_decile_lift": "Decile lift",
}

BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 42
TUNED_SHAP_CSV = TUNED / "tables" / "shap_extension3_summary.csv"
FIXED_SHAP_CSV = FIXED / "tables" / "shap_extension3_summary.csv"
TUNED_SHAP_ADDITIVITY_CSV = TUNED / "tables" / "shap_extension3_additivity.csv"

_EMITTED: list[Path] = []


def _track(paths) -> None:
    _EMITTED.extend(paths if isinstance(paths, list) else [paths])


# ---------------------------------------------------------------------------
# Loading and rescoring
# ---------------------------------------------------------------------------
def _arrays_path(root: Path, run_name: str) -> Path:
    path = root / "tables" / f"{run_name}_arrays.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing array artifact: {path}")
    return path


def _canonical_targets() -> tuple[np.ndarray, np.ndarray]:
    """Raw-holdout truth from the FIXED Pareto/NBD fit; assert the tuned
    session's re-fit reproduced it exactly (cross-session integrity check)."""
    with np.load(_arrays_path(FIXED, f"pareto_nbd_{DATASET}"), allow_pickle=False) as d:
        true_freq = np.asarray(d["per_week_true_freq"], dtype=np.float64)
        true_spend = np.asarray(d["per_week_true_spend"], dtype=np.float64)
    with np.load(_arrays_path(TUNED, f"pareto_nbd_{DATASET}"), allow_pickle=False) as d:
        if not np.array_equal(true_freq, np.asarray(d["per_week_true_freq"])):
            raise RuntimeError("Tuned-session holdout truth differs from the thesis truth "
                               "(frequency): sessions are not comparable.")
        if not np.allclose(true_spend, np.asarray(d["per_week_true_spend"])):
            raise RuntimeError("Tuned-session holdout truth differs from the thesis truth "
                               "(spend): sessions are not comparable.")
    return true_freq, true_spend


def _rescore_monetary(df: pd.DataFrame, root: Path,
                      true_freq_week: np.ndarray, true_spend_week: np.ndarray) -> pd.DataFrame:
    """Mirror of bts._rescore_monetary_against_raw_truth for one session dir."""
    out = df.copy()
    true_freq_total = true_freq_week.sum(axis=1)
    customer_ids = np.arange(len(true_freq_total))
    idxs = out[(out["dataset"] == DATASET) & (out["model"].isin(MONEY_MODELS))].index
    for idx in idxs:
        run_name = str(out.at[idx, "run_name"])
        with np.load(_arrays_path(root, run_name), allow_pickle=False) as d:
            pred_freq_week = np.asarray(d["per_week_pred_freq"], dtype=np.float64)
            pred_spend_week = np.asarray(d["per_week_pred_spend"], dtype=np.float64)
            stored_true_freq = np.asarray(d["per_week_true_freq"], dtype=np.float64)
        if not np.array_equal(stored_true_freq, true_freq_week):
            raise RuntimeError(f"Customer/holdout alignment failed for {run_name}.")
        rescored = compute_all_metrics(
            y_freq_true=true_freq_total,
            y_freq_pred=pred_freq_week.sum(axis=1),
            customer_ids=customer_ids,
            y_freq_true_per_week=true_freq_week,
            y_freq_pred_per_week=pred_freq_week,
            y_spend_true_per_week_raw=true_spend_week,
            y_spend_pred_per_week_raw=pred_spend_week,
            freq_mase_scale=out.at[idx, "freq_mase_scale"]
            if "freq_mase_scale" in out.columns and pd.notna(out.at[idx, "freq_mase_scale"])
            else None,
            spend_mase_scale=out.at[idx, "spend_mase_scale"]
            if "spend_mase_scale" in out.columns and pd.notna(out.at[idx, "spend_mase_scale"])
            else None,
            weekly_discount_rate=WEEKLY_DISCOUNT_RATE,
        )
        for key, value in rescored.items():
            if key.startswith(("spend_", "clv_")) and not key.startswith("_"):
                out.at[idx, key] = value
    return out


def load_sessions() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TUNED.exists():
        raise SystemExit(
            f"{TUNED} not found. Download results_archive_tuned.zip from the Kaggle "
            f"kernel and unzip it there first."
        )
    true_freq_week, true_spend_week = _canonical_targets()

    df_tuned = aggregate_all_results(results_dir=str(TUNED), final_only=False)
    df_tuned = df_tuned[(df_tuned["dataset"] == DATASET)].copy()
    df_tuned = df_tuned[
        (df_tuned["model"] == "pareto_nbd") | (df_tuned["version"] == "tuned")
    ]
    _validate_inventory(df_tuned, "tuned")

    df_fixed = aggregate_all_results(results_dir=str(FIXED), final_only=True)
    df_fixed = df_fixed[df_fixed["dataset"] == DATASET].copy()
    _validate_inventory(df_fixed, "fixed")

    _assert_protocol_parity(df_fixed, df_tuned)

    df_tuned = _rescore_monetary(df_tuned, TUNED, true_freq_week, true_spend_week)
    df_fixed = _rescore_monetary(df_fixed, FIXED, true_freq_week, true_spend_week)
    return df_fixed, df_tuned


def _validate_inventory(df: pd.DataFrame, session: str) -> None:
    expected = MAIN_ARCHS + [
        f"extension3_{arch}_{variant}"
        for arch in ("lstm", "transformer") for variant in EXTENSION3_VARIANTS
    ]
    problems = []
    if len(df[df["model"] == "pareto_nbd"]) != 1:
        problems.append(f"pareto_nbd: expected 1 fit, found {len(df[df['model'] == 'pareto_nbd'])}")
    for model in expected:
        sub = df[df["model"] == model]
        seeds = {int(s) for s in sub["seed"].dropna().astype(int)}
        if seeds != set(SEEDS):
            problems.append(f"{model}: expected seeds {SEEDS}, found {sorted(seeds)}")
        if "run_valid" in sub.columns and not all(
            bool(v) for v in sub["run_valid"].tolist() if pd.notna(v)
        ):
            problems.append(f"{model}: contains run_valid=False rows")
    if problems:
        raise RuntimeError(f"Incomplete {session} inventory:\n  " + "\n  ".join(problems))


def _assert_protocol_parity(df_fixed: pd.DataFrame, df_tuned: pd.DataFrame) -> None:
    """Fixed and tuned runs must share cohort and holdout protocol exactly."""
    for col in ("cohort_size", "calibration_weeks", "holdout_weeks"):
        if col not in df_fixed.columns or col not in df_tuned.columns:
            continue
        for model in MAIN_ARCHS + ["extension3_lstm_full", "extension3_transformer_full"]:
            fx = df_fixed.loc[df_fixed["model"] == model, col].dropna().unique()
            tn = df_tuned.loc[df_tuned["model"] == model, col].dropna().unique()
            if len(fx) and len(tn) and set(fx) != set(tn):
                raise RuntimeError(
                    f"Protocol drift for {model}.{col}: fixed={fx} tuned={tn}"
                )


# ---------------------------------------------------------------------------
# Winning hyperparameters (from the archived tuned configs)
# ---------------------------------------------------------------------------
def _flatten(node, prefix=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = node
    return out


def winning_hps() -> dict[str, list[str]]:
    """Changed keys (base -> tuned) per tuned config, from the archived YAMLs."""
    ignore = {"training.epochs", "training.max_epochs", "output.run_name"}
    changes: dict[str, list[str]] = {}
    cfg_dir = TUNED / "configs_tuned"
    if not cfg_dir.exists():
        print(f"  WARNING: {cfg_dir} missing — winning HPs omitted from captions.")
        return changes
    for tuned_path in sorted(cfg_dir.glob("*.yaml")):
        base_path = Path("experiments/configs_final") / tuned_path.name.replace(
            "_tuned.yaml", "_final.yaml"
        )
        if not base_path.exists():
            continue
        flat_t = _flatten(yaml.safe_load(tuned_path.read_text()))
        flat_b = _flatten(yaml.safe_load(base_path.read_text()))
        diff = [
            f"{k}: {flat_b.get(k)!r} -> {flat_t[k]!r}"
            for k in sorted(flat_t)
            if flat_t.get(k) != flat_b.get(k) and k not in ignore
        ]
        changes[tuned_path.stem] = diff
    return changes


# ---------------------------------------------------------------------------
# T7 — Tuned vs fixed headline table
# ---------------------------------------------------------------------------
def _mean_std(df: pd.DataFrame, model: str, metric: str) -> tuple[float, float, int]:
    vals = df.loc[df["model"] == model, metric].dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def make_table_t7(df_fixed: pd.DataFrame, df_tuned: pd.DataFrame,
                  hps: dict[str, list[str]]) -> pd.DataFrame:
    print("[T7] Tuned vs fixed headline (Dunnhumby 80/22)...")
    rows = []
    pareto = {m: _mean_std(df_fixed, "pareto_nbd", m) for m in HEADLINE_METRICS}
    rows.append({"Model": "Pareto/NBD + GG", "Config": "—", "N_seeds": 1,
                 **{m: pareto[m][0] for m in HEADLINE_METRICS},
                 **{f"{m}_std": float("nan") for m in HEADLINE_METRICS}})
    for arch in MAIN_ARCHS:
        for label, df in (("fixed", df_fixed), ("tuned", df_tuned)):
            stats = {m: _mean_std(df, arch, m) for m in HEADLINE_METRICS}
            rows.append({"Model": ARCH_LABELS[arch], "Config": label,
                         "N_seeds": stats[HEADLINE_METRICS[0]][2],
                         **{m: stats[m][0] for m in HEADLINE_METRICS},
                         **{f"{m}_std": stats[m][1] for m in HEADLINE_METRICS}})
        fixed_row = rows[-2]
        tuned_row = rows[-1]
        rows.append({"Model": ARCH_LABELS[arch], "Config": "Δ (tuned − fixed)",
                     "N_seeds": 3,
                     **{m: tuned_row[m] - fixed_row[m] for m in HEADLINE_METRICS},
                     **{f"{m}_std": float("nan") for m in HEADLINE_METRICS}})
    df = pd.DataFrame(rows)

    display_rows = []
    for _, row in df.iterrows():
        disp = {"Model": row["Model"], "Config": row["Config"], "N_seeds": row["N_seeds"]}
        for m in HEADLINE_METRICS:
            if row["Config"] == "Δ (tuned − fixed)":
                disp[m] = f"{row[m]:+.3f}"
            elif np.isnan(row[f"{m}_std"]):
                disp[m] = f"{row[m]:.3f}"
            else:
                disp[m] = f"{row[m]:.3f} ± {row[f'{m}_std']:.3f}"
        display_rows.append(disp)
    display_df = pd.DataFrame(display_rows)

    hp_notes = []
    for stem in ("lstm_joint_dunnhumby_tuned", "transformer_joint_dunnhumby_tuned"):
        if stem in hps:
            label = "LSTM" if stem.startswith("lstm") else "Transformer"
            joined = "; ".join(h.replace("_", r"\_") for h in hps[stem]) or "grid winner = fixed defaults"
            hp_notes.append(rf"\item {label}: {joined}")

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{threeparttable}",
        r"\caption{Tuned versus fixed-configuration performance on Dunnhumby "
        r"(80/22 protocol). Hyperparameters were selected by an exhaustive grid "
        r"search scored on a validation window carved from the last 20\% of "
        r"calibration weeks (the true holdout was never seen during selection); "
        r"winners were retrained on the full calibration window under three seeds "
        r"with the epoch cap lifted to 250 (fixed runs terminated at the 150 cap "
        r"with early stopping never firing). Monetary metrics are rescored against "
        r"the same raw-holdout truth as Table T2. Values are mean $\pm$ SD over "
        r"three seeds; the $\Delta$ rows are differences of seed means.}",
        r"\label{tab:tuned_headline}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(HEADLINE_METRICS) + "}",
        r"\toprule",
        "Model & Config & " + " & ".join(METRIC_HEADERS[m] for m in HEADLINE_METRICS) + r" \\",
        r"\midrule",
    ]
    prev_model = None
    for _, row in df.iterrows():
        if prev_model and row["Model"] != prev_model:
            lines.append(r"\midrule")
        prev_model = row["Model"]
        cells = [row["Model"], row["Config"].replace("Δ (tuned − fixed)", r"$\Delta$")]
        for m in HEADLINE_METRICS:
            if row["Config"] == "Δ (tuned − fixed)":
                cells.append(f"{row[m]:+.3f}")
            elif np.isnan(row[f"{m}_std"]):
                cells.append(f"{row[m]:.3f}")
            else:
                cells.append(f"{row[m]:.3f} $\\pm$ {row[f'{m}_std']:.3f}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}"]
    if hp_notes:
        lines += [r"\begin{tablenotes}\footnotesize",
                  r"\item Grid-selected hyperparameters (base $\rightarrow$ tuned):"]
        lines += hp_notes
        lines += [r"\end{tablenotes}"]
    lines += [r"\end{threeparttable}", r"\end{table}"]

    _track(_savetable(display_df, "T7_tuned_headline", "\n".join(lines)))
    return df


# ---------------------------------------------------------------------------
# T7b — Paired bootstrap significance (T4 protocol)
# ---------------------------------------------------------------------------
def _error_stack(root: Path, run_names: list[str], metric: str,
                 true_freq_week: np.ndarray, true_spend_week: np.ndarray) -> np.ndarray:
    true_freq = true_freq_week.sum(axis=1)
    true_spend = true_spend_week.sum(axis=1)
    true_clv = per_customer_clv(true_spend_week, WEEKLY_DISCOUNT_RATE)
    errors = []
    for run_name in run_names:
        with np.load(_arrays_path(root, run_name), allow_pickle=False) as d:
            stored_true_freq = np.asarray(d["per_week_true_freq"], dtype=np.float64)
            if not np.array_equal(stored_true_freq, true_freq_week):
                raise RuntimeError(f"Customer ordering mismatch for {run_name}.")
            pred_freq_week = np.asarray(d["per_week_pred_freq"], dtype=np.float64)
            if metric == "freq_rmse":
                err = (true_freq - pred_freq_week.sum(axis=1)) ** 2
            elif metric == "freq_mae":
                err = np.abs(true_freq - pred_freq_week.sum(axis=1))
            elif metric == "spend_mae_raw":
                pred_spend = np.asarray(d["per_week_pred_spend"], dtype=np.float64)
                err = np.abs(true_spend - pred_spend.sum(axis=1))
            elif metric == "clv_mae":
                pred_spend = np.asarray(d["per_week_pred_spend"], dtype=np.float64)
                err = np.abs(true_clv - per_customer_clv(pred_spend, WEEKLY_DISCOUNT_RATE))
            else:
                raise ValueError(f"Unsupported significance metric: {metric}")
        errors.append(np.asarray(err, dtype=np.float64))
    return np.stack(errors, axis=0)


def _run_names(model: str, config: str) -> tuple[Path, list[str]]:
    if model == "pareto_nbd":
        return FIXED, [f"pareto_nbd_{DATASET}"]
    root = FIXED if config == "fixed" else TUNED
    suffix = "final" if config == "fixed" else "tuned"
    return root, [f"{model}_{DATASET}_{suffix}_seed{seed}_sample" for seed in SEEDS]


# (model_a, config_a, model_b, config_b, metric) — pre-specified contrast set.
T7B_CONTRASTS = [
    ("lstm_joint", "tuned", "lstm_joint", "fixed", "freq_rmse"),
    ("lstm_joint", "tuned", "lstm_joint", "fixed", "spend_mae_raw"),
    ("lstm_joint", "tuned", "lstm_joint", "fixed", "clv_mae"),
    ("transformer_joint", "tuned", "transformer_joint", "fixed", "freq_rmse"),
    ("transformer_joint", "tuned", "transformer_joint", "fixed", "spend_mae_raw"),
    ("transformer_joint", "tuned", "transformer_joint", "fixed", "clv_mae"),
    ("transformer_joint", "tuned", "lstm_joint", "tuned", "spend_mae_raw"),
    ("transformer_joint", "tuned", "lstm_joint", "tuned", "clv_mae"),
    ("lstm_joint", "tuned", "pareto_nbd", "—", "clv_mae"),
    ("transformer_joint", "tuned", "pareto_nbd", "—", "clv_mae"),
]

SIG_METRIC_LABELS = {
    "freq_rmse": "Frequency RMSE",
    "spend_mae_raw": "Spend MAE",
    "clv_mae": "CLV MAE",
}


def make_table_t7b(true_freq_week: np.ndarray, true_spend_week: np.ndarray) -> pd.DataFrame:
    print("[T7b] Tuned-vs-fixed paired bootstrap (5,000 resamples, seed 42)...")
    rows = []
    for model_a, cfg_a, model_b, cfg_b, metric in T7B_CONTRASTS:
        root_a, names_a = _run_names(model_a, cfg_a)
        root_b, names_b = _run_names(model_b, cfg_b)
        stack_a = _error_stack(root_a, names_a, metric, true_freq_week, true_spend_week)
        stack_b = _error_stack(root_b, names_b, metric, true_freq_week, true_spend_week)
        res = _paired_bootstrap_seed_mean(
            stack_a, stack_b,
            n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED,
            is_squared=(metric == "freq_rmse"),
        )
        rows.append({
            "Model A": f"{ARCH_LABELS.get(model_a, 'Pareto/NBD + GG')} ({cfg_a})",
            "Model B": (f"{ARCH_LABELS[model_b]} ({cfg_b})"
                        if model_b != "pareto_nbd" else "Pareto/NBD + GG"),
            "Metric": SIG_METRIC_LABELS[metric],
            "Dataset": "Dunnhumby",
            "Delta": res["delta"],
            "CI_low": res["ci_low"],
            "CI_high": res["ci_high"],
            "p_value": res["p_value"],
            "N_customers": res["n_customers"],
        })
    df = pd.DataFrame(rows)
    df["p_holm"] = _holm_adjust(df["p_value"].to_numpy())
    df["sig_holm_p<0.05"] = df["p_holm"] < 0.05

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\begin{threeparttable}",
        r"\caption{Paired customer-bootstrap tests for the tuned configuration study "
        r"(5{,}000 resamples, bootstrap seed 42; Dunnhumby 80/22). Deep-model errors "
        r"are averaged across seeds 7, 42, and 2024 before resampling; the Pareto/NBD "
        r"benchmark is one deterministic fit. $\Delta$ = mean error of Model A $-$ "
        r"Model B; negative $\Delta$ means Model A is better. Two-sided $p$-values "
        r"are Holm-adjusted across the displayed contrasts, which were fixed before "
        r"the tuned holdout results were seen.}",
        r"\label{tab:tuned_significance}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Model A & Model B & Metric & $\Delta$ & CI$_{2.5}$ & CI$_{97.5}$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['Model A']} & {r['Model B']} & {r['Metric']} & "
            f"{r['Delta']:.4f} & {r['CI_low']:.4f} & {r['CI_high']:.4f} & "
            f"{_format_p(r['p_holm'], r['p_holm'] < 0.05)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{tablenotes}\footnotesize",
              r"\item[$^*$] Holm-adjusted $p < 0.05$. Confidence intervals are unadjusted.",
              r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    _track(_savetable(df, "T7b_tuned_significance", "\n".join(lines)))
    return df


# ---------------------------------------------------------------------------
# T8 — Tuned covariate ablation (mirror of T5)
# ---------------------------------------------------------------------------
T8_METRICS = ["freq_rmse", "freq_mape", "bias_pct", "spend_r2_log", "clv_spearman"]


def _ext3_seed_deltas(df: pd.DataFrame, architecture: str, variant: str, metric: str) -> np.ndarray:
    base = df[df["model"] == f"extension3_{architecture}_none"].set_index("seed")[metric].astype(float)
    changed = df[df["model"] == f"extension3_{architecture}_{variant}"].set_index("seed")[metric].astype(float)
    common = sorted(set(base.index) & set(changed.index))
    if {int(s) for s in common} != set(SEEDS):
        raise RuntimeError(f"Ext3 seed alignment failed for {architecture}/{variant}/{metric}.")
    return np.asarray([changed.loc[s] - base.loc[s] for s in common])


def make_table_t8(df_tuned: pd.DataFrame, df_fixed: pd.DataFrame) -> pd.DataFrame:
    print("[T8] Tuned covariate ablation (80/4, mirror of T5)...")
    rows = []
    for architecture in ("lstm", "transformer"):
        for variant in EXTENSION3_VARIANTS:
            model = f"extension3_{architecture}_{variant}"
            sub = df_tuned[df_tuned["model"] == model]
            row = {
                "Architecture": "LSTM" if architecture == "lstm" else "Transformer",
                "Covariates": EXTENSION3_LABELS[variant],
                "N_seeds": int(sub["seed"].notna().sum()),
            }
            for metric in T8_METRICS:
                vals = sub[metric].dropna().astype(float).to_numpy()
                row[metric] = float(vals.mean())
                row[f"{metric}_std"] = float(vals.std(ddof=1))
            row["dclv"] = (0.0 if variant == "none" else
                           float(_ext3_seed_deltas(df_tuned, architecture, variant,
                                                   "clv_spearman").mean()))
            # Fixed-run reference means (same variant, frozen thesis runs).
            fixed_sub = df_fixed[df_fixed["model"] == model]
            row["fixed_freq_mape"] = float(fixed_sub["freq_mape"].dropna().astype(float).mean())
            row["fixed_spend_r2_log"] = float(fixed_sub["spend_r2_log"].dropna().astype(float).mean())
            rows.append(row)
    df = pd.DataFrame(rows)

    display_rows = []
    for _, row in df.iterrows():
        disp = {"Architecture": row["Architecture"], "Covariates": row["Covariates"],
                "N_seeds": row["N_seeds"]}
        for metric in T8_METRICS:
            disp[metric] = f"{row[metric]:.3f} ± {row[f'{metric}_std']:.3f}"
        disp["dclv_vs_none"] = f"{row['dclv']:+.3f}"
        disp["fixed_freq_mape"] = f"{row['fixed_freq_mape']:.3f}"
        disp["fixed_spend_r2_log"] = f"{row['fixed_spend_r2_log']:.3f}"
        display_rows.append(disp)
    display_df = pd.DataFrame(display_rows)

    # Explicit "all covariates vs none" contrast per architecture (seed-paired).
    full_vs_none = {
        arch: {m: _ext3_seed_deltas(df_tuned, arch, "full", m) for m in
               ("freq_mape", "spend_r2_log", "clv_spearman")}
        for arch in ("lstm", "transformer")
    }

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{threeparttable}",
        r"\caption{Dunnhumby covariate ablation under TUNED hyperparameters "
        r"(80-week calibration, 4-week holdout; mirror of Table T5). All four "
        r"covariate variants use the grid-selected hyperparameters of the "
        r"full-covariate model, so rows differ only in the covariate set. Values "
        r"are mean $\pm$ SD over three seeds; $\Delta$CLV $\rho$ is the paired "
        r"seed-wise change relative to the no-covariate baseline.}",
        r"\label{tab:tuned_covariate_ablation}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Architecture & Covariates & Freq RMSE & Freq MAPE\,\% & Bias\,\% & Spend $R^2$ & "
        r"CLV $\rho$ & $\Delta$CLV $\rho$ \\",
        r"\midrule",
    ]
    prev_arch = None
    for _, row in df.iterrows():
        if prev_arch and row["Architecture"] != prev_arch:
            lines.append(r"\midrule")
        prev_arch = row["Architecture"]
        cells = [row["Architecture"], row["Covariates"]]
        for metric in T8_METRICS:
            cells.append(f"{row[metric]:.3f} $\\pm$ {row[f'{metric}_std']:.3f}")
        cells.append("--" if row["Covariates"] == EXTENSION3_LABELS["none"]
                     else f"{row['dclv']:+.3f}")
        lines.append(" & ".join(cells) + r" \\")
    notes = [r"\begin{tablenotes}\footnotesize",
             r"\item Full $-$ None (paired seed means): "]
    for arch, label in (("lstm", "LSTM"), ("transformer", "Transformer")):
        d = full_vs_none[arch]
        notes.append(
            rf"\item {label}: $\Delta$MAPE {d['freq_mape'].mean():+.3f}\,pp, "
            rf"$\Delta$Spend $R^2$ {d['spend_r2_log'].mean():+.3f}, "
            rf"$\Delta$CLV $\rho$ {d['clv_spearman'].mean():+.3f}"
        )
    notes.append(r"\end{tablenotes}")
    lines += [r"\bottomrule", r"\end{tabular}"] + notes + [r"\end{threeparttable}", r"\end{table}"]
    _track(_savetable(display_df, "T8_tuned_ablation", "\n".join(lines)))
    return df


# ---------------------------------------------------------------------------
# T9 — Tuned SHAP attribution (mirror of T6, with fixed-run reference)
# ---------------------------------------------------------------------------
def _load_tuned_shap() -> pd.DataFrame:
    if not TUNED_SHAP_CSV.exists():
        raise FileNotFoundError(f"Missing tuned SHAP summary: {TUNED_SHAP_CSV}")
    df = pd.read_csv(TUNED_SHAP_CSV)
    if set(df["n_households"]) != {701} or set(df["n_integration_samples"]) != {128}:
        raise RuntimeError("Tuned SHAP summary must be the 701-household, 128-sample run.")
    if set(df["cohort"]) != {"observed_demographics"} or set(df["n_seeds"]) != {3}:
        raise RuntimeError("Tuned SHAP summary must be the 3-seed observed-demographics run.")
    return df


def make_table_t9() -> pd.DataFrame:
    print("[T9] Tuned dual-head SHAP attribution (mirror of T6)...")
    df = _load_tuned_shap()
    fixed = pd.read_csv(FIXED_SHAP_CSV)
    keys = ["architecture", "head", "feature"]
    fixed_ref = fixed.set_index(keys)["relative_importance_pct"]
    df = df.copy()
    df["fixed_relative_importance_pct"] = [
        float(fixed_ref.get(tuple(row[k] for k in keys), float("nan")))
        for _, row in df.iterrows()
    ]
    additivity = pd.read_csv(TUNED_SHAP_ADDITIVITY_CSV)
    add_low = 100 * additivity["normalized_additivity_error"].min()
    add_high = 100 * additivity["normalized_additivity_error"].max()

    df["Architecture"] = df["architecture"].map({"lstm": "LSTM", "transformer": "Transformer"})
    df["Head"] = df["head"].map({"freq": "Frequency", "spend": "Conditional log1p-spend"})
    df["Feature type"] = df["feature_type"].map({"static": "Static", "dynamic": "Dynamic"})
    df["Feature"] = df["feature"].map(bts.SHAP_FEATURE_LABELS).fillna(df["feature"])

    display = df[[
        "Architecture", "Head", "Feature type", "Feature",
        "mean_abs_shap", "seed_sd",
        "relative_importance_pct", "fixed_relative_importance_pct",
    ]].rename(columns={
        "mean_abs_shap": "Mean absolute SHAP",
        "seed_sd": "Seed SD",
        "relative_importance_pct": "Tuned rel. (%)",
        "fixed_relative_importance_pct": "Fixed rel. (%)",
    }).sort_values(["Architecture", "Head", "Tuned rel. (%)"],
                   ascending=[True, True, False])

    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Extension 3 covariate attribution for the TUNED full-covariate "
        r"LSTM and Transformer on Dunnhumby (mirror of Table T6; identical 100 "
        r"background / 701 explained households and 128-sample expected-gradient "
        r"budget). The final column repeats the fixed-configuration relative "
        r"importance from Table T6 for direct comparison. Normalized additivity "
        f"residuals span {add_low:.1f}--{add_high:.1f}\\%.}}",
        r"\label{tab:tuned_shap_attribution}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Architecture & Head & Type & Feature & Mean $|\mathrm{SHAP}|$ & Seed SD & "
        r"Tuned rel.\,\% & Fixed rel.\,\% \\",
        r"\midrule",
    ]
    prev_group = None
    for _, row in display.iterrows():
        group = (row["Architecture"], row["Head"])
        if prev_group and group != prev_group:
            lines.append(r"\midrule")
        prev_group = group
        lines.append(
            f"{row['Architecture']} & {row['Head']} & {row['Feature type']} & "
            f"{row['Feature']} & {row['Mean absolute SHAP']:.3f} & {row['Seed SD']:.3f} & "
            f"{row['Tuned rel. (%)']:.0f} & {row['Fixed rel. (%)']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    _track(_savetable(display, "T9_tuned_shap", "\n".join(lines)))
    return display


# ---------------------------------------------------------------------------
# F10 — Tuning effect dumbbells
# ---------------------------------------------------------------------------
F10_PANELS = [
    ("clv_spearman", r"CLV Spearman $\rho$", "higher is better"),
    ("spend_r2_log", r"Spend $R^2$ (log-space)", "higher is better"),
    ("freq_mape", "Frequency MAPE %", "lower is better"),
]


def fig_tuning_effect(df_fixed: pd.DataFrame, df_tuned: pd.DataFrame) -> None:
    print("[F10] Tuning effect figure...")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (metric, title, direction) in zip(axes, F10_PANELS):
        y_positions = []
        y_labels = []
        for i, arch in enumerate(MAIN_ARCHS):
            y = len(MAIN_ARCHS) - i
            fm, fs, _ = _mean_std(df_fixed, arch, metric)
            tm, ts, _ = _mean_std(df_tuned, arch, metric)
            color = MODEL_COLORS[arch]
            ax.plot([fm, tm], [y, y], color=color, lw=2, zorder=2, alpha=0.6)
            ax.errorbar([fm], [y], xerr=[fs], fmt="o", mfc="white", mec=color,
                        ecolor=color, color=color, ms=8, zorder=3, label=None)
            ax.errorbar([tm], [y], xerr=[ts], fmt="o", color=color, ecolor=color,
                        ms=8, zorder=4)
            y_positions.append(y)
            y_labels.append(ARCH_LABELS[arch])
        pm, _, _ = _mean_std(df_fixed, "pareto_nbd", metric)
        if np.isfinite(pm):
            ax.axvline(pm, color=MODEL_COLORS["pareto_nbd"], ls="--", lw=1.2, zorder=1)
            ax.text(pm, 0.45, " Pareto/NBD", color=MODEL_COLORS["pareto_nbd"],
                    fontsize=8, ha="left", va="bottom")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_ylim(0.4, len(MAIN_ARCHS) + 0.6)
        ax.set_title(f"{title}\n({direction})", fontsize=10)
    handles = [
        plt.Line2D([], [], marker="o", mfc="white", mec="black", color="black",
                   ls="none", label="Fixed configuration"),
        plt.Line2D([], [], marker="o", color="black", ls="none", label="Tuned configuration"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Effect of hyperparameter tuning — Dunnhumby 80/22 (mean ± SD over 3 seeds)",
                 y=1.04, fontsize=11)
    fig.tight_layout()
    _track(_savefig(fig, "F10_tuning_effect"))


# ---------------------------------------------------------------------------
# F11 — Tuned covariate ablation (fixed-run reference markers)
# ---------------------------------------------------------------------------
def fig_tuned_ablation(df_tuned: pd.DataFrame, df_fixed: pd.DataFrame) -> None:
    print("[F11] Tuned ablation figure...")
    panels = [("freq_mape", "Frequency MAPE % (lower is better)"),
              ("spend_r2_log", r"Spend $R^2$ (higher is better)")]
    variant_colors = {"none": "#999999", **bts.EXTENSION3_COLORS}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col")
    for col, (metric, title) in enumerate(panels):
        for row, (arch, arch_label) in enumerate(
            (("lstm", "LSTM"), ("transformer", "Transformer"))
        ):
            ax = axes[row][col]
            xs = np.arange(len(EXTENSION3_VARIANTS))
            for x, variant in zip(xs, EXTENSION3_VARIANTS):
                model = f"extension3_{arch}_{variant}"
                sub = df_tuned[df_tuned["model"] == model][metric].dropna().astype(float)
                ax.bar(x, sub.mean(), yerr=sub.std(ddof=1), width=0.62,
                       color=variant_colors[variant], alpha=0.9, capsize=3,
                       error_kw={"lw": 1})
                fixed_mean = (
                    df_fixed[df_fixed["model"] == model][metric].dropna().astype(float).mean()
                )
                ax.plot([x], [fixed_mean], marker="D", ms=6, color="black", zorder=5)
            ax.set_xticks(xs)
            ax.set_xticklabels([EXTENSION3_LABELS[v] for v in EXTENSION3_VARIANTS])
            ax.set_title(f"{arch_label} — {title}", fontsize=9.5)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", label="Tuned (bars, mean ± SD)"),
        plt.Line2D([], [], marker="D", color="black", ls="none",
                   label="Fixed configuration (mean)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Covariate ablation under tuned hyperparameters — Dunnhumby 80/4",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    _track(_savefig(fig, "F11_tuned_ablation"))


# ---------------------------------------------------------------------------
# Addendum + manifest
# ---------------------------------------------------------------------------
def write_addendum(df_fixed: pd.DataFrame, df_tuned: pd.DataFrame,
                   sig_df: pd.DataFrame, hps: dict[str, list[str]]) -> None:
    print("[Addendum] TUNED_RESULTS_ADDENDUM.md ...")
    lines = [
        "# Tuned-performance addendum (supplementary evidence set)",
        "",
        "Generated by `build_tuned_results.py` from `results/final_kaggle_tuned/` "
        "(kaggle_tuned_runner.ipynb output). This is a SEPARATE evidence set from the "
        "frozen fixed-parameter thesis results; nothing in T1–T6 / F1–F9 changes.",
        "",
        "## Protocol (what to state in the thesis)",
        "",
        "- Exhaustive grid HPO per architecture and protocol (LSTM 18 trials: "
        "lr x hidden_size x dropout; Transformer 15 trials: lr x (d_model, n_heads)); "
        "single seed 42 per trial at reduced fidelity (max_epochs 80, 20 scenarios).",
        "- HPO trials were scored on a validation window carved from the last 20% of "
        "calibration weeks (`--hpo-val-pct 0.2`); the true holdout was never seen "
        "during hyperparameter selection.",
        "- Winners retrained on the full calibration window, 3 seeds (42/7/2024), "
        "sample-mode inference with 30 scenarios — identical to the fixed protocol.",
        "- The tuned configuration also lifts the epoch cap from 150 to 250 because "
        "every fixed Dunnhumby run terminated at the cap with early stopping never "
        "firing; patience-20 early stopping on the calibration-internal validation "
        "split decides the realised length. Part of any improvement may therefore "
        "reflect completed convergence rather than hyperparameter choice — state this.",
        "- Extension 3: the full-covariate winner's hyperparameters are transferred to "
        "all four covariate variants, so the tuned ablation compares covariate sets "
        "under identical hyperparameters.",
        "- SHAP: identical 100 background / 701 explained households (committed sample "
        "manifests) and identical 128-sample expected-gradient budget as Table T6.",
        "",
        "## Grid-selected hyperparameters (base -> tuned)",
        "",
    ]
    for stem in sorted(hps):
        joined = "; ".join(hps[stem]) if hps[stem] else "winner = fixed defaults"
        lines.append(f"- `{stem}`: {joined}")
    lines += ["", "## Headline deltas (tuned − fixed, seed means, rescored)", ""]
    for arch in MAIN_ARCHS:
        deltas = []
        for metric in HEADLINE_METRICS:
            fm, _, _ = _mean_std(df_fixed, arch, metric)
            tm, _, _ = _mean_std(df_tuned, arch, metric)
            deltas.append(f"{metric} {tm - fm:+.3f}")
        lines.append(f"- {ARCH_LABELS[arch]}: " + ", ".join(deltas))
    lines += [
        "",
        "## Significance (T7b, Holm-adjusted paired bootstrap)",
        "",
    ]
    for _, r in sig_df.iterrows():
        star = " *" if r["sig_holm_p<0.05"] else ""
        lines.append(
            f"- {r['Model A']} vs {r['Model B']} — {r['Metric']}: "
            f"Δ={r['Delta']:.4f} [{r['CI_low']:.4f}, {r['CI_high']:.4f}], "
            f"p_holm={r['p_holm']:.4f}{star}"
        )
    lines += [
        "",
        "## Caveats to carry into the text",
        "",
        "- Post-hoc supplementary study answering the §5.4 future-work item; the "
        "architectural comparison chapter remains based on the fixed configuration.",
        "- Grid budgets differ slightly by architecture (18 vs 15 trials) — note as a "
        "limitation, mirroring the fixed-budget discussion.",
        "- Three seeds: directional screening, not high-power inference (same reading "
        "as §3.5.3). A null or negative tuning effect is a reportable finding.",
        "- SHAP values remain approximate additive decompositions (see additivity "
        "residuals in T9) and are conditional model attributions, not causal effects.",
        "",
        "## Files",
        "",
    ]
    for p in sorted(set(_EMITTED)):
        lines.append(f"- `{p}`")
    path = OUT / "TUNED_RESULTS_ADDENDUM.md"
    path.write_text("\n".join(lines) + "\n")
    _track([path])
    print(f"  Saved {path}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tuned_manifest() -> None:
    manifest = {
        "purpose": "Tuned-performance supplementary evidence set (Dunnhumby)",
        "source_results_dir": str(TUNED),
        "fixed_baseline_dir": str(FIXED),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "adjustment": "holm"},
        "output_files": [
            {"path": str(p), "sha256": _sha256(p)} for p in sorted(set(_EMITTED))
        ],
    }
    path = OUT / "TUNED_ARTIFACT_MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Tuned results: {TUNED}  |  Fixed baselines: {FIXED}\nOutput: {OUT}\n")
    true_freq_week, true_spend_week = _canonical_targets()
    df_fixed, df_tuned = load_sessions()
    hps = winning_hps()

    make_table_t7(df_fixed, df_tuned, hps)
    sig_df = make_table_t7b(true_freq_week, true_spend_week)
    make_table_t8(df_tuned, df_fixed)
    make_table_t9()
    fig_tuning_effect(df_fixed, df_tuned)
    fig_tuned_ablation(df_tuned, df_fixed)
    write_addendum(df_fixed, df_tuned, sig_df, hps)
    write_tuned_manifest()
    print("\nDone. Review TUNED_RESULTS_ADDENDUM.md before adapting the thesis text.")


if __name__ == "__main__":
    main()
