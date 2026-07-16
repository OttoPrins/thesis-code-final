"""
Hyperparameter search + sensitivity sweeps for CLV models.

Two modes (selected with --mode):

* hpo (default) — Optuna search over a bounded space. The default sampler is an
  exhaustive GridSampler (reproducible, no stochasticity); a TPE sampler is also
  available (--sampler tpe) for a wider, randomised search. The winning config is
  dumped back to experiments/configs/<run_name>_hpo.yaml for the multi-seed eval.
  The primary thesis results use fixed (untuned) configs; HPO winners feed the
  separate tuned-performance analysis (kaggle_tuned_runner.ipynb).

* sensitivity — a plain one-at-a-time sweep (NO Optuna) over a single dotted config
  key (--param) across a list of --values, writing a summary CSV. Used to fill the
  Category-C governance TODOs (warmup_epochs, max_grad_norm, ...).

Design notes
------------
* Reuses train.run_experiment(config, write_outputs=False) so trials/sweeps do not
  pollute results/tables. Hyperparameters are only ever set in the in-memory config
  (never hardcoded in Python) — the winner is dumped back to a YAML config.
* No mid-training Optuna pruning (run_experiment is monolithic and returns only final
  metrics) — trials are bounded by --max-epochs and --n-trials instead.
* Single seed (42) per trial; the final evaluation still uses all 3 seeds.

Methodological note on HPO evaluation
--------------------------------------
By default (--hpo-val-pct 0) HPO trials are scored on the same holdout period that is
used for final reporting; there is no separate validation fold. The grid is
intentionally small (18 LSTM trials, 15 Transformer trials per dataset) so the risk of
data-driven overfitting is bounded; the 3-seed final evaluation provides an additional
variance check. With --hpo-val-pct > 0 the last fraction of *calibration* weeks is
carved off as an HPO-only validation window instead, so the true holdout is never seen
during hyperparameter selection — this is the protocol used by the tuned-performance
workflow (kaggle_tuned_runner.ipynb).

LSTM hidden_size (64 / 128 / 256) is included in the search even for the Base LSTM
(Stage 1 replication). Valendin et al. use 128, which serves as the reference point,
but finding a better hidden_size on the same dataset is a valid empirical result and
does not invalidate the replication claim as long as the architecture is otherwise
identical (single LSTM layer, return_sequences=True, same input embeddings).

Examples:
    # Exhaustive grid HPO (default) on the joint LSTM
    python tune.py --config experiments/configs_final/lstm_joint_cdnow_final.yaml \
        --sampler grid --max-epochs 80 --n-scenarios 20

    # Randomised TPE search
    python tune.py --config experiments/configs/lstm_base_cdnow_v3.yaml \
        --sampler tpe --n-trials 40 --max-epochs 120 --n-scenarios 50

    # One-at-a-time sensitivity sweep
    python tune.py --mode sensitivity \
        --config experiments/configs/lstm_joint_cdnow_final_hpo.yaml \
        --param loss.warmup_epochs --values 0 5 10 20
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
from pathlib import Path

import optuna
import yaml

from src.utils.config import apply_kaggle_overrides, load_config
from train import run_experiment


# Metric columns reported by the sensitivity sweep (all produced by run_experiment).
SENSITIVITY_METRICS = [
    "freq_mape",
    "freq_rmse",
    "bias_pct",
    "spend_mae_log",
    "spend_r2_log",
    "clv_spearman",
]


def build_grid(model_type: str) -> dict:
    """Return the GridSampler search_space dict for a model family.

    The grids are the pre-specified HPO spaces for the thesis. Transformer
    (d_model, n_heads) pairs are encoded as "<d_model>_<n_heads>" strings so the
    GridSampler never proposes an invalid head/dim combination (e.g. 64/8 with
    d_model not divisible by n_heads is simply not in the list).

    LSTM hidden_size {64, 128, 256} is searched for ALL LSTM configs (base and joint).
    Valendin et al. (2022) use 128; that is the reference point but not a constraint.
    Finding a better hidden_size is a valid empirical result as long as the rest of the
    architecture (single layer, return_sequences, same embeddings) matches exactly.
    """
    if model_type == "lstm":
        # 3 × 3 × 2 = 18 combinations. dropout {0.0, 0.1} gives the search a
        # regularisation axis; 0.0 is the Valendin default, so the fixed
        # configuration is always a member of the grid.
        return {
            "lr": [3e-4, 1e-3, 3e-3],
            "hidden_size": [64, 128, 256],
            "dropout": [0.0, 0.1],
        }
    if model_type == "transformer":
        # 3 × 5 = 15 combinations. "arch" = "<d_model>_<n_heads>" (all valid pairs).
        return {
            "lr": [1e-4, 5e-4, 1e-3],
            "arch": ["64_4", "128_4", "128_8", "256_4", "256_8"],
        }
    raise ValueError(f"build_grid: unsupported model.type={model_type!r}")


def _grid_size(model_type: str) -> int:
    n = 1
    for values in build_grid(model_type).values():
        n *= len(values)
    return n


def _suggest(trial, config: dict, sampler_name: str) -> None:
    """Mutate `config` in place with this trial's sampled hyperparameters.

    Search space is bounded and respects the hard constraints (single LSTM
    layer; shallow Transformer; Time2Vec+sinusoidal only — none of those keys
    are searched). Keys are only added when relevant to the model family.
    """
    tr = config["training"]
    md = config["model"]
    ls = config.setdefault("loss", {})

    is_lstm = md.get("type") == "lstm"
    joint = md.get("joint", False)
    model_type = md.get("type", "lstm")

    if sampler_name == "grid":
        # Grid mode: non-grid params fixed to base config. We ONLY suggest the
        # parameters present in build_grid(); everything else (weight_decay,
        # d_ff, dense_units, n_layers, all Kendall loss params, all
        # scheduled-sampling params, max_grad_norm) keeps its base-config value.
        grid = build_grid(model_type)
        tr["lr"] = trial.suggest_categorical("lr", grid["lr"])
        if is_lstm:
            md["hidden_size"] = trial.suggest_categorical("hidden_size", grid["hidden_size"])
            md["dropout"] = trial.suggest_categorical("dropout", grid["dropout"])
        else:  # transformer — unpack the "<d_model>_<n_heads>" arch string
            arch = trial.suggest_categorical("arch", grid["arch"])
            d_model, n_heads = int(arch.split("_")[0]), int(arch.split("_")[1])
            md["d_model"] = d_model
            md["n_heads"] = n_heads
        return

    # ── TPE mode: wider randomised search ────────────────────────────────────
    tr["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    tr["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    md["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)

    if is_lstm:
        # Scheduled-sampling dose — the dominant lever for autoregressive bias.
        ss = tr.setdefault("scheduled_sampling", {})
        ss["enabled"] = True
        ss["max_prob"] = trial.suggest_float("ss_max_prob", 0.02, 0.40)
        ss["start_epoch"] = trial.suggest_int("ss_start_epoch", 5, 40)
        ss["schedule"] = trial.suggest_categorical("ss_schedule", ["linear", "inv_sigmoid"])
        # hidden_size is a free architectural parameter for ALL LSTM model types.
        md["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 192, 256])
        if sampler_name == "tpe":
            if trial.suggest_categorical("grad_norm_disabled", [True, False]):
                tr["max_grad_norm"] = 0.0
            else:
                tr["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 5.0, log=True)
    else:  # transformer — shallow, constraint-respecting
        d_model = trial.suggest_categorical("d_model", [64, 128])
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()  # invalid head/dim combo
        md["d_model"] = d_model
        md["n_heads"] = n_heads
        md["d_ff"] = trial.suggest_categorical("d_ff", [256, 512])
        md["n_layers"] = trial.suggest_int("n_layers", 2, 3)
        if sampler_name == "tpe":
            if trial.suggest_categorical("grad_norm_disabled", [True, False]):
                tr["max_grad_norm"] = 0.0
            else:
                tr["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.1, 5.0, log=True)

    if joint:
        ls["spend_logvar_max"] = trial.suggest_categorical("spend_logvar_max", [1.0, 2.0, 3.0])
        ls["warmup_epochs"] = trial.suggest_int("kendall_warmup", 3, 12)
        ls["spend_loss_normalize"] = trial.suggest_categorical(
            "spend_loss_normalize", [True, False]
        )
        if sampler_name == "tpe":
            ls["spend_loss_scale"] = trial.suggest_categorical(
                "spend_loss_scale", [0.01, 0.05, 0.1, 0.2]
            )


def _objective_value(metrics: dict, joint: bool) -> float:
    """Lower is better. Balances replication fidelity (cohort bias), the spend
    contribution, and CLV — and guards the degenerate low-bias/no-skill trap.

    base : 0.60·|bias|/100 + 0.40·(1−gini)
    joint: 0.40·|bias|/100 + 0.20·(1−gini)
           + 0.20·(1−clip(spend_r2_log,−1,1)) + 0.20·(1−clip(clv_spearman,0,1))
    Invalid runs are heavily penalised so the search avoids them.
    """
    if not metrics.get("run_valid", True):
        return 10.0

    def _clip(x, lo, hi):
        try:
            return max(lo, min(hi, float(x)))
        except (TypeError, ValueError):
            return lo

    bias = abs(float(metrics.get("bias_pct", 100.0))) / 100.0
    gini = _clip(metrics.get("freq_normalized_gini", 0.0), 0.0, 1.0)

    if not joint:
        return 0.60 * bias + 0.40 * (1.0 - gini)

    spend_r2 = _clip(metrics.get("spend_r2_log", -1.0), -1.0, 1.0)
    clv_sp = _clip(metrics.get("clv_spearman", 0.0), 0.0, 1.0)
    return (
        0.40 * bias
        + 0.20 * (1.0 - gini)
        + 0.20 * (1.0 - spend_r2)
        + 0.20 * (1.0 - clv_sp)
    )


def _cast_like(existing, raw: str):
    """Cast a string CLI value to the type of the existing config value."""
    if isinstance(existing, bool):
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "t")
    if isinstance(existing, int) and not isinstance(existing, bool):
        # Allow "0.0" -> 0 only when it is integral; otherwise fall back to float.
        try:
            return int(raw)
        except ValueError:
            return float(raw)
    if isinstance(existing, float):
        return float(raw)
    if existing is None:
        # Best-effort numeric inference for keys absent from the base config.
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return type(existing)(raw)


def _set_dotted(config: dict, dotted: str, raw: str):
    """Traverse a dotted key (e.g. 'loss.warmup_epochs'), set it, casting to the
    existing value's type. Returns the casted value actually written."""
    parts = dotted.split(".")
    node = config
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    leaf = parts[-1]
    value = _cast_like(node.get(leaf), raw)
    node[leaf] = value
    return value


def run_sensitivity(args, base_config: dict, base_run: str, on_kaggle: bool) -> None:
    """One-at-a-time sweep over a single dotted config key. No Optuna."""
    if not args.param or not args.values:
        raise SystemExit("sensitivity mode requires --param <dotted.key> and --values <...>.")

    param_slug = args.param.split(".")[-1]
    results_dir = base_config.get("output", {}).get("results_dir", "results")
    out_dir = Path(results_dir) / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{param_slug}_sweep.csv"

    print(
        f"[sensitivity] sweeping {args.param} over {args.values} "
        f"(max_epochs={args.max_epochs}, n_scenarios={args.n_scenarios})"
    )

    rows = []
    for raw in args.values:
        cfg = copy.deepcopy(base_config)
        cfg["training"]["seed"] = args.seed
        cfg["training"]["epochs"] = args.max_epochs
        cfg["training"]["max_epochs"] = args.max_epochs
        cfg.setdefault("inference", {})["mode"] = "sample"
        cfg["inference"]["n_scenarios"] = args.n_scenarios
        value = _set_dotted(cfg, args.param, raw)
        value_slug = str(value).replace(".", "p").replace("-", "m").replace(" ", "_")
        cfg.setdefault("output", {})["run_name"] = (
            f"{base_run}_sens_{param_slug}_{value_slug}"
        )
        if on_kaggle:
            apply_kaggle_overrides(cfg, args.kaggle_data_root)

        print(f"\n[sensitivity] {args.param} = {value!r}  (run_name={cfg['output']['run_name']})")
        metrics = run_experiment(
            cfg, config_path=f"{args.config}#sens:{args.param}={value}", write_outputs=False
        )
        row = {"param_value": value, "run_valid": bool(metrics.get("run_valid", False))}
        for m in SENSITIVITY_METRICS:
            row[m] = metrics.get(m, float("nan"))
        rows.append(row)

    columns = ["param_value", *SENSITIVITY_METRICS, "run_valid"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # ── Pretty ASCII table to stdout ─────────────────────────────────────────
    widths = {c: max(len(c), *(len(_fmt(r[c])) for r in rows)) for c in columns}
    header = " | ".join(c.ljust(widths[c]) for c in columns)
    sep = "-+-".join("-" * widths[c] for c in columns)
    print(f"\n=== Sensitivity sweep: {args.param} ===")
    print(header)
    print(sep)
    for r in rows:
        print(" | ".join(_fmt(r[c]).ljust(widths[c]) for c in columns))
    print(f"\nSaved: {out_path}")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    # ── T4x2 DUAL-GPU PATTERN (launch from kaggle_runner.ipynb): ──────────────
    # DB=/kaggle/working/hpo_{study_name}.db
    # CUDA_VISIBLE_DEVICES=0 python tune.py --config <cfg> \
    #     --sampler grid --storage sqlite:///$DB \
    #     --study-name <name> --kaggle &
    # CUDA_VISIBLE_DEVICES=1 python tune.py --config <cfg> \
    #     --sampler grid --storage sqlite:///$DB \
    #     --study-name <name> --kaggle &
    # wait
    #
    # Both workers share the study; GridSampler distributes trials
    # without duplication via Optuna's atomic trial claiming.
    # load_if_exists=True is already set when --storage is provided.
    p = argparse.ArgumentParser(
        description="Grid/TPE HPO and one-at-a-time sensitivity sweeps for CLV configs."
    )
    p.add_argument("--config", required=True, help="Base YAML to tune / sweep.")
    p.add_argument("--mode", choices=["hpo", "sensitivity"], default="hpo",
                   help="hpo = Optuna search (default); sensitivity = one-at-a-time sweep.")
    p.add_argument("--sampler", choices=["grid", "tpe"], default="grid",
                   help="hpo sampler: grid (exhaustive, default) or tpe (randomised).")
    p.add_argument("--n-trials", type=int, default=None,
                   help="HPO trials. Default: grid size (grid) or 40 (tpe).")
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--n-scenarios", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (e.g. sqlite:///hpo.db) for resumable studies.")
    p.add_argument("--timeout", type=int, default=None, help="Wall-clock seconds cap.")
    p.add_argument("--out-config", default=None,
                   help="Where to write the winning config (default: <base>_hpo.yaml).")
    p.add_argument("--param", default=None,
                   help="sensitivity mode: dotted config key to sweep (e.g. loss.warmup_epochs).")
    p.add_argument("--values", nargs="+", default=None,
                   help="sensitivity mode: values to sweep for --param.")
    p.add_argument("--kaggle", action="store_true",
                   help="Apply Kaggle raw_dir/results_dir overrides (or set KAGGLE_ENV=1).")
    p.add_argument("--kaggle-data-root", default=None,
                   help="Override the /kaggle/input parent dir for mounted datasets.")
    p.add_argument(
        "--hpo-val-pct", type=float, default=0.0,
        help=(
            "Fraction of calibration weeks to hold back as an HPO validation window "
            "(0.0 = default, scores trials on the true holdout). When > 0, the last "
            "hpo_val_pct of calibration is used as the HPO objective target and the "
            "true holdout is never seen during HPO. Callers should append '_valNN' to "
            "the study name so holdout-scored .db files are never mixed with "
            "val-fold-scored trials (kaggle_tuned_runner.ipynb does this)."
        ),
    )
    args = p.parse_args()

    print(
        "[tune.py] Primary thesis results use fixed configs; HPO winners feed the "
        "tuned-performance analysis. See module docstring for leakage discussion."
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base_config = load_config(args.config)
    base_run = base_config.get("output", {}).get("run_name", Path(args.config).stem)
    joint = base_config.get("model", {}).get("joint", False)
    model_type = base_config.get("model", {}).get("type", "lstm")
    sampler_name = args.sampler

    on_kaggle = args.kaggle or os.environ.get("KAGGLE_ENV", "0") == "1"

    # ── Sensitivity mode: plain loop, no Optuna ──────────────────────────────
    if args.mode == "sensitivity":
        run_sensitivity(args, base_config, base_run, on_kaggle)
        return

    # ── HPO mode ─────────────────────────────────────────────────────────────
    study_name = args.study_name or f"hpo_{base_run}"

    # Resolve n_trials: exhaust the grid when the user didn't ask for a count.
    if args.n_trials is not None:
        n_trials = args.n_trials
    elif sampler_name == "grid":
        n_trials = _grid_size(model_type)
    else:
        n_trials = 40

    def objective(trial) -> float:
        cfg = copy.deepcopy(base_config)
        cfg["training"]["seed"] = args.seed
        cfg["training"]["epochs"] = args.max_epochs
        cfg["training"]["max_epochs"] = args.max_epochs
        cfg.setdefault("inference", {})["mode"] = "sample"
        cfg["inference"]["n_scenarios"] = args.n_scenarios
        cfg.setdefault("output", {})["run_name"] = f"{base_run}_hpo_trial{trial.number}"
        _suggest(trial, cfg, sampler_name)
        # Validation-fold mode: carve the last hpo_val_pct of calibration as a
        # proxy evaluation window so the true holdout is never seen during HPO.
        # calibration_weeks shrinks; the carved tail acts as a short holdout for
        # scoring only — the final 3-seed eval still uses the full calibration +
        # original holdout.  Default (hpo_val_pct=0) leaves config unchanged.
        # (TemporalSplitter anchors at week 0, so new_cal + val window together
        # span exactly the original calibration weeks.)
        if args.hpo_val_pct > 0.0:
            cal_key = "calibration_weeks" if "calibration_weeks" in cfg["dataset"] \
                else "lookback_weeks"
            orig_cal = cfg["dataset"][cal_key]
            new_cal = max(10, round(orig_cal * (1.0 - args.hpo_val_pct)))
            cfg["dataset"][cal_key] = new_cal
            cfg["dataset"]["holdout_weeks"] = orig_cal - new_cal
        # Remap raw_dir/results_dir for Kaggle's read-only input layout so the
        # data pipeline finds the mounted datasets (results aren't written —
        # write_outputs=False — but the pipeline still reads raw_dir).
        if on_kaggle:
            apply_kaggle_overrides(cfg, args.kaggle_data_root)
        metrics = run_experiment(
            cfg, config_path=f"{args.config}#trial{trial.number}", write_outputs=False
        )
        val = _objective_value(metrics, joint)
        trial.set_user_attr("bias_pct", float(metrics.get("bias_pct", float("nan"))))
        trial.set_user_attr("freq_gini", float(metrics.get("freq_normalized_gini", float("nan"))))
        if joint:
            trial.set_user_attr("spend_r2_log", float(metrics.get("spend_r2_log", float("nan"))))
            trial.set_user_attr("clv_spearman", float(metrics.get("clv_spearman", float("nan"))))
        return val

    if sampler_name == "grid":
        sampler = optuna.samplers.GridSampler(build_grid(model_type))
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_trials, timeout=args.timeout)

    best = study.best_trial
    print(f"\n=== Best trial #{best.number} (objective={best.value:.4f}) ===")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    for k, v in best.user_attrs.items():
        print(f"  [metric] {k}: {v}")

    # Apply the winning params to a fresh copy of the base config and dump it.
    # Guard against a race when two workers share a study: only write if absent.
    winner = copy.deepcopy(base_config)
    winner["training"]["seed"] = args.seed
    # Re-run _suggest using a FixedTrial so the same mapping logic writes the winner.
    _suggest(optuna.trial.FixedTrial(best.params), winner, sampler_name)
    winner.setdefault("output", {})["run_name"] = f"{base_run}_hpo"
    out_path = Path(args.out_config) if args.out_config else \
        Path("experiments/configs") / f"{base_run}_hpo.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"\nWinning config already present (another worker wrote it): {out_path}")
    else:
        if args.hpo_val_pct > 0.0:
            scoring_note = (
                f"# HPO note: trials were scored on an HPO validation window carved from\n"
                f"# the last {args.hpo_val_pct:.0%} of calibration weeks (--hpo-val-pct); the\n"
                f"# true holdout was never seen during hyperparameter selection.\n"
            )
        else:
            scoring_note = (
                f"# HPO note: trials were scored on the holdout period; grid size={n_trials} bounds\n"
                f"# data-driven overfitting risk. See tune.py docstring for full discussion.\n"
            )
        out_path.write_text(
            f"# Auto-generated by tune.py from {args.config}\n"
            f"# HPO-selected config (sampler={sampler_name}).\n"
            + scoring_note
            + f"# Best Optuna objective={best.value:.4f} "
            f"(trial #{best.number}, {n_trials} trials, max_epochs={args.max_epochs})\n"
            + yaml.safe_dump(winner, sort_keys=False, default_flow_style=False)
        )
        print(f"\nWinning config written: {out_path}")
    print("Next: python run_seeds.py \\")
    print("  --config_dir experiments/configs \\")
    print(f"  --configs {base_run}_hpo \\")
    print("  --seeds 42 7 2024 --modes sample [--kaggle]")


if __name__ == "__main__":
    main()
