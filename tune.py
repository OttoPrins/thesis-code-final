"""
Exploratory Optuna hyperparameter search for CLV robustness variants.

This is deliberately not part of the headline empirical protocol. Use it only
as exploratory evidence or as a bounded robustness exercise that is explicitly
labelled as such in the thesis. The final protocol should prefer the proposal
models plus the pre-specified minimal stability repairs in train.py.

Design notes
------------
* Reuses train.run_experiment(config, write_outputs=False) so trials do not
  pollute results/. Hyperparameters are only ever set in the in-memory config
  (never hardcoded in Python) — the winner is dumped back to a YAML config.
* Evaluated in `sample` mode with a reduced n_scenarios for trial speed. A
  winning config is still exploratory unless it is promoted into a separately
  documented robustness manifest.
* No mid-training Optuna pruning (run_experiment is monolithic and returns only
  final metrics) — trials are bounded by --max-epochs and --n-trials instead.
* Single seed (42) per trial; the final evaluation still uses all 3 seeds.

Example:
    python tune.py --config experiments/configs/lstm_base_cdnow_v3.yaml \
        --n-trials 40 --max-epochs 120 --n-scenarios 50
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import optuna
import yaml

from src.utils.config import apply_kaggle_overrides, load_config
from train import run_experiment


def _suggest(trial, config: dict) -> None:
    """Mutate `config` in place with this trial's sampled hyperparameters.

    Search space is bounded and respects the hard constraints (single LSTM
    layer; shallow Transformer; Time2Vec+sinusoidal only — none of those keys
    are searched). Keys are only added when relevant to the model family.
    """
    tr = config["training"]
    md = config["model"]
    ls = config.setdefault("loss", {})

    tr["lr"] = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    tr["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    md["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)

    is_lstm = md.get("type") == "lstm"
    joint = md.get("joint", False)

    if is_lstm:
        # Scheduled-sampling dose — the dominant lever for autoregressive bias.
        ss = tr.setdefault("scheduled_sampling", {})
        ss["enabled"] = True
        ss["max_prob"] = trial.suggest_float("ss_max_prob", 0.02, 0.40)
        ss["start_epoch"] = trial.suggest_int("ss_start_epoch", 5, 40)
        ss["schedule"] = trial.suggest_categorical("ss_schedule", ["linear", "inv_sigmoid"])
        md["hidden_size"] = trial.suggest_categorical("hidden_size", [128, 192, 256])
    else:  # transformer — shallow, constraint-respecting
        d_model = trial.suggest_categorical("d_model", [64, 128])
        n_heads = trial.suggest_categorical("n_heads", [4, 8])
        if d_model % n_heads != 0:
            raise optuna.TrialPruned()  # invalid head/dim combo
        md["d_model"] = d_model
        md["n_heads"] = n_heads
        md["d_ff"] = trial.suggest_categorical("d_ff", [256, 512])
        md["n_layers"] = trial.suggest_int("n_layers", 2, 3)

    if joint:
        ls["spend_logvar_max"] = trial.suggest_categorical("spend_logvar_max", [1.0, 2.0, 3.0])
        ls["warmup_epochs"] = trial.suggest_int("kendall_warmup", 3, 12)
        ls["spend_loss_normalize"] = trial.suggest_categorical(
            "spend_loss_normalize", [True, False]
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exploratory bounded Optuna HPO for a CLV robustness config."
    )
    p.add_argument("--config", required=True, help="Base v3 YAML to tune.")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--max-epochs", type=int, default=120)
    p.add_argument("--n-scenarios", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL (e.g. sqlite:///hpo.db) for resumable studies.")
    p.add_argument("--timeout", type=int, default=None, help="Wall-clock seconds cap.")
    p.add_argument("--out-config", default=None,
                   help="Where to write the winning config (default: <base>_hpo.yaml).")
    p.add_argument("--kaggle", action="store_true",
                   help="Apply Kaggle raw_dir/results_dir overrides (or set KAGGLE_ENV=1).")
    p.add_argument("--kaggle-data-root", default=None,
                   help="Override the /kaggle/input parent dir for mounted datasets.")
    args = p.parse_args()

    print(
        "[exploratory] tune.py is outside the headline empirical protocol. "
        "Treat generated *_hpo configs as robustness evidence only unless the "
        "methodology and manifest explicitly promote them."
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base_config = load_config(args.config)
    base_run = base_config.get("output", {}).get("run_name", Path(args.config).stem)
    joint = base_config.get("model", {}).get("joint", False)
    study_name = args.study_name or f"hpo_{base_run}"

    on_kaggle = args.kaggle or os.environ.get("KAGGLE_ENV", "0") == "1"

    def objective(trial) -> float:
        cfg = copy.deepcopy(base_config)
        cfg["training"]["seed"] = args.seed
        cfg["training"]["epochs"] = args.max_epochs
        cfg["training"]["max_epochs"] = args.max_epochs
        cfg.setdefault("inference", {})["mode"] = "sample"
        cfg["inference"]["n_scenarios"] = args.n_scenarios
        cfg.setdefault("output", {})["run_name"] = f"{base_run}_hpo_trial{trial.number}"
        _suggest(trial, cfg)
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

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best = study.best_trial
    print(f"\n=== Best trial #{best.number} (objective={best.value:.4f}) ===")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    for k, v in best.user_attrs.items():
        print(f"  [metric] {k}: {v}")

    # Apply the winning params to a fresh copy of the base config and dump it.
    winner = copy.deepcopy(base_config)
    winner["training"]["seed"] = args.seed
    # Re-run _suggest using a FixedTrial so the same mapping logic writes the winner.
    _suggest(optuna.trial.FixedTrial(best.params), winner)
    winner.setdefault("output", {})["run_name"] = f"{base_run}_hpo"
    out_path = Path(args.out_config) if args.out_config else \
        Path("experiments/configs") / f"{base_run}_hpo.yaml"
    out_path.write_text(
        f"# Auto-generated by tune.py from {args.config}\n"
        f"# EXPLORATORY ONLY: not part of headline thesis results unless explicitly promoted.\n"
        f"# Best Optuna objective={best.value:.4f} (trial #{best.number}, "
        f"{args.n_trials} trials, max_epochs={args.max_epochs})\n"
        + yaml.safe_dump(winner, sort_keys=False, default_flow_style=False)
    )
    print(f"\nWinning config written: {out_path}")
    print("Run the final multi-seed evaluation with:")
    print(f"  python run_seeds.py --configs {out_path.stem} --seeds 42 7 123 --modes sample")


if __name__ == "__main__":
    main()
