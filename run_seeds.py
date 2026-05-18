"""
Multi-seed sweep runner.

For each (config, seed) pair, invokes `python train.py --config <c> --seed_override <s>`.
Each run produces metrics at results/tables/<run_name>_seed<N>_metrics.json.

Usage:
    # All DL configs, default seeds {42, 7, 123}
    python run_seeds.py

    # Specific configs and seeds
    python run_seeds.py --configs lstm_base_cdnow lstm_joint_cdnow --seeds 42 7

    # Both inference modes per seed (doubles the run count)
    python run_seeds.py --modes sample expected
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path

from src.utils.final_manifest import load_final_manifest, result_matches_manifest

CONFIGS_DIR = Path("experiments/configs")

# Default sweep: every DL training config (excludes pure-benchmark configs).
DEFAULT_CONFIGS = [
    # V2 configs (hardened: rolling-origin, 300 epochs, n_scenarios=200, hurdle spend head)
    "lstm_base_cdnow_v2",
    "lstm_joint_cdnow_v2",
    "transformer_joint_cdnow_v2",
    "lstm_base_uci_v2",
    "lstm_joint_uci_v2",
    "transformer_joint_uci_v2",
    "lstm_base_tafeng_v2",
    "lstm_joint_tafeng_v2",
    "transformer_joint_tafeng_v2",
    "lstm_base_dunnhumby_v2",
    "lstm_joint_dunnhumby_v2",
    "transformer_joint_dunnhumby_v2",
    # Extension 3 (Dunnhumby covariate ablation — no v2 suffix; uses separate config track)
    "extension3_lstm_none_dunnhumby",
    "extension3_lstm_static_dunnhumby",
    "extension3_lstm_dynamic_dunnhumby",
    "extension3_lstm_full_dunnhumby",
    "extension3_transformer_none_dunnhumby",
    "extension3_transformer_static_dunnhumby",
    "extension3_transformer_dynamic_dunnhumby",
    "extension3_transformer_full_dunnhumby",
]

DEFAULT_SEEDS = [42, 7, 123]


def _existing_metrics_are_final_valid(path: Path) -> bool:
    """Return True only when an existing metrics file passes the final manifest gate."""
    if not path.exists():
        return False
    manifest = load_final_manifest()
    if manifest is None:
        return False
    try:
        with open(path) as f:
            metrics = json.load(f)
    except Exception:
        return False
    stem = path.stem.removesuffix("_metrics")
    ok, reason = result_matches_manifest(stem, metrics, manifest)
    if not ok:
        print(f"  [rerun] {path} exists but is not final-valid: {reason}")
    return ok


def main():
    p = argparse.ArgumentParser(description="Sweep train.py over configs × seeds × inference modes.")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="Config basenames (without .yaml). Default: all final DL configs.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Seeds to sweep over. Default: 42 7 123.")
    p.add_argument("--modes", nargs="+", default=["sample"],
                   choices=["sample", "expected"],
                   help="Inference modes to sweep. Default: just 'sample'.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs whose metrics JSON already exists in results/tables/.")
    p.add_argument("--max_epochs", type=int, default=None,
                   help="Override training.epochs (smoke/CI use).")
    p.add_argument("--n_scenarios", type=int, default=None,
                   help="Override inference.n_scenarios (smoke/CI use).")
    args = p.parse_args()

    # Build the full job list and report up front so the user knows what's coming.
    jobs = []
    for cfg_name in args.configs:
        cfg_path = CONFIGS_DIR / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            print(f"WARN: config not found: {cfg_path}", file=sys.stderr)
            continue
        for seed in args.seeds:
            for mode in args.modes:
                jobs.append((str(cfg_path), seed, mode))

    print(f"Sweep: {len(jobs)} runs ({len(args.configs)} configs × {len(args.seeds)} seeds × {len(args.modes)} modes)")
    if args.dry_run:
        for cfg, seed, mode in jobs:
            print(f"  python train.py --config {cfg} --seed_override {seed} --inference_mode {mode}")
        return

    failures = []
    t0 = time.time()
    for i, (cfg, seed, mode) in enumerate(jobs, start=1):
        elapsed_min = (time.time() - t0) / 60
        print(f"\n[{i}/{len(jobs)}  t={elapsed_min:.1f}min]  {Path(cfg).stem} seed={seed} mode={mode}")

        # Build the expected run_name from config basename + seed + mode
        cfg_basename = Path(cfg).stem
        # Peek at the YAML to find the configured run_name suffix and version
        import yaml as _yaml
        with open(cfg) as _f:
            _cfg_yaml = _yaml.safe_load(_f)
        base_run_name = _cfg_yaml.get("output", {}).get("run_name", cfg_basename)
        run_name = f"{base_run_name}_seed{seed}_{mode}"
        expected_metrics = Path("results/tables") / f"{run_name}_metrics.json"
        if args.skip_existing and _existing_metrics_are_final_valid(expected_metrics):
            print(f"  [skip] {expected_metrics} already exists and is final-valid")
            continue

        cmd = [
            sys.executable, "train.py",
            "--config", cfg,
            "--seed_override", str(seed),
            "--inference_mode", mode,
        ]
        if args.max_epochs is not None:
            cmd += ["--max_epochs", str(args.max_epochs)]
        if args.n_scenarios is not None:
            cmd += ["--n_scenarios", str(args.n_scenarios)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((cfg, seed, mode, result.returncode))
            print(f"  FAILED (exit {result.returncode})")

    print(f"\n=== Sweep complete in {(time.time()-t0)/60:.1f} min ===")
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for cfg, seed, mode, code in failures:
            print(f"  {cfg} seed={seed} mode={mode} exit={code}")
        sys.exit(1)


if __name__ == "__main__":
    main()
