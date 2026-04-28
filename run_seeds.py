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
from pathlib import Path

CONFIGS_DIR = Path("experiments/configs")

# Default sweep: every DL training config (excludes pure-benchmark configs).
DEFAULT_CONFIGS = [
    "lstm_base_cdnow",
    "lstm_joint_cdnow",
    "transformer_joint_cdnow",
    "lstm_base_uci",
    "lstm_joint_uci",
    "transformer_joint_uci",
    "lstm_base_tafeng",
    "lstm_joint_tafeng",
    "transformer_joint_tafeng",
    "lstm_base_dunnhumby",
    "lstm_joint_dunnhumby",
    "transformer_joint_dunnhumby",
    "extension3_dunnhumby",
]

DEFAULT_SEEDS = [42, 7, 123]


def main():
    p = argparse.ArgumentParser(description="Sweep train.py over configs × seeds × inference modes.")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="Config basenames (without .yaml). Default: all 13 DL configs.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Seeds to sweep over. Default: 42 7 123.")
    p.add_argument("--modes", nargs="+", default=["sample"],
                   choices=["sample", "expected"],
                   help="Inference modes to sweep. Default: just 'sample'.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing.")
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
        cmd = [
            sys.executable, "train.py",
            "--config", cfg,
            "--seed_override", str(seed),
            "--inference_mode", mode,
        ]
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
