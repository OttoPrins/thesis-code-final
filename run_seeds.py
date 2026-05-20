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

    # Smoke/repair dry run with bounded resource override
    python run_seeds.py --configs lstm_base_cdnow_v2 --max_epochs 1 --n_scenarios 2 --batch_size_override 32 --dry_run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import selectors
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


def _default_results_dir() -> Path:
    if os.environ.get("KAGGLE_ENV", "0") == "1":
        return Path("/kaggle/working/results")
    return Path("results")


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
    ok, reason = result_matches_manifest(
        stem,
        metrics,
        manifest,
        include_repaired=stem.endswith("_repair"),
    )
    if not ok:
        print(f"  [rerun] {path} exists but is not final-valid: {reason}", flush=True)
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
    p.add_argument("--batch_size_override", type=int, default=None,
                   help="Override training.batch_size for all jobs (OOM repair/smoke use).")
    p.add_argument("--heartbeat_interval", type=int, default=60,
                   help="Seconds between heartbeat lines for quiet subprocesses. Use 0 to disable.")
    p.add_argument("--logs_dir", default=None,
                   help="Directory for per-run subprocess logs. Default: <results_dir>/logs.")
    args = p.parse_args()

    # Build the full job list and report up front so the user knows what's coming.
    jobs = []
    for cfg_name in args.configs:
        cfg_path = CONFIGS_DIR / f"{cfg_name}.yaml"
        if not cfg_path.exists():
            print(f"WARN: config not found: {cfg_path}", file=sys.stderr, flush=True)
            continue
        for seed in args.seeds:
            for mode in args.modes:
                jobs.append((str(cfg_path), seed, mode))

    print(
        f"Sweep: {len(jobs)} runs "
        f"({len(args.configs)} configs × {len(args.seeds)} seeds × {len(args.modes)} modes)",
        flush=True,
    )
    if args.dry_run:
        for cfg, seed, mode in jobs:
            cmd = f"  {sys.executable} -u train.py --config {cfg} --seed_override {seed} --inference_mode {mode}"
            if args.max_epochs is not None:
                cmd += f" --max_epochs {args.max_epochs}"
            if args.n_scenarios is not None:
                cmd += f" --n_scenarios {args.n_scenarios}"
            if args.batch_size_override is not None:
                cmd += f" --batch_size_override {args.batch_size_override}"
            print(cmd, flush=True)
        return

    # Detect available GPUs for parallel dispatch.
    try:
        import torch as _torch
        _n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
    except ImportError:
        _n_gpus = 1
    n_parallel = min(_n_gpus, 2)

    # Pre-filter jobs serially (YAML peek + skip-existing check must be sequential
    # to avoid file-read races and interleaved stdout).
    import yaml as _yaml
    t0 = time.time()
    pending = []  # list of (cfg, seed, mode, cmd, env, label, run_name)
    for i, (cfg, seed, mode) in enumerate(jobs):
        cfg_basename = Path(cfg).stem
        with open(cfg) as _f:
            _cfg_yaml = _yaml.safe_load(_f)
        base_run_name = _cfg_yaml.get("output", {}).get("run_name", cfg_basename)
        run_name = f"{base_run_name}_seed{seed}_{mode}"
        expected_metrics = _default_results_dir() / "tables" / f"{run_name}_metrics.json"
        if args.skip_existing:
            if _existing_metrics_are_final_valid(expected_metrics):
                print(f"  [skip] {expected_metrics} already exists and is final-valid", flush=True)
                continue
            repair_metrics = expected_metrics.with_name(f"{run_name}_repair_metrics.json")
            if _existing_metrics_are_final_valid(repair_metrics):
                print(
                    f"  [skip] {repair_metrics} already exists and is final-valid repair",
                    flush=True,
                )
                continue

        gpu_id = i % n_parallel
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            sys.executable, "-u", "train.py",
            "--config", cfg,
            "--seed_override", str(seed),
            "--inference_mode", mode,
        ]
        if args.max_epochs is not None:
            cmd += ["--max_epochs", str(args.max_epochs)]
        if args.n_scenarios is not None:
            cmd += ["--n_scenarios", str(args.n_scenarios)]
        if args.batch_size_override is not None:
            cmd += ["--batch_size_override", str(args.batch_size_override)]

        label = f"{cfg_basename}  seed={seed}  mode={mode}  GPU={gpu_id}"
        pending.append((cfg, seed, mode, cmd, env, label, run_name))

    print(f"Running {len(pending)} job(s) with {n_parallel} parallel worker(s)", flush=True)

    failures = []

    logs_dir = Path(args.logs_dir) if args.logs_dir else _default_results_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _run_one(job):
        cfg, seed, mode, cmd, env, label, run_name = job
        elapsed_min = (time.time() - t0) / 60
        log_path = logs_dir / f"{run_name}.log"
        print(f"\n[t={elapsed_min:.1f}min]  {label}  log={log_path}", flush=True)

        started = time.time()
        latest_line = "(no output yet)"
        heartbeat_interval = max(0, int(args.heartbeat_interval))
        with open(log_path, "w", buffering=1) as log_f:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                timeout = heartbeat_interval if heartbeat_interval > 0 else None
                events = selector.select(timeout=timeout)
                if events:
                    line = process.stdout.readline()
                    if line:
                        latest_line = line.rstrip()
                        log_f.write(line)
                        print(line, end="", flush=True)
                    elif process.poll() is not None:
                        break
                else:
                    if process.poll() is not None:
                        break
                    run_min = (time.time() - started) / 60
                    total_min = (time.time() - t0) / 60
                    print(
                        f"[heartbeat t={total_min:.1f}min run={run_min:.1f}min] "
                        f"{label} still running; latest: {latest_line}",
                        flush=True,
                    )
            selector.unregister(process.stdout)
            # Drain any buffered tail after process exit.
            tail = process.stdout.read()
            if tail:
                log_f.write(tail)
                print(tail, end="", flush=True)
            rc = process.wait()
        return cfg, seed, mode, rc

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        for cfg, seed, mode, rc in executor.map(_run_one, pending):
            if rc != 0:
                failures.append((cfg, seed, mode, rc))
                print(f"  FAILED: {Path(cfg).stem} seed={seed} mode={mode} (exit {rc})", flush=True)

    print(f"\n=== Sweep complete in {(time.time()-t0)/60:.1f} min ===", flush=True)
    if failures:
        print(f"FAILURES ({len(failures)}):", flush=True)
        for cfg, seed, mode, code in failures:
            print(f"  {cfg} seed={seed} mode={mode} exit={code}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
