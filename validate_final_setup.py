"""
Final-analysis preflight checks.

This script verifies the data identity and dependency gates that matter before
starting final thesis runs. Smoke mode checks only the lightweight CDNOW path;
full mode also requires Online Retail II, Pareto/GGG dependencies, and CmdStan
for the true CDNOW GPPM.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from src.data.datasets import CDNOWPipeline, DunnhumbyPipeline, UCIRetailPipeline
from src.utils.config import load_config
from src.utils.final_manifest import (
    expected_config_hashes,
    load_final_manifest,
    manifest_config_hash,
)


def _ok(message: str) -> None:
    print(f"[OK]   {message}")


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _check_manifest() -> dict:
    manifest = load_final_manifest()
    if manifest is None:
        _fail("experiments/final_manifest.yaml is missing.")
    _ok(f"final manifest loaded: {manifest['version']}")
    frozen_hashes = expected_config_hashes(manifest)
    if not frozen_hashes:
        _fail("final manifest must store deep_learning_config_hashes.")
    for cfg_path in manifest.get("deep_learning_configs", []):
        cfg = load_config(cfg_path)
        actual = manifest_config_hash(cfg)
        expected = frozen_hashes.get(cfg_path)
        if expected is None:
            _fail(f"{cfg_path}: missing frozen config hash in final manifest.")
        if actual != expected:
            _fail(
                f"{cfg_path}: frozen hash mismatch "
                f"(manifest={expected}, current={actual})."
            )
        _ok(f"{cfg_path}: frozen hash={actual} run={cfg['output']['run_name']}")
    runtime = manifest.get("runtime_expectations", {}).get("deep_learning", {})
    if runtime.get("epochs") != 100:
        _fail("final manifest runtime must freeze deep-learning epochs at 100.")
    if runtime.get("inference_mode") != manifest.get("methodology", {}).get("inference_primary"):
        _fail("final manifest runtime inference_mode must match methodology.inference_primary.")
    if runtime.get("n_scenarios") != 30:
        _fail("final manifest runtime must freeze n_scenarios at 30.")
    return manifest


def _check_cdnow() -> None:
    cfg = load_config("experiments/configs/lstm_base_cdnow.yaml")
    pipe = CDNOWPipeline()
    pipe._current_dataset_cfg = cfg["dataset"]
    raw = pipe.load_raw(cfg["dataset"].get("raw_dir", "data/raw"))
    clean = pipe.clean(raw)
    raw_customers = raw["customer_id"].nunique()
    clean_customers = clean["customer_id"].nunique()
    if raw_customers != 2357:
        _fail(
            f"CDNOW final run must use the canonical 2,357-customer sample; "
            f"loaded {raw_customers} raw customers."
        )
    if cfg["dataset"]["calibration_weeks"] != 52 or cfg["dataset"]["holdout_weeks"] != 26:
        _fail("CDNOW split must be 52/26 for final runs.")
    _ok(
        "CDNOW canonical sample detected "
        f"(2,357 raw customers; {clean_customers} with positive spend, split 52/26)"
    )


def _check_uci_online_retail_ii() -> None:
    cfg = load_config("experiments/configs/lstm_base_uci.yaml")
    pipe = UCIRetailPipeline()
    pipe._current_dataset_cfg = cfg["dataset"]
    raw = pipe.load_raw(cfg["dataset"].get("raw_dir", "data/raw"))
    clean = pipe.clean(raw)
    pipe._validate_online_retail_ii_identity(clean, cfg["dataset"].get("raw_dir", "data/raw"))
    _ok(
        "UCI Online Retail II detected "
        f"({clean['date'].min().date()} to {clean['date'].max().date()})"
    )


def _check_dunnhumby_window() -> None:
    cfg = load_config("experiments/configs/lstm_base_dunnhumby.yaml")
    raw_dir = Path(cfg["dataset"].get("raw_dir", "data/raw/Dunnhumby datasets"))
    path = raw_dir / "transaction_data.csv"
    if not path.exists():
        path = raw_dir / "transactions.csv"
    if not path.exists():
        _fail(f"Dunnhumby transaction file missing in {raw_dir}")
    df = pd.read_csv(path, usecols=["DAY", "household_key"])
    max_week = int(df["DAY"].max() // 7)
    needed_week = cfg["dataset"]["calibration_weeks"] + cfg["dataset"]["holdout_weeks"] - 1
    if max_week < needed_week:
        _fail(
            f"Dunnhumby file ends at week {max_week}, but final split needs week "
            f"{needed_week} for {cfg['dataset']['calibration_weeks']}/"
            f"{cfg['dataset']['holdout_weeks']}."
        )
    if df["household_key"].nunique() < 2000:
        _fail("Dunnhumby household count is unexpectedly low.")
    _ok(f"Dunnhumby supports 80/22 split (max week {max_week})")


def _check_import(module: str, purpose: str) -> None:
    try:
        importlib.import_module(module)
    except Exception as exc:
        _fail(f"{purpose} dependency missing: import {module!r} failed ({exc})")
    _ok(f"{purpose} dependency importable: {module}")


def _check_pareto_ggg() -> None:
    _check_import("rpy2", "Pareto/GGG")
    try:
        import rpy2.robjects.packages as rpackages

        if not rpackages.isinstalled("BTYDplus"):
            _fail(
                "Pareto/GGG dependency missing: R package BTYDplus is not installed. "
                "Install with: R -e 'remotes::install_github(\"mplatzer/BTYDplus\", dependencies=TRUE)'"
            )
    except RuntimeError as exc:
        _fail(f"Pareto/GGG R dependency check failed: {exc}")
    _ok("Pareto/GGG R package available: BTYDplus")


def _check_gppm() -> None:
    _check_import("cmdstanpy", "true GPPM")
    try:
        from cmdstanpy import cmdstan_path

        cmdstan_path()
    except Exception as exc:
        _fail(
            "true GPPM dependency missing: CmdStan is not installed or configured "
            f"({exc}). Run: python -m cmdstanpy.install_cmdstan"
        )
    _ok("true GPPM CmdStan installation available")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final thesis analysis setup.")
    parser.add_argument("--smoke", action="store_true", help="Check only smoke-run prerequisites")
    args = parser.parse_args()

    try:
        _check_manifest()
        _check_import("lifetimes", "BTYD benchmark")
        _check_cdnow()
        if not args.smoke:
            _check_uci_online_retail_ii()
            _check_dunnhumby_window()
            _check_pareto_ggg()
            _check_gppm()
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    print("[OK]   final setup preflight complete")


if __name__ == "__main__":
    main()
