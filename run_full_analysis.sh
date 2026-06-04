#!/usr/bin/env bash
# run_full_analysis.sh — Full overnight analysis pipeline
#
# Usage:
#   nohup bash run_full_analysis.sh > run_full_analysis.log 2>&1 &   # full overnight (~8-14 h)
#   FORCE=1 bash run_full_analysis.sh                                 # force re-run all steps
#   bash run_full_analysis.sh --smoke                                 # 8-min CDNOW sanity sweep
#
# Estimated wall clock: 8–14 hours on Apple Silicon MPS (full mode).
# Outputs land in results/tables/, results/plots/ (notebook), and experiments/insights/.
#
# Environment: activate your venv before running, e.g.:
#   source .venv/bin/activate && nohup bash run_full_analysis.sh > run.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
    if [[ -x "./venv/bin/python" ]]; then
        PYTHON_BIN="./venv/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    else
        echo "No Python interpreter found. Set PYTHON_BIN=/path/to/python." >&2
        exit 127
    fi
fi

LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRIPT_DIR/results/cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/results/cache/matplotlib}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Parse --smoke flag
SMOKE=0
for arg in "$@"; do
    if [[ "$arg" == "--smoke" ]]; then
        SMOKE=1
    fi
done

# Force flag: FORCE=1 re-runs even if outputs exist
FORCE="${FORCE:-0}"

# Timestamp helper
ts() { date '+%Y-%m-%d %H:%M:%S'; }

# skip_if_exists <file> : returns 0 (skip) if file exists and FORCE=0, else 1 (run)
skip_if_exists() {
    local f="$1"
    if [[ "$FORCE" != "1" ]] && [[ -f "$f" ]]; then
        echo "$(ts)   [skip] $(basename "$f") already exists (set FORCE=1 to overwrite)"
        return 0
    fi
    return 1
}

metric_is_final_valid() {
    local f="$1"
    [[ -f "$f" ]] || return 1
    "$PYTHON_BIN" - "$f" <<'PY'
import json
import sys
from pathlib import Path

from src.utils.final_manifest import load_final_manifest, result_matches_manifest

path = Path(sys.argv[1])
manifest = load_final_manifest()
if manifest is None:
    sys.exit(1)
try:
    metrics = json.load(open(path))
except Exception:
    sys.exit(1)
stem = path.stem.removesuffix("_metrics")
ok, reason = result_matches_manifest(stem, metrics, manifest)
if not ok:
    print(f"[stale] {path.name}: {reason}")
sys.exit(0 if ok else 1)
PY
}

if [[ "$SMOKE" == "1" ]]; then
    echo "$(ts) === SMOKE MODE (CDNOW only, 1 seed, 3 epochs, 2 scenarios) ==="
else
    echo "$(ts) === Starting full analysis pipeline ==="
fi
echo "$(ts) Working directory: $(pwd)"
echo "$(ts) Python: $("$PYTHON_BIN" --version 2>&1) ($PYTHON_BIN)"

# ---------------------------------------------------------------------------
# Step 0 — Final preflight and pipeline sanity checks
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) [Step 0] Final preflight ..."
if [[ "$SMOKE" == "1" ]]; then
    "$PYTHON_BIN" validate_final_setup.py --smoke 2>&1 | tee "$LOG_DIR/final_preflight.log"
    "$PYTHON_BIN" validate_pipelines.py --dataset cdnow 2>&1 | tee "$LOG_DIR/validate_pipelines.log"
else
    "$PYTHON_BIN" validate_final_setup.py 2>&1 | tee "$LOG_DIR/final_preflight.log"
    "$PYTHON_BIN" validate_pipelines.py 2>&1 | tee "$LOG_DIR/validate_pipelines.log"
fi
echo "$(ts) [Step 0] Done."

# ---------------------------------------------------------------------------
# Step 0.5 — Phase-1 gate: CDNOW Base LSTM must have bias_pct ≤ 8%
#   Checks the most-recent CDNOW base metrics file. If none exist, runs a
#   canary train. If bias_pct > 8%, the sweep exits early (fail-fast).
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) [Step 0.5] CDNOW replication gate ..."
if [[ "$SMOKE" == "1" ]]; then
    echo "$(ts)   Smoke mode — skipping replication gate (unconverged 3-epoch run expected)."
    echo "$(ts) [Step 0.5] Done."
fi
if [[ "$SMOKE" != "1" ]]; then
	CDNOW_METRICS=$(for f in $(ls results/tables/lstm_base_cdnow*_metrics.json 2>/dev/null \
	    | grep -v "comparison_\|_seed\|_sample\|_expected" | sort); do
	        metric_is_final_valid "$f" >/dev/null 2>&1 && echo "$f"
	    done | tail -1 || echo "")

if [[ -z "$CDNOW_METRICS" ]]; then
    echo "$(ts)   No CDNOW Base LSTM metrics found — running canary training ..."
    "$PYTHON_BIN" train.py --config experiments/configs_final/lstm_base_cdnow_final.yaml \
        2>&1 | tee "$LOG_DIR/gate_cdnow.log"
	    CDNOW_METRICS=$(for f in $(ls results/tables/lstm_base_cdnow*_metrics.json 2>/dev/null \
	        | grep -v "comparison_" | sort); do
	            metric_is_final_valid "$f" >/dev/null 2>&1 && echo "$f"
	        done | tail -1 || echo "")
fi

if [[ -n "$CDNOW_METRICS" ]]; then
    BIAS=$("$PYTHON_BIN" -c "
import json, sys
d = json.load(open('$CDNOW_METRICS'))
b = d.get('bias_pct', 999.0)
print(f'{b:.1f}')
sys.exit(0 if abs(float(b)) <= 8.0 else 1)
" 2>/dev/null) || {
        BIAS=$("$PYTHON_BIN" -c "import json; d=json.load(open('$CDNOW_METRICS')); print(f\"{d.get('bias_pct',999):.1f}\")" 2>/dev/null || echo "999")
        echo "$(ts) [FATAL] CDNOW Base LSTM bias_pct=${BIAS}% > 8% — fix replication before sweep."
        echo "$(ts)   Check: cohort_end_date and origin_date in experiments/configs_final/lstm_base_cdnow_final.yaml"
        echo "$(ts)   Then re-run: python train.py --config experiments/configs_final/lstm_base_cdnow_final.yaml"
        exit 1
    }
    echo "$(ts)   CDNOW Base LSTM bias_pct=${BIAS}% ≤ 8% — gate passed."
fi
echo "$(ts) [Step 0.5] Done."
fi  # end of non-smoke gate block

# ---------------------------------------------------------------------------
# Step 1 — Probabilistic benchmarks
#   The fixed final manifest includes Pareto/NBD under the same base configs.
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) [Step 1] Running probabilistic benchmarks ..."

if [[ "$SMOKE" == "1" ]]; then
    BENCHMARK_CONFIGS=("experiments/configs_final/lstm_base_cdnow_final.yaml")
    BENCH_MODELS_BY_DATASET_cdnow="pareto_nbd"
else
    BENCHMARK_CONFIGS=(
        "experiments/configs_final/lstm_base_cdnow_final.yaml"
        "experiments/configs_final/lstm_base_uci_final.yaml"
        "experiments/configs_final/lstm_base_tafeng_final.yaml"
        "experiments/configs_final/lstm_base_dunnhumby_final.yaml"
    )
    BENCH_MODELS_BY_DATASET_cdnow="pareto_nbd"
    BENCH_MODELS_BY_DATASET_uci="pareto_nbd"
    BENCH_MODELS_BY_DATASET_tafeng="pareto_nbd"
    BENCH_MODELS_BY_DATASET_dunnhumby="pareto_nbd"
fi

for cfg in "${BENCHMARK_CONFIGS[@]}"; do
    ds=$(basename "$cfg" .yaml | sed 's/lstm_base_//' | sed 's/_final//')
    var_name="BENCH_MODELS_BY_DATASET_${ds}"
    BENCH_MODELS="${!var_name}"
    # Check if ALL expected benchmark outputs exist (skip only if all present)
    all_exist=1
    for m in $BENCH_MODELS; do
        f="results/tables/${m}_${ds}_metrics.json"
        metric_is_final_valid "$f" || { all_exist=0; break; }
    done
    if [[ "$all_exist" == "1" ]] && [[ "$FORCE" != "1" ]]; then
        echo "$(ts)   [skip] Benchmarks for ${ds} already complete"
        continue
    fi
    echo "$(ts)   Benchmarks for ${ds} ..."
    "$PYTHON_BIN" run_benchmarks.py --config "$cfg" --models $BENCH_MODELS \
        2>&1 | tee "$LOG_DIR/benchmarks_${ds}.log"
    echo "$(ts)   ${ds} benchmarks done."
done

echo "$(ts) [Step 1] Done."

# ---------------------------------------------------------------------------
# Step 2 — Multi-seed deep-learning sweep (sample mode — thesis default)
#   20 configs × 3 seeds = 60 invocations (full mode).
#   Smoke: 1 config × 1 seed × 3 epochs = 1 invocation.
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) [Step 2] Multi-seed DL sweep (sample mode, seeds 42 7 2024) ..."

if [[ "$SMOKE" == "1" ]]; then
    "$PYTHON_BIN" run_seeds.py \
        --config_dir experiments/configs_final \
        --configs lstm_base_cdnow_final \
        --seeds 42 \
        --modes sample \
        --skip_existing \
        --max_epochs 3 \
        --n_scenarios 2 \
        2>&1 | tee "$LOG_DIR/run_seeds_sample.log"
else
    "$PYTHON_BIN" run_seeds.py --config_dir experiments/configs_final --seeds 42 7 2024 --modes sample --skip_existing \
        2>&1 | tee "$LOG_DIR/run_seeds_sample.log"
fi

echo "$(ts) [Step 2] Done."

# ---------------------------------------------------------------------------
# Step 3 — Expected-mode diagnostic (base configs only — cheap, diagnostic only)
#   Compares bias under sampling vs deterministic expected-value inference.
#   Skipped in smoke mode.
# ---------------------------------------------------------------------------
if [[ "$SMOKE" != "1" ]]; then
    echo ""
    echo "$(ts) [Step 3] Expected-mode diagnostic (base configs, seeds 42 7 2024) ..."
    "$PYTHON_BIN" run_seeds.py \
        --config_dir experiments/configs_final \
        --configs lstm_base_cdnow_final lstm_base_uci_final lstm_base_tafeng_final lstm_base_dunnhumby_final \
        --seeds 42 7 2024 \
        --modes expected \
        --skip_existing \
        2>&1 | tee "$LOG_DIR/run_seeds_expected.log"
    echo "$(ts) [Step 3] Done."
else
    echo ""
    echo "$(ts) [Step 3] Skipped in smoke mode."
fi

# ---------------------------------------------------------------------------
# Step 4 — Extension 3: SHAP covariate attribution (Dunnhumby only)
#   Requires extension3_lstm_full_dunnhumby to have run in Step 2.
#   Skipped in smoke mode.
# ---------------------------------------------------------------------------
if [[ "$SMOKE" != "1" ]]; then
    echo ""
    echo "$(ts) [Step 4] SHAP analysis for Extension 3 (Dunnhumby) ..."

    SHAP_OUT="results/tables/shap_extension3_lstm_full_dunnhumby.csv"
    if skip_if_exists "$SHAP_OUT"; then
        echo "$(ts) [Step 4] Done (skipped — already exists)."
    else
        # Find the latest full-covariate LSTM checkpoint (seed 42 is canonical)
        SHAP_CKPT=$(ls results/checkpoints/extension3_lstm_full_dunnhumby_final*seed42*.pt 2>/dev/null | tail -1 || \
                    ls results/checkpoints/extension3_lstm_full_dunnhumby_final*.pt 2>/dev/null | tail -1 || echo "")

        if [ -z "$SHAP_CKPT" ]; then
            echo "$(ts)   WARNING: No extension3_dunnhumby checkpoint found — skipping SHAP."
            echo "$(ts)   Ensure Step 2 completed successfully for extension3_lstm_full_dunnhumby."
        else
            echo "$(ts)   Using checkpoint: ${SHAP_CKPT}"
            "$PYTHON_BIN" -m src.evaluation.shap_analysis \
                --config experiments/configs_final/extension3_lstm_full_dunnhumby_final.yaml \
                --checkpoint "$SHAP_CKPT" \
                --n_background 100 \
                --n_explain 200 \
                --out_dir results/tables \
                2>&1 | tee "$LOG_DIR/shap_analysis.log"
            echo "$(ts)   SHAP done."
        fi
        echo "$(ts) [Step 4] Done."
    fi
else
    echo ""
    echo "$(ts) [Step 4] Skipped in smoke mode."
fi

# ---------------------------------------------------------------------------
# Step 5 — Aggregate all results into comparison CSVs
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) [Step 5] Aggregating results ..."
if [[ "$SMOKE" == "1" ]]; then
    "$PYTHON_BIN" -m src.evaluation.compare --seeds --latex \
        2>&1 | tee "$LOG_DIR/compare.log"
else
    "$PYTHON_BIN" -m src.evaluation.compare --seeds --latex --plots --weights \
        2>&1 | tee "$LOG_DIR/compare.log"
fi
echo "$(ts) [Step 5] Done."

# ---------------------------------------------------------------------------
# Step 6 — Paired bootstrap significance tests
#   Skipped in smoke mode (not enough seeds).
# ---------------------------------------------------------------------------
if [[ "$SMOKE" != "1" ]]; then
    echo ""
    echo "$(ts) [Step 6] Running significance tests ..."
    "$PYTHON_BIN" -m src.evaluation.significance \
        --metrics_dir results/tables \
        --pairs \
            lstm_joint_cdnow_final_seed42_sample:lstm_base_cdnow_final_seed42_sample \
            transformer_joint_cdnow_final_seed42_sample:lstm_joint_cdnow_final_seed42_sample \
            lstm_joint_cdnow_final_seed42_sample:pareto_nbd_cdnow \
            lstm_joint_cdnow_final_seed42_sample:bgnbd_gg_cdnow \
            lstm_joint_tafeng_final_seed42_sample:lstm_base_tafeng_final_seed42_sample \
            transformer_joint_tafeng_final_seed42_sample:lstm_joint_tafeng_final_seed42_sample \
            lstm_joint_uci_final_seed42_sample:lstm_base_uci_final_seed42_sample \
            lstm_joint_dunnhumby_final_seed42_sample:lstm_base_dunnhumby_final_seed42_sample \
        --metrics freq_rmse spend_mae clv_mae \
        --n_resamples 10000 \
        --seed_filter 42 \
        --mode_filter sample \
        --out results/tables/significance_tests.csv \
        2>&1 | tee "$LOG_DIR/significance.log"
    echo "$(ts) [Step 6] Done."
else
    echo ""
    echo "$(ts) [Step 6] Skipped in smoke mode."
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
echo "$(ts) === Pipeline complete ==="
echo ""
echo "Results:"
echo "  results/tables/comparison_seeds.csv  — master comparison table"
echo "  results/tables/comparison_all.tex    — LaTeX comparison table"
if [[ "$SMOKE" != "1" ]]; then
    echo "  results/tables/significance_tests.csv — bootstrap significance tests"
fi
echo "  experiments/insights/                 — compare.py figures and Kendall weight plots"
echo "  results/plots/                        — notebook-generated thesis figures"
echo "  results/logs/                         — per-step logs"
echo ""

if [[ "$SMOKE" != "1" ]]; then
    echo "Phase 3 verification checklist:"
    echo "  1. Check comparison_seeds.csv: all DL models should have n_seeds=3"
    echo "  2. CDNOW Base LSTM bias_pct should be ≤ 5% (Valendin target ~2%)"
    echo "  3. Joint LSTM spend_r2_log should be > 0 on all datasets"
    echo "  4. Joint LSTM clv_mae and clv_spearman present for all datasets"
    echo "  5. Open notebooks/06_results_comparison.ipynb to inspect all plots"
    echo ""
fi
echo "$(ts) Done. Open run_full_analysis.log for full transcript."
