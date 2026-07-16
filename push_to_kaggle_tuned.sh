#!/usr/bin/env bash
# push_to_kaggle_tuned.sh — Push the tuned-performance notebook
# (kaggle_tuned_runner.ipynb) as a SEPARATE Kaggle kernel.
#
# Like the primary runner, code ships via a runtime `git clone` of
# https://github.com/OttoPrins/thesis-code-final.git — commit and push the repo
# BEFORE running the kernel, or the clone will miss tune.py / configs changes.
#
# The primary push_to_kaggle.sh and its guards are untouched; this script
# stages the tuned notebook + its own kernel metadata into a temp dir so the
# two kernels never share a kernel-metadata.json.
#
# Usage:
#   bash push_to_kaggle_tuned.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Load KAGGLE_API_TOKEN from ~/.kaggle/kaggle.json (Bearer auth for KGAT_) ──
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: $KAGGLE_JSON not found. Download it from kaggle.com → Settings → API."
    exit 1
fi
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['key'])")

# ── Guard: the notebook must be the tuned workflow, not the fixed runner ──────
python3 - << 'PY'
import json
from pathlib import Path

meta = json.load(open("kernel-metadata-tuned.json"))
code_file = Path(meta["code_file"])
if not code_file.exists():
    raise SystemExit(f"ERROR: code_file does not exist: {code_file}")
text = code_file.read_text()
notebook = json.loads(text)
if notebook.get("nbformat") != 4:
    raise SystemExit("ERROR: unsupported nbformat")
required = [
    "Tuned Performance Workflow",
    "hpo-val-pct",
    "configs_tuned",
    "make_tuned_configs.py",
    "results_archive_tuned",
    "extension3_lstm_full_dunnhumby_tuned",
]
missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit("ERROR: tuned notebook is missing markers: " + ", ".join(missing))
forbidden = [
    "RUN_ALL_STAGES",          # fixed-runner stage switch
    "lstm_base_cdnow_final",   # fixed-runner config lists
    "Fixed Validation Workflow",
]
present = [needle for needle in forbidden if needle in text]
if present:
    raise SystemExit(
        "ERROR: tuned notebook contains fixed-runner content: " + ", ".join(present)
    )
if meta["id"] == "ottoprins/clv-thesis-runner":
    raise SystemExit("ERROR: tuned metadata points at the primary kernel id.")
print(f"  code_file: {code_file} — tuned workflow checks passed")
PY

# ── Warn if the repo the kernel will clone is behind the working tree ─────────
if ! git diff --quiet HEAD -- tune.py scripts/ experiments/ kaggle_tuned_runner.ipynb 2>/dev/null \
   || [[ -n "$(git status --porcelain tune.py scripts/ experiments/shap_samples experiments/configs_tuned 2>/dev/null)" ]]; then
    echo "WARNING: uncommitted changes in tune.py / scripts/ / experiments/."
    echo "         The kernel clones GitHub at runtime — commit & push first."
fi
if ! git diff --quiet origin/main..HEAD 2>/dev/null; then
    echo "WARNING: local commits not on origin/main — push before running the kernel."
fi

# ── Stage notebook + metadata in a temp dir and push ──────────────────────────
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp kaggle_tuned_runner.ipynb "$STAGE/"
cp kernel-metadata-tuned.json "$STAGE/kernel-metadata.json"

echo "=== Pushing tuned notebook to Kaggle ==="
kaggle kernels push -p "$STAGE"
echo "      Done."
echo ""
python3 - << 'PY'
import json
meta = json.load(open("kernel-metadata-tuned.json"))
print(f"  Notebook: https://www.kaggle.com/code/{meta['id']}")
print("")
print("  Before Run All:")
print("    1. Hard-refresh the Kaggle tab (Cmd-Shift-R) — the editor caches revisions.")
print("    2. Settings -> Accelerator -> GPU T4 x2.")
print("    3. Confirm the repo commit you want is on origin/main (or set THESIS_REF).")
PY
