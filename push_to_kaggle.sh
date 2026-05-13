#!/usr/bin/env bash
# push_to_kaggle.sh — Push the Kaggle notebook (kaggle_runner.ipynb).
#
# Code is no longer shipped as a Kaggle dataset; Cell 1 in the notebook does a
# `git clone https://github.com/OttoPrins/thesis-code-final.git` at runtime.
# This script therefore only pushes the notebook itself.
#
# Usage:
#   bash push_to_kaggle.sh
#
# After running, hard-refresh the Kaggle notebook tab (⌘⇧R / Ctrl-Shift-R)
# before clicking Run All — the web editor caches the previous revision.

set -euo pipefail

# ── Load KAGGLE_API_TOKEN from ~/.kaggle/kaggle.json ──────────────────────────
# KGAT_ tokens require Bearer auth. Exporting KAGGLE_API_TOKEN forces Bearer mode
# in the kagglesdk HTTP client instead of the default Basic auth, which rejects KGAT_ tokens.
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: $KAGGLE_JSON not found. Download it from kaggle.com → Settings → API."
    exit 1
fi
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['key'])")

echo "=== Pushing notebook to Kaggle ==="
kaggle kernels push -p .
echo "      Done."
echo ""

python3 - << 'PY'
import json
k = json.load(open("kernel-metadata.json"))
print(f"  Notebook: https://www.kaggle.com/code/{k['id']}")
print("")
print("  Reminder: hard-refresh the Kaggle tab (⌘⇧R / Ctrl-Shift-R) before Run All.")
PY
