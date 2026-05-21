#!/usr/bin/env bash
# upload_data_to_kaggle.sh — Upload the three raw datasets that Kaggle needs.
#
# Run this ONCE from the repo root before running the notebook.
# Subsequent runs create new versions of already-existing datasets (idempotent).
#
# Datasets created:
#   ottoprins/cdnow-dataset   ← data/raw/CDNOW_sample.txt + data/raw/CDNOW_master.txt
#   ottoprins/uci-retail      ← data/raw/online_retail_II.xlsx
#   ottoprins/tafeng-dataset  ← data/raw/ta_feng_all_months_merged.csv
#
# Dunnhumby is already on Kaggle — not touched here.
#
# After this script completes:
#   bash push_to_kaggle.sh     ← re-push the notebook (so metadata picks up new slugs)
#   Hard-refresh the Kaggle notebook tab
#   All four datasets will auto-attach.

set -euo pipefail

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: $KAGGLE_JSON not found. Download from kaggle.com → Settings → API."
    exit 1
fi
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['key'])")

KAGGLE_USER="ottoprins"
MSG="Raw data upload $(date +%Y-%m-%d)"

# ── Helper: create or version a dataset ──────────────────────────────────────
# Accepts one or more source files; all are copied into the staging dir.
upload_dataset() {
    local slug="$1"
    shift
    local stage
    stage=$(mktemp -d)
    trap "rm -rf '$stage'" RETURN

    # Write dataset-metadata.json
    python3 - <<PY
import json
meta = {"title": "$slug", "id": "$KAGGLE_USER/$slug", "licenses": [{"name": "other"}]}
with open("$stage/dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
PY

    # Copy each raw file into the staging dir
    local src_file
    for src_file in "$@"; do
        if [[ ! -f "$src_file" ]]; then
            echo "  ✗  Missing source file: $src_file"
            return 1
        fi
        cp "$src_file" "$stage/"
        echo "    staged: $(basename "$src_file")"
    done

    echo "  Uploading $slug ..."
    if kaggle datasets metadata "$KAGGLE_USER/$slug" &>/dev/null; then
        kaggle datasets version -p "$stage" -m "$MSG" --dir-mode zip --quiet
        echo "  ✓  $slug — new version created."
    else
        kaggle datasets create -p "$stage" --dir-mode zip --quiet
        echo "  ✓  $slug — created."
    fi
}

echo "=== Uploading raw datasets to Kaggle ==="
echo ""

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# CDNOW: ship BOTH the 2,357-customer sample (default replication runs) AND the
# 23,570-customer master (Valendin et al. 2022 39×39 replication protocol).
upload_dataset "cdnow-dataset" \
    "$REPO_ROOT/data/raw/CDNOW_sample.txt" \
    "$REPO_ROOT/data/raw/CDNOW_master.txt"
upload_dataset "uci-retail"     "$REPO_ROOT/data/raw/online_retail_II.xlsx"
upload_dataset "tafeng-dataset" "$REPO_ROOT/data/raw/ta_feng_all_months_merged.csv"

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. bash push_to_kaggle.sh          (re-push notebook so metadata is fresh)"
echo "  2. Open the Kaggle notebook and hard-refresh (Cmd+Shift+R)"
echo "  3. All four datasets should auto-attach in the right panel."
