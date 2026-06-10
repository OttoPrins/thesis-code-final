#!/usr/bin/env bash
# Build and upload the self-contained SHAP code/checkpoint bundle, then push
# the dedicated full-cohort Kaggle notebook.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_JSON" ]]; then
    echo "ERROR: $KAGGLE_JSON not found."
    exit 1
fi
export KAGGLE_API_TOKEN
KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$KAGGLE_JSON'))['key'])")

DATASET_ID="ottoprins/thesis-shap-full-run"
KERNEL_ID="ottoprins/clv-thesis-shap-full-cohort"
MESSAGE="Full-cohort SHAP bundle $(date +%Y-%m-%d)"

DATA_STAGE=$(mktemp -d)
KERNEL_STAGE=$(mktemp -d)
cleanup() {
    rm -rf "$DATA_STAGE" "$KERNEL_STAGE"
}
trap cleanup EXIT

BUNDLE_ROOT="$DATA_STAGE/thesis-code"
mkdir -p \
    "$BUNDLE_ROOT/scripts" \
    "$BUNDLE_ROOT/experiments/configs_final" \
    "$BUNDLE_ROOT/results/final_kaggle/checkpoints" \
    "$BUNDLE_ROOT/pilot"

cp -R src "$BUNDLE_ROOT/"
find "$BUNDLE_ROOT/src" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_ROOT/src" -type f -name '*.pyc' -delete
find "$BUNDLE_ROOT" -type f -name '.DS_Store' -delete
cp train.py "$BUNDLE_ROOT/"
cp scripts/run_extension3_shap.py "$BUNDLE_ROOT/scripts/"
cp experiments/configs_final/extension3_lstm_full_dunnhumby_final.yaml \
    "$BUNDLE_ROOT/experiments/configs_final/"
cp experiments/configs_final/extension3_transformer_full_dunnhumby_final.yaml \
    "$BUNDLE_ROOT/experiments/configs_final/"
cp results/final_kaggle/tables/shap_extension3_convergence.csv \
    "$BUNDLE_ROOT/pilot/"
cp requirements-kaggle.txt "$BUNDLE_ROOT/"

for architecture in lstm transformer; do
    for seed in 7 42 2024; do
        checkpoint="results/final_kaggle/checkpoints/extension3_${architecture}_full_dunnhumby_final_seed${seed}_sample.pt"
        if [[ ! -f "$checkpoint" ]]; then
            echo "ERROR: missing checkpoint $checkpoint"
            exit 1
        fi
        cp "$checkpoint" "$BUNDLE_ROOT/results/final_kaggle/checkpoints/"
    done
done

python3 - "$BUNDLE_ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
files = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "BUNDLE_MANIFEST.json":
        continue
    files.append(
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
manifest = {
    "purpose": "Extension 3 SHAP full observed-demographic cohort",
    "architectures": ["lstm", "transformer"],
    "seeds": [7, 42, 2024],
    "n_checkpoints": 6,
    "files": files,
}
(root / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"Bundle manifest: {len(files)} files")
PY

(
    cd "$DATA_STAGE"
    zip -qr thesis_shap_bundle.zip thesis-code
    rm -rf thesis-code
)

cat > "$DATA_STAGE/dataset-metadata.json" <<'JSON'
{
  "title": "Thesis SHAP Full Run",
  "id": "ottoprins/thesis-shap-full-run",
  "licenses": [{"name": "other"}]
}
JSON

echo "Uploading private SHAP bundle dataset..."
if kaggle datasets metadata "$DATASET_ID" >/dev/null 2>&1; then
    kaggle datasets version -p "$DATA_STAGE" -m "$MESSAGE" --dir-mode zip
else
    kaggle datasets create -p "$DATA_STAGE" --dir-mode zip
fi

# A kernel pushed immediately after a dataset version can receive the previous
# version while Kaggle finishes indexing the new archive.
echo "Waiting for Kaggle to index the uploaded bundle..."
sleep 20

cp kaggle_shap_runner.ipynb "$KERNEL_STAGE/"
cp kernel-metadata-shap.json "$KERNEL_STAGE/kernel-metadata.json"

python3 - "$KERNEL_STAGE/kaggle_shap_runner.ipynb" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
notebook = json.loads(path.read_text())
text = path.read_text()
required = [
    "'--n_explain', '701'",
    "'--sensitivity_n_explain', '100'",
    "'--fixed_integration_samples', '128'",
    "extension3_shap_full_cohort",
]
missing = [value for value in required if value not in text]
if missing:
    raise SystemExit("Notebook validation failed: " + ", ".join(missing))
if notebook.get("nbformat") != 4:
    raise SystemExit("Notebook validation failed: unsupported nbformat")
print("Notebook validation passed.")
PY

echo "Pushing dedicated Kaggle notebook..."
kaggle kernels push -p "$KERNEL_STAGE"

echo
echo "Prepared successfully."
echo "Notebook: https://www.kaggle.com/code/$KERNEL_ID"
echo "Input bundle: https://www.kaggle.com/datasets/$DATASET_ID"
echo
echo "The Kaggle CLI push starts a run automatically."
echo "For a manual rerun, open the notebook, confirm GPU is enabled, and click Run All."
