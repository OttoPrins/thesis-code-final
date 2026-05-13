"""YAML config loading utility."""

import os

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# Maps dataset.name (short internal key used in YAML configs) to the Kaggle dataset
# slug (the URL component after /datasets/ottoprins/).  Kaggle enforces a minimum
# slug length of 7 characters, so short names like "cdnow" or "uci" cannot be used
# as slugs directly.
_KAGGLE_SLUG_MAP = {
    "cdnow": "cdnow-dataset",
    "uci": "uci-retail",
    "tafeng": "tafeng-dataset",
    "dunnhumby": "dunnhumby",
}


def apply_kaggle_overrides(cfg: dict, data_root: str | None = None) -> dict:
    """Remap raw_dir and results_dir for Kaggle's read-only input / writable output layout.

    Kaggle mounts user datasets at /kaggle/input/<dataset-slug>/ (read-only) and
    requires all outputs to go to /kaggle/working/ (writable, downloadable).

    dataset.name in the YAML configs uses short internal keys (cdnow, uci, tafeng,
    dunnhumby).  _KAGGLE_SLUG_MAP translates these to the actual Kaggle dataset slugs,
    which must be ≥7 characters.  Pass data_root (or set KAGGLE_DATA_ROOT) to override
    the /kaggle/input parent dir entirely.

    This function mutates cfg in-place and returns it for convenience.
    It is a no-op when called outside a Kaggle environment (the caller controls
    when to invoke it via --kaggle flag or KAGGLE_ENV=1 env var).
    """
    if data_root is None:
        data_root = os.environ.get("KAGGLE_DATA_ROOT", "/kaggle/input")
    dataset_name = cfg.get("dataset", {}).get("name", "data")
    slug = _KAGGLE_SLUG_MAP.get(dataset_name, dataset_name)
    cfg["dataset"]["raw_dir"] = os.path.join(data_root, slug)
    cfg["output"]["results_dir"] = "/kaggle/working/results"
    return cfg
