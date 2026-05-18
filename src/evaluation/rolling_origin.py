"""
Rolling-origin configuration helpers.

The final holdout remains untouched. These origins carve validation windows out
of each dataset's calibration period so model selection and calibration can use
time-respecting validation data.
"""

from __future__ import annotations

from copy import deepcopy


DEFAULT_ROLLING_ORIGINS = {
    "cdnow": [(26, 13), (39, 13)],
    "uci": [(52, 13), (65, 13)],
    "tafeng": [(8, 2), (10, 2)],
    "dunnhumby": [(58, 11), (69, 11)],
}


def origins_for_dataset(dataset_name: str, configured_origins=None) -> list[tuple[int, int]]:
    """Return `(train_weeks, validation_weeks)` origins for a dataset."""
    origins = configured_origins or DEFAULT_ROLLING_ORIGINS.get(dataset_name, [])
    return [(int(train), int(val)) for train, val in origins]


def rolling_origin_config(base_config: dict, train_weeks: int, validation_weeks: int) -> dict:
    """
    Clone a final-run config into a rolling-origin validation config.

    `calibration_weeks` becomes the rolling training window and `holdout_weeks`
    becomes the validation window immediately following it.
    """
    cfg = deepcopy(base_config)
    cfg.setdefault("dataset", {})["calibration_weeks"] = int(train_weeks)
    cfg["dataset"]["holdout_weeks"] = int(validation_weeks)
    cfg.setdefault("output", {})
    base_name = cfg["output"].get("run_name", "rolling_origin")
    cfg["output"]["run_name"] = f"{base_name}_ro{train_weeks}_{validation_weeks}"
    cfg.setdefault("validation", {})["mode"] = "rolling_origin"
    cfg["validation"]["origin"] = [int(train_weeks), int(validation_weeks)]
    return cfg


def rolling_origin_configs(base_config: dict) -> list[dict]:
    """Expand a config into all configured/default rolling-origin variants."""
    dataset_name = base_config.get("dataset", {}).get("name")
    validation_cfg = base_config.get("validation", {})
    origins = origins_for_dataset(dataset_name, validation_cfg.get("origins"))
    return [rolling_origin_config(base_config, train, val) for train, val in origins]
