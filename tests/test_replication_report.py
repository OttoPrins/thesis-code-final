import json
from pathlib import Path

import numpy as np
import yaml

from run_seeds import DEFAULT_SEEDS
from src.evaluation.metrics import compute_all_metrics, save_metrics_with_artifacts
from src.evaluation.replication_report import (
    build_array_ensemble,
    build_replication_interpretation_table,
    load_replication_decision,
    metrics_path,
)


def _write_seed_metrics(results_dir: Path, run_name: str, pred_week: np.ndarray) -> None:
    true_week = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.float32)
    metrics = compute_all_metrics(
        y_freq_true=true_week.sum(axis=1),
        y_freq_pred=pred_week.sum(axis=1),
        y_freq_true_per_week=true_week,
        y_freq_pred_per_week=pred_week,
        customer_ids=np.arange(true_week.shape[0]),
        freq_mase_scale=1.0,
    )
    metrics.update({
        "model": "lstm_base",
        "dataset": "cdnow",
        "protocol_name": "valendin_cdnow_master_39x39",
        "cohort_size": 23570,
        "calibration_weeks": 39,
        "holdout_weeks": 39,
        "inference_mode": "sample",
        "n_scenarios": 30,
        "run_valid": True,
        "run_invalid_reason": "",
        "run_warning": "",
    })
    save_metrics_with_artifacts(metrics, metrics_path(results_dir, run_name))


def test_valendin_replication_decision_freezes_baseline_and_rejects_keras_exact():
    decision = load_replication_decision()

    assert decision["status"] == "baseline_frozen"
    assert decision["official_baseline"]["config"] == (
        "experiments/configs/lstm_base_cdnow_replication.yaml"
    )
    assert decision["paper_consistent_variant"]["config"] == (
        "experiments/configs/lstm_base_cdnow_replication_paper_finetune.yaml"
    )
    rejected = decision["rejected_ablations"][0]
    assert rejected["config"] == (
        "experiments/configs/lstm_base_cdnow_replication_keras_exact.yaml"
    )
    assert rejected["decision"] == "reject"
    assert abs(rejected["observed_seed42"]["ablation"]["bias_pct"]) > (
        rejected["promotion_rule"]["max_abs_bias_pct"]
    )
    assert rejected["observed_seed42"]["ablation"]["freq_mape"] > (
        rejected["observed_seed42"]["strict"]["freq_mape"]
    )


def test_final_replication_seed_defaults_use_2024_not_123():
    decision = load_replication_decision()

    assert DEFAULT_SEEDS == [42, 7, 2024]
    assert decision["final_evidence_set"]["seeds"] == DEFAULT_SEEDS


def test_build_array_ensemble_averages_saved_per_week_predictions(tmp_path):
    results_dir = tmp_path / "results"
    base_run = "lstm_base_cdnow_replication_demo_pure"
    preds = [
        np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32),
        np.array([[0, 1, 2], [1, 0, 0]], dtype=np.float32),
        np.array([[2, 0, 1], [0, 0, 1]], dtype=np.float32),
    ]
    for seed, pred in zip([42, 7, 2024], preds):
        _write_seed_metrics(results_dir, f"{base_run}_seed{seed}_sample", pred)

    out_path = build_array_ensemble(
        results_dir=results_dir,
        base_run_name=base_run,
        seeds=[42, 7, 2024],
        mode="sample",
        output_run_name=f"{base_run}_ensemble_sample",
    )

    expected = np.mean(np.stack(preds, axis=0), axis=0)
    with np.load(out_path.with_name(out_path.name.replace("_metrics.json", "_arrays.npz"))) as arrays:
        assert np.allclose(arrays["per_week_pred_freq"], expected)
        assert np.allclose(
            arrays["per_customer_pred_freq"],
            expected.sum(axis=1),
        )
    with open(out_path) as f:
        metrics = json.load(f)
    assert metrics["ensemble_seeds"] == [42, 7, 2024]
    assert metrics["ensemble_source"] == "saved_prediction_arrays"
    assert metrics["run_valid"] is True


def test_replication_interpretation_table_can_render_missing_final_runs(tmp_path):
    df = build_replication_interpretation_table(
        results_dir=tmp_path / "empty_results",
        write_outputs=False,
    )

    labels = set(df["label"])
    assert "Valendin et al. (2022) paper target" in labels
    assert "Base LSTM strict 3-seed mean" in labels
    assert "Base LSTM paper-finetune seed ensemble" in labels
    rejected = df[df["role"] == "rejected_ablation"].iloc[0]
    assert rejected["status"] == "missing_rejected"


def test_replication_governance_yaml_commands_are_parseable():
    path = Path("experiments/configs/governance/valendin_replication_decision.yaml")
    cfg = yaml.safe_load(path.read_text())

    assert "lstm_base_cdnow_replication_paper_finetune" in cfg["commands"]["final_seed_sweep"]
    assert "--build-array-ensembles" in cfg["commands"]["build_array_ensembles_and_table"]
    assert cfg["baseline_invariants"]["repair_on_failure"] is False
    assert cfg["baseline_invariants"]["aggregate_scaling"] is False
    assert cfg["baseline_invariants"]["cohort_size"] == 23570
