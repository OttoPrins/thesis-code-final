import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

from src.evaluation.benchmarks import (
    GPPMModel,
    GammaPoissonPropensityModel,
    get_benchmark_model,
    stan_sampler_diagnostics,
)
from src.evaluation.calibration import (
    collect_teacher_forced_validation,
    conservative_ratio_factor,
    fit_aggregate_calibration,
    fit_temperature_from_loader,
)
import src.evaluation.compare as compare_module
from src.evaluation.compare import (
    _filter_metric_files,
    export_latex_table,
    parse_run_name,
    protocol_variant_from_stem,
)
from src.evaluation.metrics import (
    METRIC_DEFINITION_VERSION,
    compute_all_metrics,
    freq_mape,
    horizon_abs_bias_pct,
    mase,
    mase_scale,
    metrics_arrays_path,
    normalized_gini,
    per_week_aggregate_metrics,
    save_metrics_with_artifacts,
    split_metric_artifacts,
)
from src.evaluation.validate_pipeline import validate_pipeline_inputs
from src.models.heads import SpendHead
from src.models.losses import KendallMultiTaskLoss, MetricAlignedFrequencyLoss
from src.models import LSTMModel, TransformerModel
from src.training.callbacks import EarlyStopping
from src.training.inference import _step_from_logits, _zero_inactive_spend_feedback
from src.training.trainer import Trainer
from src.utils.final_manifest import (
    attach_manifest_metadata,
    manifest_config_hash,
    result_matches_manifest,
)
from train import (
    _add_result_validity_checks,
    _add_run_metadata,
    _build_customer_dataloader,
    _build_repair_config,
    _compute_val_smearing,
    _failed_training_metrics,
    _resolve_loader_batch_size,
)


def test_metrics_json_is_scalar_only_and_arrays_go_to_npz(tmp_path):
    metrics = {
        "model": "lstm_joint",
        "freq_rmse": 1.25,
        "_per_customer_freq_se": [1.0, 4.0, 9.0],
    }
    metrics_path = tmp_path / "lstm_joint_cdnow_final_seed42_sample_metrics.json"
    _, arrays_path = save_metrics_with_artifacts(metrics, metrics_path)

    with open(metrics_path) as f:
        saved = json.load(f)
    assert "_per_customer_freq_se" not in saved
    assert saved["freq_rmse"] == 1.25
    assert saved["arrays_file"] == arrays_path.name

    with np.load(metrics_arrays_path(metrics_path)) as data:
        assert np.array_equal(data["per_customer_freq_se"], np.array([1.0, 4.0, 9.0]))


def test_metrics_sidecar_keeps_true_pred_vectors_for_analysis_notebook(tmp_path):
    class IdentitySpendScaler:
        def inverse_transform_spend(self, values):
            return np.asarray(values, dtype=np.float32)

    metrics = compute_all_metrics(
        y_freq_true=np.array([1.0, 3.0]),
        y_freq_pred=np.array([2.0, 2.5]),
        y_spend_true_per_week=np.array([[10.0, 0.0], [5.0, 6.0]], dtype=np.float32),
        y_spend_pred_per_week=np.array([[8.0, 1.0], [7.0, 4.0]], dtype=np.float32),
        customer_ids=np.array([101, 102]),
        scaler=IdentitySpendScaler(),
        weekly_discount_rate=0.0,
    )
    metrics_path = tmp_path / "lstm_joint_cdnow_final_seed42_sample_metrics.json"
    _, arrays_path = save_metrics_with_artifacts(metrics, metrics_path)

    with open(metrics_path) as f:
        saved = json.load(f)
    assert "per_customer_true_clv" not in saved
    assert saved["arrays_file"] == arrays_path.name

    with np.load(arrays_path) as data:
        assert np.array_equal(data["per_customer_true_freq"], np.array([1.0, 3.0]))
        assert np.array_equal(data["per_customer_pred_freq"], np.array([2.0, 2.5]))
        assert np.array_equal(data["per_customer_true_spend"], np.array([10.0, 11.0]))
        assert np.array_equal(data["per_customer_pred_spend"], np.array([9.0, 11.0]))
        assert np.array_equal(data["per_customer_true_clv"], np.array([10.0, 11.0]))
        assert np.array_equal(data["per_customer_pred_clv"], np.array([9.0, 11.0]))
        assert np.array_equal(data["per_week_true_spend"], np.array([[10.0, 0.0], [5.0, 6.0]]))
        assert np.array_equal(data["per_week_pred_spend"], np.array([[8.0, 1.0], [7.0, 4.0]]))


def test_sparse_metrics_include_weekly_scaled_gini_and_decile_artifacts(tmp_path):
    class IdentitySpendScaler:
        def inverse_transform_spend(self, values):
            return np.asarray(values, dtype=np.float32)

    true_freq_week = np.array([[0, 1, 0, 2], [1, 0, 0, 0], [0, 2, 1, 0]], dtype=np.float32)
    pred_freq_week = np.array([[0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0]], dtype=np.float32)
    metrics = compute_all_metrics(
        y_freq_true=true_freq_week.sum(axis=1),
        y_freq_pred=pred_freq_week.sum(axis=1),
        y_freq_true_per_week=true_freq_week,
        y_freq_pred_per_week=pred_freq_week,
        y_spend_true_per_week=np.array([[0, 5, 0, 7], [2, 0, 0, 0], [0, 4, 3, 0]], dtype=np.float32),
        y_spend_pred_per_week=np.array([[0, 4, 1, 6], [0, 0, 0, 0], [1, 3, 4, 0]], dtype=np.float32),
        customer_ids=np.array([1, 2, 3]),
        scaler=IdentitySpendScaler(),
        weekly_discount_rate=0.0,
    )
    assert metrics["metric_definition_version"] == METRIC_DEFINITION_VERSION
    assert metrics["freq_mape"] >= 0
    assert metrics["freq_valendin_mape"] == metrics["freq_mape"]
    assert metrics["freq_horizon_abs_bias_pct"] >= 0
    assert metrics["freq_weekly_mean_pct_mape"] >= 0
    assert np.isfinite(metrics["freq_mase"])
    assert "freq_normalized_gini" in metrics
    assert "spend_normalized_gini" in metrics
    assert "clv_normalized_gini" in metrics

    metrics_path = tmp_path / "joint_metrics.json"
    _, arrays_path = save_metrics_with_artifacts(metrics, metrics_path)
    with np.load(arrays_path) as data:
        assert data["per_week_true_freq"].shape == (3, 4)
        assert data["per_week_pred_freq"].shape == (3, 4)
        assert data["freq_decile_actual_mean"].ndim == 1
        assert data["spend_decile_pred_mean"].ndim == 1
        assert data["clv_decile_actual_lift"].ndim == 1


def test_mase_weekly_metrics_and_normalized_gini_helpers():
    y_true = np.array([[1, 2, 4], [0, 1, 1]], dtype=np.float32)
    assert mase(y_true, y_true) == pytest.approx(0.0)
    assert mase_scale(y_true) == pytest.approx(np.mean([1, 2, 1, 0]))
    assert mase(y_true, y_true + 1, scale=mase_scale(y_true)) == pytest.approx(1 / 1.0)
    weekly = per_week_aggregate_metrics(y_true, y_true, "freq")
    assert weekly["freq_valendin_mape"] == pytest.approx(0.0)
    assert weekly["freq_horizon_abs_bias_pct"] == pytest.approx(0.0)
    assert weekly["freq_weekly_mean_pct_mape"] == pytest.approx(0.0)
    assert normalized_gini(np.array([0, 1, 3]), np.array([0, 1, 3])) == pytest.approx(1.0)
    assert normalized_gini(np.array([0, 1, 3]), np.array([3, 1, 0])) < 0


def test_valendin_mape_and_horizon_bias_are_distinct_hand_computed():
    true = np.array([[2, 0, 1], [0, 3, 0]], dtype=np.float32)
    pred = np.array([[1, 1, 1], [0, 2, 2]], dtype=np.float32)

    # Weekly aggregates are true=[2, 3, 1], pred=[1, 3, 3].
    # Valendin MAPE = 100 * (|2-1| + |3-3| + |1-3|) / (2+3+1) = 50.
    assert freq_mape(true, pred) == pytest.approx(50.0)

    # Full-horizon totals are true=6, pred=7, so absolute horizon bias is 16.67%.
    assert horizon_abs_bias_pct(float(true.sum()), float(pred.sum())) == pytest.approx(
        100 / 6
    )
    weekly = per_week_aggregate_metrics(true, pred, "freq")
    assert weekly["freq_valendin_mape"] == pytest.approx(50.0)
    assert weekly["freq_horizon_abs_bias_pct"] == pytest.approx(100 / 6)
    assert weekly["freq_horizon_bias_pct"] == pytest.approx(100 / 6)
    assert weekly["freq_weekly_mean_pct_mape"] == pytest.approx(
        ((1 / 2) + 0 + (2 / 1)) / 3 * 100
    )


def test_comparison_table_backfills_valendin_mape_from_sidecar_arrays(tmp_path):
    tables_dir = tmp_path / "tables"
    true = np.array([[2, 0, 1], [0, 3, 0]], dtype=np.float32)
    pred = np.array([[1, 1, 1], [0, 2, 2]], dtype=np.float32)
    metrics = {
        "model": "lstm_base",
        "dataset": "cdnow",
        "run_valid": True,
        "freq_rmse": 1.0,
        "freq_mae": 1.0,
        "freq_mape": 999.0,  # stale pre-correction value; must not survive aggregation
        "bias_pct": 100 / 6,
        "_per_week_true_freq": true,
        "_per_week_pred_freq": pred,
    }
    save_metrics_with_artifacts(
        metrics,
        tables_dir / "lstm_base_cdnow_v3_seed42_sample_metrics.json",
    )

    df = compare_module.build_comparison_table(
        str(tmp_path),
        "cdnow",
        final_only=False,
        include_expected=False,
        protocol_variant="all",
    )

    assert len(df) == 1
    assert df.loc[0, "freq_mape"] == pytest.approx(50.0)
    assert df.loc[0, "freq_valendin_mape"] == pytest.approx(50.0)


def test_split_metric_artifacts_rejects_non_scalar_non_numeric_values():
    with pytest.raises(TypeError):
        split_metric_artifacts({"bad": {"nested": "dict"}})


def test_gamma_poisson_is_explicit_diagnostic_not_gppm():
    model = GammaPoissonPropensityModel()
    assert isinstance(model, GammaPoissonPropensityModel)
    assert model.name() == "gamma_poisson"
    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_benchmark_model("gamma_poisson")


def test_true_gppm_requires_weekly_event_log_or_dependencies():
    try:
        model = GPPMModel(iter_sampling=1, iter_warmup=1, chains=1)
    except ImportError as exc:
        assert "cmdstanpy" in str(exc) or "CmdStan" in str(exc)
        return

    rfm = pd.DataFrame({
        "customer_id": [1],
        "frequency": [1],
        "recency": [1.0],
        "T": [10.0],
        "monetary_value": [12.0],
        "litt": [0.0],
    })
    with pytest.raises(RuntimeError, match="weekly event log"):
        model.fit(rfm)


def test_gppm_sampler_diagnostics_gate_invalid_stan_fits():
    class FakeFit:
        def __init__(self, diagnose_text, rhats):
            self._diagnose_text = diagnose_text
            self._rhats = rhats

        def diagnose(self):
            return self._diagnose_text

        def summary(self):
            return pd.DataFrame({"R_hat": self._rhats})

    ok = stan_sampler_diagnostics(
        FakeFit("No divergent transitions found.", [1.0, 1.01])
    )
    assert ok["benchmark_valid"]

    bad = stan_sampler_diagnostics(
        FakeFit("3 of 1000 (0.3%) transitions ended with a divergence.", [1.0, 1.01])
    )
    assert not bad["benchmark_valid"]
    assert bad["stan_divergent_transitions"] == 3

    bad_rhat = stan_sampler_diagnostics(
        FakeFit("No divergent transitions found.", [1.0, 1.2])
    )
    assert not bad_rhat["benchmark_valid"]


def test_deep_run_validity_checks_flag_mechanical_failures():
    metrics = {
        "freq_rmse": 1.0,
        "freq_mae": 1.0,
        "freq_mape": 0.0,
        "bias_pct": 0.0,
    }
    results = {"pred_freq": np.zeros((2, 3), dtype=np.float32)}
    _add_result_validity_checks(
        metrics,
        results=results,
        true_per_week=np.ones((2, 3), dtype=np.float32),
        evaluation_cfg={},
        joint=False,
    )
    assert not metrics["run_valid"]
    assert "all-zero frequency forecast" in metrics["run_invalid_reason"]

    metrics = {
        "freq_rmse": 1.0,
        "freq_mae": 1.0,
        "freq_mape": 0.0,
        "bias_pct": 0.0,
        "spend_mae_raw": 1.0,
        "spend_rmse_raw": 1.0,
        "spend_bias_pct": 0.0,
        "spend_weekly_total_true": 10.0,
        "spend_weekly_total_pred": 101.0,
    }
    results = {
        "pred_freq": np.ones((2, 3), dtype=np.float32),
        "pred_activity": np.ones((2, 3), dtype=np.float32),
        "pred_spend": np.ones((2, 3), dtype=np.float32),
    }
    _add_result_validity_checks(
        metrics,
        results=results,
        true_per_week=np.ones((2, 3), dtype=np.float32),
        evaluation_cfg={"validity_max_spend_total_ratio": 10.0},
        joint=True,
    )
    assert not metrics["run_valid"]
    assert "spend total ratio" in metrics["run_invalid_reason"]


def _repair_base_config(tmp_path):
    return {
        "dataset": {"name": "cdnow"},
        "model": {"type": "lstm", "joint": True},
        "training": {
            "seed": 42,
            "lr": 1e-3,
            "batch_size": 128,
            "max_grad_norm": 1.0,
        },
        "inference": {"mode": "sample", "n_scenarios": 20},
        "output": {"run_name": "lstm_joint_cdnow_v2_seed42_sample", "results_dir": str(tmp_path)},
    }


def test_repair_config_is_bounded_and_separate_from_headline_run(tmp_path):
    cfg = _repair_base_config(tmp_path)
    repaired = _build_repair_config(cfg, RuntimeError("CUDA out of memory"))

    assert repaired["training"]["amp_enabled"] is False
    assert repaired["training"]["lr"] == pytest.approx(5e-4)
    assert repaired["training"]["max_grad_norm"] == pytest.approx(0.5)
    assert repaired["training"]["batch_size"] == 64
    assert repaired["training"]["repair_attempt"] is True
    assert repaired["output"]["run_name"].endswith("_repair")
    assert cfg["output"]["run_name"] == "lstm_joint_cdnow_v2_seed42_sample"


def test_failed_training_metrics_are_invalid_and_auditable(tmp_path):
    cfg = _repair_base_config(tmp_path)
    metrics = _failed_training_metrics(
        cfg,
        config_path="experiments/configs/lstm_joint_cdnow_v2.yaml",
        exc=FloatingPointError("Non-finite gradients in lstm.weight_ih_l0"),
        repair_attempted=True,
        repair_error=FloatingPointError("Non-finite training loss: nan"),
    )

    assert metrics["run_valid"] is False
    assert metrics["training_failed"] is True
    assert metrics["repair_attempted"] is True
    assert metrics["repair_failed"] is True
    assert "training failed" in metrics["run_invalid_reason"]
    assert "repair failed" in metrics["run_invalid_reason"]


def test_validation_calibration_uses_top_bin_class_values():
    class ConstantTopBinLSTM(LSTMModel):
        def __init__(self):
            super().__init__(max_week=3, max_trans=3, memory_units=4, dense_units=4, joint=False)

        def forward(self, week, trans, hidden=None, **kwargs):
            B, T = week.shape
            logits = torch.full((B, T, 4), -20.0, dtype=torch.float32, device=week.device)
            logits[..., -1] = 20.0
            return logits, hidden

    batch = {
        "week": torch.tensor([[0, 1]], dtype=torch.long),
        "trans": torch.tensor([[0, 0]], dtype=torch.long),
        "mask": torch.ones((1, 2), dtype=torch.float32),
        "y_freq": torch.zeros((1, 2), dtype=torch.long),
    }
    totals = collect_teacher_forced_validation(
        ConstantTopBinLSTM(),
        [batch],
        device=torch.device("cpu"),
        class_values=torch.tensor([0.0, 1.0, 2.0, 9.0]),
    )

    assert totals["freq_pred_total"] == pytest.approx(18.0)


def test_comparison_filter_skips_stale_deep_results_without_run_valid(tmp_path):
    stale = tmp_path / "lstm_joint_cdnow_final_seed42_sample_metrics.json"
    stale.write_text(json.dumps({"freq_rmse": 1.0}))
    valid = tmp_path / "lstm_joint_cdnow_final_seed7_sample_metrics.json"
    valid.write_text(json.dumps({"freq_rmse": 1.0, "run_valid": True}))
    bench = tmp_path / "pareto_nbd_cdnow_metrics.json"
    bench.write_text(json.dumps({"freq_rmse": 1.0, "model": "pareto_nbd"}))
    stale_gppm = tmp_path / "gppm_cdnow_metrics.json"
    stale_gppm.write_text(json.dumps({"freq_rmse": 1.0, "model": "gppm"}))
    valid_gppm = tmp_path / "gppm_cdnow_valid_metrics.json"
    valid_gppm.write_text(json.dumps({
        "freq_rmse": 1.0,
        "model": "gppm",
        "benchmark_valid": True,
    }))

    kept = _filter_metric_files(
        [str(stale), str(valid), str(bench), str(stale_gppm), str(valid_gppm)],
        final_only=False,
    )
    assert str(stale) not in kept
    assert str(stale_gppm) not in kept
    assert str(valid) in kept
    assert str(bench) in kept
    assert str(valid_gppm) not in kept


def test_final_filter_requires_deep_arrays_and_checkpoint(tmp_path, monkeypatch):
    tables = tmp_path / "tables"
    checkpoints = tmp_path / "checkpoints"
    tables.mkdir()
    checkpoints.mkdir()
    metrics_path = tables / "lstm_base_cdnow_v2_seed42_sample_metrics.json"
    metrics_path.write_text(json.dumps({
        "freq_rmse": 1.0,
        "run_valid": True,
        "arrays_file": "lstm_base_cdnow_v2_seed42_sample_arrays.npz",
    }))

    monkeypatch.setattr(compare_module, "load_final_manifest", lambda: {"benchmark_run_names": []})
    monkeypatch.setattr(
        compare_module,
        "result_matches_manifest",
        lambda *args, **kwargs: (True, "ok"),
    )

    assert _filter_metric_files([str(metrics_path)], final_only=True) == []

    np.savez(tables / "lstm_base_cdnow_v2_seed42_sample_arrays.npz", x=np.array([1.0]))
    assert _filter_metric_files([str(metrics_path)], final_only=True) == []

    (checkpoints / "lstm_base_cdnow_v2_seed42_sample.pt").write_bytes(b"checkpoint")
    assert _filter_metric_files([str(metrics_path)], final_only=True) == [str(metrics_path)]


def test_repair_run_parsing_and_filtering_stays_separate(tmp_path):
    strict = tmp_path / "transformer_joint_tafeng_v2_seed42_sample_metrics.json"
    strict.write_text(json.dumps({"freq_rmse": 1.0, "run_valid": True}))
    repaired = tmp_path / "transformer_joint_tafeng_v2_seed42_sample_repair_metrics.json"
    repaired.write_text(json.dumps({
        "freq_rmse": 0.9,
        "run_valid": True,
        "repair_attempted": True,
    }))
    bad_repair = tmp_path / "lstm_joint_tafeng_v2_seed42_sample_repair_metrics.json"
    bad_repair.write_text(json.dumps({"freq_rmse": 0.8, "run_valid": True}))

    assert parse_run_name("transformer_joint_tafeng_v2_seed42_sample_repair") == (
        "transformer_joint",
        "tafeng",
        "v2",
        42,
        "sample",
    )
    assert protocol_variant_from_stem("transformer_joint_tafeng_v2_seed42_sample_repair") == "stability_repair"

    strict_kept = _filter_metric_files(
        [str(strict), str(repaired), str(bad_repair)],
        final_only=False,
        protocol_variant="strict",
    )
    repaired_kept = _filter_metric_files(
        [str(strict), str(repaired), str(bad_repair)],
        final_only=False,
        protocol_variant="repaired",
    )

    assert strict_kept == [str(strict)]
    assert repaired_kept == [str(repaired)]


def test_lstm_is_one_layer_seq2seq_with_shared_joint_encoder():
    torch.manual_seed(0)
    model = LSTMModel(max_week=51, max_trans=3, memory_units=8, dense_units=8, joint=True)
    model.eval()
    week = torch.arange(5).unsqueeze(0).repeat(2, 1)
    trans = torch.zeros((2, 5), dtype=torch.long)
    spend_history = torch.zeros((2, 5), dtype=torch.float32)
    richer_spend_history = torch.full((2, 5), 2.0, dtype=torch.float32)
    logits, spend, _ = model(week, trans, spend=spend_history)
    richer_logits, richer_spend, _ = model(week, trans, spend=richer_spend_history)
    assert model.lstm.num_layers == 1
    assert logits.shape == (2, 5, 4)
    assert spend.shape == (2, 5)
    assert not torch.allclose(logits, richer_logits)
    assert not torch.allclose(spend, richer_spend)


def test_lstm_v2_hurdle_head_returns_mu_and_log_var():
    torch.manual_seed(0)
    model = LSTMModel(
        max_week=51,
        max_trans=3,
        memory_units=8,
        dense_units=8,
        joint=True,
        spend_head="hurdle_lognormal",
        state_feature_dim=1,
    )
    week = torch.arange(5).unsqueeze(0)
    trans = torch.zeros((1, 5), dtype=torch.long)
    spend_history = torch.zeros((1, 5), dtype=torch.float32)
    state_features = torch.zeros((1, 5, 1), dtype=torch.float32)
    logits, spend_mu, spend_log_var, _ = model(
        week, trans, spend=spend_history, state_features=state_features
    )
    assert logits.shape == (1, 5, 4)
    assert spend_mu.shape == (1, 5)
    assert spend_log_var.shape == (1, 5)


def test_transformer_uses_position_and_time2vec_only_for_temporal_encoding():
    torch.manual_seed(0)
    model = TransformerModel(
        max_week=51,
        max_trans=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        time2vec_dim=4,
        joint=True,
    )
    model.eval()
    week = torch.tensor([[50, 51, 0, 1]])
    position = torch.tensor([[50, 51, 52, 53]])
    trans = torch.zeros_like(week)
    delta_t = torch.zeros_like(week, dtype=torch.float32)
    spend_history = torch.zeros_like(delta_t)
    richer_spend_history = torch.full_like(delta_t, 2.0)

    logits, spend = model(
        week, trans, spend=spend_history, position=position, delta_t=delta_t
    )
    richer_logits, richer_spend = model(
        week, trans, spend=richer_spend_history, position=position, delta_t=delta_t
    )
    assert logits.shape == (1, 4, 4)
    assert spend.shape == (1, 4)
    assert not torch.allclose(logits, richer_logits)
    assert not torch.allclose(spend, richer_spend)
    assert hasattr(model, "time2vec")
    assert hasattr(model, "pos_enc")
    assert not hasattr(model, "tape")
    assert not hasattr(model, "erpe")


def _tiny_transformer_batch():
    return {
        "week": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
        "position": torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long),
        "trans": torch.tensor([[0, 1, 0, 2, 0], [0, 0, 1, 0, 0]], dtype=torch.long),
        "spend": torch.tensor([[0.0, 1.0, 0.0, 1.2, 0.0], [0.0, 0.0, 0.8, 0.0, 0.0]]),
        "delta_t": torch.tensor([[1, 0, 1, 0, 1], [1, 2, 0, 1, 2]], dtype=torch.float32),
        "mask": torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=torch.float32),
        "y_freq": torch.tensor([[1, 0, 2, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.long),
        "y_spend": torch.tensor([[1.0, 0.0, 1.1, 0.0, 0.0], [0.0, 0.7, 0.0, 0.0, 0.0]]),
    }


def _run_transformer_finite_smoke(device: torch.device):
    torch.manual_seed(0)
    model = TransformerModel(
        max_week=51,
        max_trans=3,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        time2vec_dim=4,
        joint=True,
    ).to(device)
    loss_fn = KendallMultiTaskLoss(n_tasks=2).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=1e-3,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        joint=True,
        multi_task_loss=loss_fn,
        kendall_warmup_epochs=0,
        max_grad_norm=1.0,
    )
    for _ in range(3):
        metrics = trainer.train_epoch([_tiny_transformer_batch()])
        assert np.isfinite(metrics["total_loss"])


def test_hurdle_spend_loss_ignores_inactive_targets():
    model = LSTMModel(
        max_week=51,
        max_trans=1,
        memory_units=4,
        dense_units=4,
        joint=True,
        spend_head="hurdle_lognormal",
    )
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        joint=True,
        multi_task_loss=KendallMultiTaskLoss(n_tasks=2),
        spend_loss="nll",
    )
    loss = trainer._compute_spend_loss(
        spend_mu=torch.tensor([[99.0, 1.0]]),
        y_spend=torch.tensor([[0.0, 1.0]]),
        mask=torch.ones((1, 2)),
        active_mask=torch.tensor([[0.0, 1.0]]),
        spend_log_var=torch.zeros((1, 2)),
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_metric_aligned_frequency_loss_is_finite_and_respects_mask():
    logits = torch.tensor(
        [
            [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]],
            [[0.0, 0.0, 0.0, 4.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]],
        ],
        requires_grad=True,
    )
    y_freq = torch.tensor([[0, 1, 2], [3, 0, 3]])
    y_raw = torch.tensor([[0.0, 1.0, 2.0], [5.0, 0.0, 99.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    loss_fn = MetricAlignedFrequencyLoss(
        {
            "enabled": True,
            "class_weights": "auto",
            "activity_bce_weight": 0.2,
            "count_huber_weight": 0.5,
            "aggregate_bias_weight": 0.3,
            "temporal_tail_weight": 0.4,
        },
        class_values=torch.tensor([0.0, 1.0, 2.0, 5.0]),
    )

    loss, metrics = loss_fn(logits, y_freq, mask, y_freq_raw=y_raw)
    assert torch.isfinite(loss)
    assert metrics["freq_count_huber_loss"] < 5.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad[1, 2] == 0)


def test_transformer_scheduled_noising_uses_only_previous_predictions():
    model = TransformerModel(
        max_week=51,
        max_trans=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        time2vec_dim=2,
        joint=False,
    )
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        joint=False,
        scheduled_sampling={"enabled": True, "start_epoch": 0, "max_prob": 1.0},
    )
    batch = {
        "week": torch.tensor([[0, 1, 2, 3]]),
        "position": torch.tensor([[0, 1, 2, 3]]),
        "trans": torch.tensor([[0, 0, 0, 0]]),
        "spend": torch.zeros((1, 4)),
        "state_features": torch.zeros((1, 4, 1)),
        "delta_t": torch.zeros((1, 4)),
        "mask": torch.ones((1, 4)),
        "y_freq": torch.tensor([[0, 0, 0, 0]]),
    }
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 3] = 10.0
    logits[0, 3, 0] = 10.0

    noisy = trainer._build_scheduled_noising_batch(batch, logits, None, ss_prob=1.0)
    assert noisy["trans"].tolist() == [[0, 1, 2, 3]]


def _last_linear(module: nn.Module) -> nn.Linear:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    assert linears
    return linears[-1]


def test_spend_head_uses_small_output_initialization():
    torch.manual_seed(0)
    head = SpendHead(4096)
    assert float(head.fc.weight.std()) == pytest.approx(0.01, rel=0.08)
    assert torch.count_nonzero(head.fc.bias).item() == 0


def test_live_model_spend_heads_use_small_output_initialization():
    torch.manual_seed(0)
    lstm = LSTMModel(
        max_week=51,
        max_trans=3,
        memory_units=8,
        dense_units=1024,
        joint=True,
        spend_head="hurdle_lognormal",
    )
    lstm_out = _last_linear(lstm.spend_head)
    assert float(lstm_out.weight.std()) == pytest.approx(0.01, rel=0.12)
    assert torch.count_nonzero(lstm_out.bias).item() == 0

    transformer = TransformerModel(
        max_week=51,
        max_trans=3,
        d_model=512,
        n_heads=8,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        time2vec_dim=4,
        joint=True,
    )
    transformer_out = _last_linear(transformer.spend_head)
    assert float(transformer_out.weight.std()) == pytest.approx(0.01, rel=0.15)
    assert torch.count_nonzero(transformer_out.bias).item() == 0


def test_lstm_replication_mode_is_linear_and_keras_initialized():
    torch.manual_seed(0)
    model = LSTMModel(
        max_week=51,
        max_trans=3,
        memory_units=8,
        dense_units=8,
        dropout=0.0,
        dense_activation="linear",
        keras_initialization=True,
    )

    assert isinstance(model.dense_activation, nn.Identity)
    probe = torch.tensor([-1.0, 0.5])
    assert torch.equal(model.dense_activation(probe), probe)
    assert float(model.embed_week.weight.abs().max()) <= 0.050001
    assert float(model.embed_trans.weight.abs().max()) <= 0.050001

    hidden = model.lstm.hidden_size
    assert torch.allclose(
        model.lstm.bias_ih_l0[hidden:2 * hidden],
        torch.ones(hidden),
    )
    assert torch.count_nonzero(model.lstm.bias_ih_l0[:hidden]).item() == 0
    assert torch.count_nonzero(model.lstm.bias_ih_l0[2 * hidden:]).item() == 0
    assert torch.count_nonzero(model.lstm.bias_hh_l0).item() == 0


def test_lstm_keras_cardinality_embedding_and_whole_matrix_init():
    torch.manual_seed(0)
    model = LSTMModel(
        max_week=51,
        max_trans=24,
        memory_units=8,
        dense_units=8,
        dropout=0.0,
        dense_activation="linear",
        keras_initialization=True,
        embedding_size_mode="keras_cardinality",
        keras_lstm_init_mode="whole_matrix",
        freeze_lstm_bias_hh=True,
    )

    assert model.embed_week.embedding_dim == 8
    assert model.embed_trans.embedding_dim == 6

    hidden = model.lstm.hidden_size
    assert torch.allclose(
        model.lstm.bias_ih_l0[hidden:2 * hidden],
        torch.ones(hidden),
    )
    assert torch.count_nonzero(model.lstm.bias_hh_l0).item() == 0
    assert model.lstm.bias_hh_l0.requires_grad is False

    week = torch.zeros((2, 3), dtype=torch.long)
    trans = torch.zeros((2, 3), dtype=torch.long)
    logits, _ = model(week, trans)
    assert logits.shape == (2, 3, 25)


def test_replication_loader_parity_switches_are_explicit():
    assert _resolve_loader_batch_size(
        "full",
        default=32,
        dataset_len=2357,
        name="training.val_batch_size",
    ) == 2357
    assert _resolve_loader_batch_size(
        None,
        default=32,
        dataset_len=2357,
        name="training.inference_batch_size",
    ) == 32
    assert _resolve_loader_batch_size(
        "val",
        default=32,
        dataset_len=23570,
        name="training.inference_batch_size",
        reference_sizes={"val": 2357},
    ) == 2357

    dataset = list(range(10))
    keep_partial = _build_customer_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate=lambda rows: rows,
    )
    drop_partial = _build_customer_dataloader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate=lambda rows: rows,
    )

    assert list(keep_partial) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert list(drop_partial) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_scheduled_sampling_forward_equals_teacher_forcing_at_zero_prob():
    """At ε=0 the stepwise SS unroll must equal the vectorized teacher-forced
    forward (LSTM recurrence equivalence) — guarantees SS adds no bias when off."""
    torch.manual_seed(0)
    model = LSTMModel(max_week=51, max_trans=3, memory_units=8, dense_units=8, dropout=0.0)
    model.eval()  # disable dropout for an exact comparison
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        joint=False,
    )
    batch = {
        "week": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "trans": torch.tensor([[0, 1, 0, 2], [1, 0, 0, 3]]),
        "mask": torch.ones((2, 4)),
        "y_freq": torch.tensor([[1, 0, 2, 0], [0, 0, 3, 1]]),
    }
    tf_logits, _, _ = trainer._forward(batch)
    ss_logits, _, _ = trainer._scheduled_sampling_forward_lstm(batch, ss_prob=0.0)
    assert ss_logits.shape == tf_logits.shape
    assert torch.allclose(ss_logits, tf_logits, atol=1e-5)


def test_ss_prob_gating_and_ramp():
    """ε is 0 when disabled / before start / for non-LSTM, and ramps to max_prob."""
    model = LSTMModel(max_week=51, max_trans=3, memory_units=4, dense_units=4)
    model.train()
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        device=torch.device("cpu"),
        joint=False,
        scheduled_sampling={"enabled": True, "start_epoch": 5, "max_prob": 0.3,
                            "schedule": "linear"},
    )
    trainer._total_epochs = 25
    trainer.current_epoch = 0
    assert trainer._ss_prob() == 0.0          # before start_epoch
    trainer.current_epoch = 5
    assert trainer._ss_prob() == pytest.approx(0.0, abs=1e-9)  # at start
    trainer.current_epoch = 25
    assert trainer._ss_prob() == pytest.approx(0.3, rel=1e-6)  # capped at max_prob
    trainer.current_epoch = 15
    assert 0.0 < trainer._ss_prob() < 0.3     # ramping
    model.eval()                              # SS only during training
    assert trainer._ss_prob() == 0.0
    trainer.scheduled_sampling = None         # disabled → off
    model.train()
    assert trainer._ss_prob() == 0.0


def test_scheduled_sampling_joint_fit_is_finite_end_to_end():
    """Joint hurdle LSTM trained via fit() with scheduled sampling enabled across
    the start_epoch boundary must stay finite (exercises the stepwise unroll +
    spend feedback + Kendall + backward end-to-end)."""
    torch.manual_seed(0)
    model = LSTMModel(
        max_week=51, max_trans=3, memory_units=8, dense_units=8,
        joint=True, spend_head="hurdle_lognormal", state_feature_dim=1,
    )
    loss_fn = KendallMultiTaskLoss(n_tasks=2, spend_logvar_max=2.0)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters())),
        device=torch.device("cpu"),
        joint=True,
        multi_task_loss=loss_fn,
        spend_loss="nll",
        kendall_warmup_epochs=1,
        scheduled_sampling={"enabled": True, "start_epoch": 1, "max_prob": 0.5,
                            "schedule": "linear"},
    )
    batch = {
        "week": torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]),
        "trans": torch.tensor([[0, 1, 0, 2], [1, 0, 0, 3]]),
        "spend": torch.tensor([[0.0, 1.2, 0.0, 0.7], [0.9, 0.0, 0.0, 1.5]]),
        "state_features": torch.zeros((2, 4, 1)),
        "mask": torch.ones((2, 4)),
        "y_freq": torch.tensor([[1, 0, 2, 0], [0, 0, 3, 1]]),
        "y_spend": torch.tensor([[1.0, 0.0, 0.8, 0.0], [0.0, 0.0, 1.3, 0.5]]),
        "active_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
    }
    history = trainer.fit(
        train_loader=[batch], val_loader=[batch], epochs=4,
        early_stopping=EarlyStopping(patience=10),
    )
    assert all(np.isfinite(history["train_loss"]))
    assert all(np.isfinite(history["val_loss"]))
    assert len(history["train_loss"]) == 4


def test_trainer_skips_nonfinite_training_losses_until_limit(tmp_path):
    class OneParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    class ScriptedLossTrainer(Trainer):
        def __init__(self, *args, losses, **kwargs):
            super().__init__(*args, **kwargs)
            self.losses = list(losses)

        def _compute_loss(self, batch):
            value = self.losses.pop(0)
            loss = self.model.weight.sum() * 0 + torch.tensor(value)
            return loss, {"freq_loss": float(value)}

    model = OneParamModel()
    trainer = ScriptedLossTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        joint=False,
        losses=[float("nan"), float("nan"), 1.0],
        dataset_name="cdnow",
        run_id="nan_test",
        checkpoint_dir=tmp_path,
    )
    metrics = trainer.train_epoch([object(), object(), object()])
    assert metrics["freq_loss"] == pytest.approx(1.0)
    assert trainer._nan_count == 2


def test_trainer_aborts_after_six_nonfinite_losses_and_saves_checkpoint(tmp_path):
    class OneParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    class NaNTrainer(Trainer):
        def _compute_loss(self, batch):
            loss = self.model.weight.sum() * 0 + torch.tensor(float("nan"))
            return loss, {"freq_loss": float("nan")}

    model = OneParamModel()
    trainer = NaNTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        joint=False,
        dataset_name="cdnow",
        run_id="abort_test",
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match=">5 NaN losses"):
        trainer.train_epoch([object()] * 6)
    assert (tmp_path / "last_valid_cdnow_abort_test.pt").exists()


def _validation_gate_batch(y_spend=None):
    return {
        "week": torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        "trans": torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.long),
        "spend": torch.zeros((2, 3), dtype=torch.float32),
        "dynamic_covariates": torch.zeros((2, 5, 1), dtype=torch.float32),
        "y_spend": (
            y_spend
            if y_spend is not None
            else torch.zeros((2, 3), dtype=torch.float32)
        ),
    }


def test_validate_pipeline_inputs_rejects_bad_inputs_and_outputs(tmp_path):
    model = LSTMModel(
        max_week=51, max_trans=3, memory_units=4, dense_units=4, joint=True,
        dynamic_cov_dim=1,
    )
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)
    config = {"model": {"type": "lstm", "joint": True}}

    assert validate_pipeline_inputs(
        model, [_validation_gate_batch()], config, ckpt, torch.device("cpu")
    )

    bad_spend = torch.zeros((2, 3), dtype=torch.float32)
    bad_spend[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite spend targets"):
        validate_pipeline_inputs(
            model, [_validation_gate_batch(bad_spend)], config, ckpt, torch.device("cpu")
        )

    with pytest.raises(ValueError, match="Checkpoint does not exist"):
        validate_pipeline_inputs(
            model, [_validation_gate_batch()], config, tmp_path / "missing.pt", torch.device("cpu")
        )

    class BadSoftmaxLSTM(LSTMModel):
        def forward(self, week, trans, hidden=None, **kwargs):
            b, t = week.shape
            logits = torch.full((b, t, self.max_trans + 1), float("nan"))
            return logits, torch.zeros((b, t)), hidden

    bad_softmax = BadSoftmaxLSTM(
        max_week=51, max_trans=3, memory_units=4, dense_units=4, joint=True
    )
    bad_softmax_ckpt = tmp_path / "bad_softmax.pt"
    torch.save(bad_softmax.state_dict(), bad_softmax_ckpt)
    with pytest.raises(ValueError, match="Softmax sanity check failed"):
        validate_pipeline_inputs(
            bad_softmax,
            [_validation_gate_batch()],
            config,
            bad_softmax_ckpt,
            torch.device("cpu"),
        )

    class BadSpendLSTM(LSTMModel):
        def forward(self, week, trans, hidden=None, **kwargs):
            b, t = week.shape
            logits = torch.zeros((b, t, self.max_trans + 1))
            return logits, torch.full((b, t), float("nan")), hidden

    bad_spend_model = BadSpendLSTM(
        max_week=51, max_trans=3, memory_units=4, dense_units=4, joint=True
    )
    bad_spend_ckpt = tmp_path / "bad_spend.pt"
    torch.save(bad_spend_model.state_dict(), bad_spend_ckpt)
    with pytest.raises(ValueError, match="Non-finite regression head output"):
        validate_pipeline_inputs(
            bad_spend_model,
            [_validation_gate_batch()],
            config,
            bad_spend_ckpt,
            torch.device("cpu"),
        )


def _teacher_forced_full_dynamic_batch():
    return {
        "week": torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
        "position": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        "trans": torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.long),
        "spend": torch.zeros((2, 3), dtype=torch.float32),
        "state_features": torch.zeros((2, 3, 1), dtype=torch.float32),
        "delta_t": torch.zeros((2, 3), dtype=torch.float32),
        "dynamic_covariates": torch.ones((2, 5, 1), dtype=torch.float32),
        "mask": torch.ones((2, 3), dtype=torch.float32),
        "y_freq": torch.tensor([[0, 1, 0], [1, 0, 2]], dtype=torch.long),
        "y_spend": torch.zeros((2, 3), dtype=torch.float32),
    }


class ShapeCheckingLSTM(LSTMModel):
    def forward(self, week, trans, hidden=None, **kwargs):
        dynamic_cov = kwargs.get("dynamic_covariates")
        if dynamic_cov is not None:
            assert dynamic_cov.shape[1] == week.shape[1]
        return super().forward(week, trans, hidden=hidden, **kwargs)


class ShapeCheckingTransformer(TransformerModel):
    def forward(self, week, trans, **kwargs):
        dynamic_cov = kwargs.get("dynamic_covariates")
        if dynamic_cov is not None:
            assert dynamic_cov.shape[1] == week.shape[1]
        return super().forward(week, trans, **kwargs)


class IdentityLogScaler:
    def inverse_transform_log1p(self, values):
        return np.asarray(values, dtype=np.float32)


def test_teacher_forced_paths_slice_full_dynamic_covariates():
    model = ShapeCheckingLSTM(
        max_week=51,
        max_trans=3,
        memory_units=4,
        dense_units=4,
        joint=True,
        dynamic_cov_dim=1,
        state_feature_dim=1,
    )
    batch = _teacher_forced_full_dynamic_batch()
    loss_fn = KendallMultiTaskLoss(n_tasks=2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters())),
        device=torch.device("cpu"),
        joint=True,
        multi_task_loss=loss_fn,
    )

    logits, spend_mu, _ = trainer._forward(batch)
    assert logits.shape[:2] == batch["week"].shape
    assert spend_mu.shape == batch["week"].shape

    totals = collect_teacher_forced_validation(
        model,
        [batch],
        device=torch.device("cpu"),
        scaler=None,
    )
    assert totals["freq_true_total"] == pytest.approx(4.0)

    temp = fit_temperature_from_loader(model, [batch], device=torch.device("cpu"), max_iter=2)
    assert np.isfinite(temp)

    smear = _compute_val_smearing(
        model,
        [batch],
        torch.device("cpu"),
        joint=True,
        scaler=IdentityLogScaler(),
    )
    assert np.isfinite(smear)


def test_teacher_forced_transformer_paths_slice_full_dynamic_covariates():
    model = ShapeCheckingTransformer(
        max_week=51,
        max_trans=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
        time2vec_dim=4,
        joint=True,
        dynamic_cov_dim=1,
        state_feature_dim=1,
    )
    batch = _teacher_forced_full_dynamic_batch()
    loss_fn = KendallMultiTaskLoss(n_tasks=2)
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters())),
        device=torch.device("cpu"),
        joint=True,
        multi_task_loss=loss_fn,
    )

    logits, spend_mu, _ = trainer._forward(batch)
    assert logits.shape[:2] == batch["week"].shape
    assert spend_mu.shape == batch["week"].shape

    totals = collect_teacher_forced_validation(
        model,
        [batch],
        device=torch.device("cpu"),
        scaler=None,
    )
    assert totals["freq_true_total"] == pytest.approx(4.0)

    temp = fit_temperature_from_loader(model, [batch], device=torch.device("cpu"), max_iter=2)
    assert np.isfinite(temp)

    smear = _compute_val_smearing(
        model,
        [batch],
        torch.device("cpu"),
        joint=True,
        scaler=IdentityLogScaler(),
    )
    assert np.isfinite(smear)


def test_final_configs_freeze_valendin_fixed_tuning_protocol():
    manifest = yaml.safe_load(open("experiments/final_manifest.yaml"))
    final_configs = manifest["deep_learning_configs"]
    manifest_hashes = manifest["deep_learning_config_hashes"]
    assert len(final_configs) == 20

    for path in final_configs:
        cfg = yaml.safe_load(open(path))
        assert manifest_hashes[path] == manifest_config_hash(cfg), path
        assert path.startswith("experiments/configs_final/"), path

        dataset = cfg["dataset"]
        model = cfg["model"]
        training = cfg["training"]
        inference = cfg["inference"]

        assert dataset["val_fraction"] == pytest.approx(0.1), path
        assert dataset["split_seed"] == 42, path
        assert training["epochs"] == 150, path
        assert training["max_epochs"] == 150, path
        assert training["early_stopping_patience"] == 20, path
        assert training["restore_best_checkpoint"] is True, path
        assert training["final_finetune"]["enabled"] is True, path
        assert training["amp_enabled"] is False, path
        assert inference["mode"] == "sample", path
        assert inference["n_scenarios"] == 30, path

        if model["type"] == "transformer":
            assert model["d_model"] == 128, path
            assert model["n_heads"] == 4, path
            assert training["lr"] == pytest.approx(5e-4), path
            assert training["max_grad_norm"] == pytest.approx(1.0), path
        else:
            assert model["hidden_size"] == 128, path
            assert model["dense_units"] == 128, path
            assert training["lr"] == pytest.approx(1e-3), path
            if model.get("joint"):
                assert training["max_grad_norm"] == pytest.approx(1.0), path
            else:
                assert training["max_grad_norm"] == pytest.approx(0.0), path

        if dataset["name"] == "cdnow":
            assert dataset["protocol_name"] == "valendin_cdnow_master_39x39", path
            assert dataset["calibration_weeks"] == 39, path
            assert dataset["holdout_weeks"] == 39, path
            assert dataset["calendar_mode"] == "valendin_year_week", path
            assert dataset["keep_zero_amount_transactions"] is True, path


def test_obsolete_cdnow_v4_soft_configs_are_not_in_final_manifest():
    manifest = yaml.safe_load(open("experiments/final_manifest.yaml"))
    assert not any(
        "v4_soft" in path for path in manifest["deep_learning_configs"]
    )


def test_hpo_objective_penalizes_invalid_and_degenerate_runs():
    """The HPO objective must (a) hard-penalize invalid runs and (b) NOT reward a
    near-zero-bias model that has collapsed per-customer discrimination (the
    degenerate trap seen in the SS smoke: bias≈0 but gini≈0)."""
    from tune import _objective_value

    # Invalid run → max penalty regardless of other metrics.
    assert _objective_value({"run_valid": False, "bias_pct": 0.0}, joint=False) == 10.0

    # Degenerate: tiny bias but no discrimination (gini≈0) must score WORSE than
    # a model with modest bias but real discrimination (gini high).
    degenerate = _objective_value(
        {"run_valid": True, "bias_pct": 1.0, "freq_normalized_gini": 0.0}, joint=False
    )
    skilled = _objective_value(
        {"run_valid": True, "bias_pct": 10.0, "freq_normalized_gini": 0.65}, joint=False
    )
    assert skilled < degenerate

    # Joint objective rewards positive spend R² and CLV Spearman.
    worse_spend = _objective_value(
        {"run_valid": True, "bias_pct": 5.0, "freq_normalized_gini": 0.6,
         "spend_r2_log": -0.6, "clv_spearman": 0.4}, joint=True
    )
    better_spend = _objective_value(
        {"run_valid": True, "bias_pct": 5.0, "freq_normalized_gini": 0.6,
         "spend_r2_log": 0.45, "clv_spearman": 0.6}, joint=True
    )
    assert better_spend < worse_spend


def test_short_diagnostic_run_metadata_marks_smoke_run():
    metrics = {"diagnostic_only": True, "run_warning": "large frequency bias (42.0%)"}

    _add_run_metadata(
        metrics,
        history={"train_loss": [1.0, 0.9, 0.8], "val_loss": [1.1, 1.0, 0.95]},
        training_cfg={"max_epochs": 3},
        evaluation_cfg={},
    )

    assert metrics["diagnostic_only"] is True
    assert metrics["epochs_completed"] == 3
    assert metrics["max_epochs_configured"] == 3
    assert metrics["smoke_run"] is True
    assert "smoke run only" in metrics["run_warning"]
    assert "large frequency bias" in metrics["run_warning"]


def test_full_run_metadata_does_not_mark_smoke_run():
    metrics = {}

    _add_run_metadata(
        metrics,
        history={"train_loss": [1.0] * 20},
        training_cfg={"max_epochs": 60},
        evaluation_cfg={},
    )

    assert metrics["epochs_completed"] == 20
    assert metrics["max_epochs_configured"] == 60
    assert "smoke_run" not in metrics


def test_transformer_cpu_training_smoke_is_finite_with_loss_mask():
    _run_transformer_finite_smoke(torch.device("cpu"))


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS backend is not available",
)
def test_transformer_mps_training_smoke_is_finite_with_loss_mask():
    _run_transformer_finite_smoke(torch.device("mps"))


def test_sample_mode_activity_is_binary_and_matches_sampled_class():
    torch.manual_seed(0)
    logits = torch.tensor([
        [6.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0],
        [0.0, 0.0, 6.0, 0.0],
    ])
    next_class, pred, activity = _step_from_logits(logits, mode="sample", max_trans=3)
    assert torch.equal(pred, next_class.float())
    assert set(activity.cpu().numpy().tolist()).issubset({0.0, 1.0})
    assert torch.equal(activity, (next_class > 0).float())


def test_expected_mode_is_deterministic_and_temperature_supported():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    before = torch.random.get_rng_state()
    a = _step_from_logits(logits, mode="expected", max_trans=3, temperature=1.5)
    after = torch.random.get_rng_state()
    b = _step_from_logits(logits, mode="expected", max_trans=3, temperature=1.5)
    expected_class = torch.round(a[1]).long().clamp(0, 3)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])
    assert torch.equal(a[0], expected_class)
    assert torch.equal(before, after)


def test_conservative_aggregate_calibration_shrinks_and_clips():
    assert conservative_ratio_factor(100.0, 50.0, shrinkage=0.5) == pytest.approx(1.5)
    assert conservative_ratio_factor(100.0, 1.0, max_factor=4.0) == pytest.approx(4.0)
    cal = fit_aggregate_calibration({
        "freq_true_total": 100.0,
        "freq_pred_total": 50.0,
        "spend_true_total": 10.0,
        "spend_pred_total": 20.0,
    })
    assert cal.freq_factor == pytest.approx(1.5)
    assert cal.spend_factor == pytest.approx(0.75)


def test_inactive_frequency_zeros_spend_feedback():
    log_spend = torch.tensor([-1.0, 0.5, 2.0])
    next_class = torch.tensor([0, 1, 0])
    gated = _zero_inactive_spend_feedback(log_spend, next_class)
    assert torch.equal(gated, torch.tensor([0.0, 0.5, 0.0]))


def test_final_manifest_rejects_hash_and_runtime_overrides(tmp_path):
    cfg_path = tmp_path / "toy_cdnow.yaml"
    cfg = {
        "dataset": {"name": "cdnow"},
        "model": {"type": "lstm"},
        "training": {"seed": 7, "epochs": 100},
        "inference": {"mode": "sample", "n_scenarios": 30},
        "output": {"run_name": "toy_cdnow_v1"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    manifest = {
        "version": "test-final",
        "methodology": {"seeds": [42, 7, 123], "inference_primary": "sample"},
        "runtime_expectations": {
            "deep_learning": {
                "epochs": 100,
                "inference_mode": "sample",
                "n_scenarios": 30,
            }
        },
        "deep_learning_configs": [str(cfg_path)],
        "deep_learning_config_hashes": {
            str(cfg_path).lstrip("./"): manifest_config_hash(cfg)
        },
        "benchmark_run_names": [],
    }
    manifest_path = tmp_path / "final_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    runtime_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    runtime_cfg["output"]["run_name"] = "toy_cdnow_v1_seed7_sample"
    metrics = attach_manifest_metadata(
        {},
        config_path=cfg_path,
        config=runtime_cfg,
        run_name="toy_cdnow_v1_seed7_sample",
        manifest_path=manifest_path,
    )
    ok, reason = result_matches_manifest("toy_cdnow_v1_seed7_sample", metrics, manifest)
    assert ok, reason

    bad_hash = dict(metrics)
    bad_hash["final_manifest_config_hash"] = "000000000000"
    ok, reason = result_matches_manifest("toy_cdnow_v1_seed7_sample", bad_hash, manifest)
    assert not ok and "hash" in reason

    bad_epoch = dict(metrics)
    bad_epoch["final_manifest_epochs"] = 1
    ok, reason = result_matches_manifest("toy_cdnow_v1_seed7_sample", bad_epoch, manifest)
    assert not ok and "epoch" in reason

    bad_scenarios = dict(metrics)
    bad_scenarios["final_manifest_n_scenarios"] = 1
    ok, reason = result_matches_manifest("toy_cdnow_v1_seed7_sample", bad_scenarios, manifest)
    assert not ok and "scenario" in reason


def test_final_manifest_uses_benchmark_specific_hashes():
    manifest = {
        "version": "test-final",
        "methodology": {"seeds": [42], "inference_primary": "sample"},
        "deep_learning_configs": [],
        "deep_learning_config_hashes": {},
        "benchmark_run_names": ["pareto_nbd_cdnow"],
        "benchmark_config_hashes": {"pareto_nbd_cdnow": "benchmark123"},
    }
    metrics = {
        "final_manifest_version": "test-final",
        "final_manifest_run_name": "pareto_nbd_cdnow",
        "final_manifest_benchmark_name": "pareto_nbd_cdnow",
        "final_manifest_config_hash": "benchmark123",
    }

    ok, reason = result_matches_manifest("pareto_nbd_cdnow", metrics, manifest)
    assert ok, reason

    metrics["final_manifest_config_hash"] = "stale"
    ok, reason = result_matches_manifest("pareto_nbd_cdnow", metrics, manifest)
    assert not ok and "benchmark config hash" in reason


def test_final_manifest_hash_uses_frozen_config_not_runtime_mutations(tmp_path):
    cfg_path = tmp_path / "toy_cdnow.yaml"
    frozen_cfg = {
        "dataset": {"name": "cdnow", "raw_dir": "data/raw"},
        "model": {"type": "lstm", "max_trans": 3},
        "training": {"seed": 42, "epochs": 300},
        "inference": {"mode": "sample", "n_scenarios": 200},
        "output": {"run_name": "toy_cdnow_v2", "results_dir": "results"},
    }
    cfg_path.write_text(yaml.safe_dump(frozen_cfg))
    cfg_norm = str(cfg_path).lstrip("./")
    manifest = {
        "version": "test-final",
        "methodology": {"seeds": [42], "inference_primary": "sample"},
        "runtime_expectations": {
            "deep_learning": {
                "epochs": 300,
                "inference_mode": "sample",
                "n_scenarios": 200,
            }
        },
        "deep_learning_configs": [str(cfg_path)],
        "deep_learning_config_hashes": {cfg_norm: manifest_config_hash(frozen_cfg)},
        "benchmark_run_names": [],
    }
    manifest_path = tmp_path / "final_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    runtime_cfg = yaml.safe_load(yaml.safe_dump(frozen_cfg))
    runtime_cfg["dataset"]["raw_dir"] = "/kaggle/working/input/cdnow-dataset"
    runtime_cfg["model"]["max_trans"] = 7
    runtime_cfg["output"]["run_name"] = "toy_cdnow_v2_seed42_sample"
    runtime_cfg["output"]["results_dir"] = "/kaggle/working/results"
    runtime_cfg["calibration"] = {"freq_factor": 0.9, "spend_factor": 1.1}

    metrics = attach_manifest_metadata(
        {},
        config_path=cfg_path,
        config=runtime_cfg,
        run_name="toy_cdnow_v2_seed42_sample",
        manifest_path=manifest_path,
        manifest_config=frozen_cfg,
    )

    assert metrics["final_manifest_config_hash"] == manifest_config_hash(frozen_cfg)
    assert metrics["final_manifest_actual_config_hash"] != metrics["final_manifest_config_hash"]
    ok, reason = result_matches_manifest("toy_cdnow_v2_seed42_sample", metrics, manifest)
    assert ok, reason


def test_final_manifest_supports_per_config_runtime_overrides(tmp_path):
    cfg_path = tmp_path / "extension3_lstm_full_dunnhumby.yaml"
    cfg = {
        "dataset": {"name": "dunnhumby"},
        "model": {"type": "lstm"},
        "training": {"seed": 42, "epochs": 100},
        "inference": {"mode": "sample", "n_scenarios": 30},
        "output": {"run_name": "extension3_lstm_full_dunnhumby_final"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg))
    cfg_norm = str(cfg_path).lstrip("./")
    manifest = {
        "version": "test-final",
        "methodology": {"seeds": [42], "inference_primary": "sample"},
        "runtime_expectations": {
            "deep_learning": {
                "inference_mode": "sample",
                "default": {"epochs": 300, "n_scenarios": 200},
                "by_config": {cfg_norm: {"epochs": 100, "n_scenarios": 30}},
            }
        },
        "deep_learning_configs": [str(cfg_path)],
        "deep_learning_config_hashes": {cfg_norm: manifest_config_hash(cfg)},
        "benchmark_run_names": [],
    }

    runtime_cfg = yaml.safe_load(yaml.safe_dump(cfg))
    runtime_cfg["output"]["run_name"] = "extension3_lstm_full_dunnhumby_final_seed42_sample"
    metrics = attach_manifest_metadata(
        {},
        config_path=cfg_path,
        config=runtime_cfg,
        run_name="extension3_lstm_full_dunnhumby_final_seed42_sample",
        manifest_path=tmp_path / "missing_manifest.yaml",
    )
    metrics.update({
        "final_manifest_version": "test-final",
        "final_manifest_run_name": "extension3_lstm_full_dunnhumby_final_seed42_sample",
        "final_manifest_config": cfg_norm,
        "final_manifest_config_hash": manifest_config_hash(runtime_cfg),
        "final_manifest_seed": 42,
        "final_manifest_epochs": 100,
        "final_manifest_inference_mode": "sample",
        "final_manifest_n_scenarios": 30,
    })

    ok, reason = result_matches_manifest(
        "extension3_lstm_full_dunnhumby_final_seed42_sample", metrics, manifest
    )
    assert ok, reason

    bad = dict(metrics)
    bad["final_manifest_epochs"] = 300
    ok, reason = result_matches_manifest(
        "extension3_lstm_full_dunnhumby_final_seed42_sample", bad, manifest
    )
    assert not ok and "epoch" in reason


def test_latex_export_uses_current_log_raw_and_clv_metric_names(tmp_path):
    df = pd.DataFrame({
        "model": ["lstm_joint"],
        "dataset": ["cdnow"],
        "freq_rmse": [1.0],
        "freq_mape": [2.0],
        "bias_pct": [3.0],
        "spend_mae_log": [0.5],
        "spend_mae_raw": [12.0],
        "spend_r2_log": [0.25],
        "clv_mae": [14.0],
        "clv_spearman": [0.7],
        "clv_decile_lift": [1.2],
    })
    out_path = tmp_path / "comparison.tex"
    export_latex_table(df, out_path=str(out_path))
    text = out_path.read_text()
    assert "Spend MAE (log)" in text
    assert "Spend MAE (\\$)" in text
    assert "Spend $R^2$ (log)" in text
    assert "0.500" in text


def test_trainer_restores_best_validation_weights_without_early_stop():
    class OneParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([0.0]))

    class ScriptedTrainer(Trainer):
        def __init__(self, *args, val_losses, **kwargs):
            super().__init__(*args, **kwargs)
            self.val_losses = list(val_losses)
            self.epoch_idx = 0

        def train_epoch(self, dataloader):
            with torch.no_grad():
                self.model.weight.fill_(float(self.epoch_idx + 1))
            return {"freq_loss": float(self.epoch_idx + 1)}

        def validate(self, dataloader):
            loss = self.val_losses[self.epoch_idx]
            self.epoch_idx += 1
            return {"freq_loss": loss}

    model = OneParamModel()
    trainer = ScriptedTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        joint=False,
        val_losses=[3.0, 1.0, 2.0],
    )
    stopper = EarlyStopping(patience=10)
    trainer.fit(train_loader=[object()], val_loader=[object()], epochs=3, early_stopping=stopper)

    assert model.weight.item() == pytest.approx(2.0)


def test_spend_logvar_max_floors_spend_task_weight():
    """A drifted spend log_var must be clamped so the spend task is not silenced."""
    capped = KendallMultiTaskLoss(n_tasks=2, spend_logvar_max=2.0)
    uncapped = KendallMultiTaskLoss(n_tasks=2)
    with torch.no_grad():
        for m in (capped, uncapped):
            m.log_vars.copy_(torch.tensor([0.0, 9.0]))  # spend log_var drifted high

    losses = [torch.tensor(1.0), torch.tensor(1.0)]
    # Effective spend weight = 0.5 * exp(-clamped_log_var).
    # Uncapped: 0.5*exp(-9) ≈ 6e-5 (silenced). Capped at 2.0: 0.5*exp(-2) ≈ 0.068.
    capped(losses)
    floor = 0.5 * float(torch.exp(-torch.tensor(2.0)))
    assert float(capped.task_weights[1]) == pytest.approx(floor, rel=1e-5)
    assert float(uncapped.task_weights[1]) < 1e-3
    # Frequency task (index 0) is untouched by the spend cap.
    assert float(capped.task_weights[0]) == pytest.approx(0.5, rel=1e-5)
