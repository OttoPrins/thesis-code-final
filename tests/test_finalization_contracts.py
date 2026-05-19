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
from src.evaluation.calibration import conservative_ratio_factor, fit_aggregate_calibration
from src.evaluation.compare import _filter_metric_files, export_latex_table
from src.evaluation.metrics import (
    compute_all_metrics,
    mase,
    mase_scale,
    metrics_arrays_path,
    normalized_gini,
    per_week_aggregate_metrics,
    save_metrics_with_artifacts,
    split_metric_artifacts,
)
from src.models.losses import KendallMultiTaskLoss
from src.models import LSTMModel, TransformerModel
from src.training.callbacks import EarlyStopping
from src.training.inference import _step_from_logits, _zero_inactive_spend_feedback
from src.training.trainer import Trainer
from src.utils.final_manifest import (
    attach_manifest_metadata,
    manifest_config_hash,
    result_matches_manifest,
)
from train import _add_result_validity_checks


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
    assert metrics["freq_weekly_mape"] >= 0
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
    assert weekly["freq_weekly_mape"] == pytest.approx(0.0)
    assert normalized_gini(np.array([0, 1, 3]), np.array([0, 1, 3])) == pytest.approx(1.0)
    assert normalized_gini(np.array([0, 1, 3]), np.array([3, 1, 0])) < 0


def test_split_metric_artifacts_rejects_non_scalar_non_numeric_values():
    with pytest.raises(TypeError):
        split_metric_artifacts({"bad": {"nested": "dict"}})


def test_gamma_poisson_is_explicit_diagnostic_not_gppm():
    model = get_benchmark_model("gamma_poisson")
    assert isinstance(model, GammaPoissonPropensityModel)
    assert model.name() == "gamma_poisson"


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
    assert str(valid_gppm) in kept


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
    a = _step_from_logits(logits, mode="expected", max_trans=3, temperature=1.5)
    b = _step_from_logits(logits, mode="expected", max_trans=3, temperature=1.5)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


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
