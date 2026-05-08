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
)
from src.evaluation.compare import export_latex_table
from src.evaluation.metrics import (
    metrics_arrays_path,
    save_metrics_with_artifacts,
    split_metric_artifacts,
)
from src.models import LSTMModel, TransformerModel
from src.training.callbacks import EarlyStopping
from src.training.trainer import Trainer
from src.utils.final_manifest import (
    attach_manifest_metadata,
    manifest_config_hash,
    result_matches_manifest,
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

    metrics = attach_manifest_metadata(
        {},
        config_path=cfg_path,
        config=cfg,
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
