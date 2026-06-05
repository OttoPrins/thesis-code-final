import json
import numpy as np
import pandas as pd
import pytest
import copy
import yaml
from pathlib import Path

from run_benchmarks import _holdout_weekly_matrices
from src.data.datasets.cdnow import CDNOWPipeline
from src.data.datasets.dunnhumby import DunnhumbyPipeline
from src.data.datasets.uci_retail import UCIRetailPipeline
from src.data.collate import collate_fn
from src.data.dataset import CustomerDataset
from src.data.transforms import SequenceBuilder, SpendScaler
from src.evaluation.calibration import (
    RollingOriginInferenceDataset,
    validate_rolling_origins,
)
from train import (
    _resolve_final_finetune_batch_size,
    _resolve_loader_batch_size,
    _teacher_forced_week_rows,
)


def test_cdnow_loads_canonical_5_column_sample(tmp_path):
    path = tmp_path / "CDNOW_sample.txt"
    path.write_text(
        "00004 0001 19970101 2 29.33\n"
        "00018 0002 19970104 1 14.96\n"
    )

    df = CDNOWPipeline().load_raw(str(tmp_path))
    assert list(df.columns) == [
        "master_customer_id",
        "customer_id",
        "date",
        "num_cds",
        "amount",
    ]

    clean = CDNOWPipeline().clean(df)
    assert clean["customer_id"].tolist() == [1, 2]


def test_cdnow_loads_4_column_master_for_diagnostic_runs(tmp_path):
    path = tmp_path / "CDNOW_master.txt"
    path.write_text(
        "00004 19970101 2 29.33\n"
        "00018 19970104 1 14.96\n"
    )

    df = CDNOWPipeline().load_raw(str(tmp_path))
    assert list(df.columns) == ["customer_id", "date", "num_cds", "amount"]


def test_cdnow_final_config_rejects_master_when_sample_required(tmp_path):
    path = tmp_path / "CDNOW_master.txt"
    path.write_text("00004 19970101 2 29.33\n")

    pipe = CDNOWPipeline()
    pipe._current_dataset_cfg = {"prefer_sample_file": True}
    with pytest.raises(ValueError, match="canonical 2,357-customer sample"):
        pipe.load_raw(str(tmp_path))


def test_cdnow_valendin_master_protocol_stats_match_paper_order():
    raw_path = Path("data/raw/CDNOW_master.txt")
    if not raw_path.exists():
        pytest.skip("CDNOW_master.txt is not available in this checkout")

    df = pd.read_csv(
        raw_path,
        sep=r"\s+",
        header=None,
        names=["customer_id", "date", "num_cds", "amount"],
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    first_purchase = df.groupby("customer_id")["date"].min()
    cohort_ids = first_purchase[first_purchase <= pd.Timestamp("1997-03-31")].index
    assert len(cohort_ids) == 23570

    df = df[df["customer_id"].isin(cohort_ids)].copy()
    df["week"] = ((df["date"] - pd.Timestamp("1997-01-01")).dt.days // 7).astype(int)
    cohort_index = pd.Index(cohort_ids)
    calib_counts = (
        df[df["week"].between(0, 38)]
        .groupby("customer_id")
        .size()
        .reindex(cohort_index, fill_value=0)
    )
    holdout_counts = (
        df[df["week"].between(39, 77)]
        .groupby("customer_id")
        .size()
        .reindex(cohort_index, fill_value=0)
    )

    assert calib_counts.mean() == pytest.approx(2.1, abs=0.05)
    assert (calib_counts == 1).mean() * 100 == pytest.approx(59.0, abs=1.0)
    assert holdout_counts.mean() == pytest.approx(0.9, abs=0.05)
    assert (holdout_counts == 0).mean() * 100 == pytest.approx(70.0, abs=1.0)


def test_cdnow_valendin_master_config_parses_expected_protocol():
    cfg_path = Path("experiments/configs/lstm_base_cdnow_replication.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["dataset"]["protocol_name"] == "valendin_cdnow_master_39x39"
    assert cfg["dataset"]["raw_file"] == "CDNOW_master.txt"
    assert cfg["dataset"]["prefer_sample_file"] is False
    assert cfg["dataset"]["calibration_weeks"] == 39
    assert cfg["dataset"]["holdout_weeks"] == 39
    assert cfg["dataset"]["freq_cap"] == "observed"
    assert cfg["dataset"]["calendar_mode"] == "valendin_year_week"
    assert cfg["dataset"]["loss_mask_mode"] == "all"
    assert cfg["dataset"]["keep_zero_amount_transactions"] is True
    assert cfg["model"]["dense_activation"] == "linear"
    assert cfg["model"]["keras_initialization"] is True
    assert cfg["model"]["neutralize_unseen_week_embeddings"] is False
    assert cfg["training"]["optimizer"] == "adam"
    assert cfg["training"]["eps"] == pytest.approx(1e-7)
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["val_batch_size"] == "full"
    assert cfg["training"]["drop_last_train"] is True
    assert cfg["training"]["amp_enabled"] is False
    assert cfg["training"]["repair_on_failure"] is False
    assert cfg["training"]["max_grad_norm"] == pytest.approx(0.0)
    assert cfg["training"]["scheduler"]["type"] == "none"
    assert cfg["training"]["scheduled_sampling"]["enabled"] is False
    assert cfg["inference"]["mode"] == "sample"
    assert cfg["inference"]["n_scenarios"] >= 30
    assert cfg["inference"]["decode_top_bin"] is False
    assert cfg["calibration"]["temperature_scaling"] is False
    assert cfg["calibration"]["aggregate_scaling"] is False


def test_strict_cdnow_config_matches_public_rfm2lstm_reference_contract():
    cfg = yaml.safe_load(Path("experiments/configs/lstm_base_cdnow_replication.yaml").read_text())
    nb = json.loads(Path("reference/rfm2lstm/banking_transactions_demo.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
    compact = "".join(source.split())

    assert "BATCH_SIZE_TRAIN=32" in compact
    assert "BATCH_SIZE_PRED=no_valid_samples" in compact
    assert "BATCH_SIZE_VALID=no_valid_samples" in compact
    assert "valid_samples,valid_targets=samples[-validation_size:],targets[-validation_size:]" in compact
    assert "patience=5" in compact
    assert "optimizer=Adam()" in compact
    assert "Dense(dense_units,name='dense')" in compact
    assert "Categorical(probs=probs).sample()" in compact

    assert cfg["dataset"]["validation_split_mode"] == "shuffled_tail"
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["val_batch_size"] == "full"
    assert cfg["training"]["inference_batch_size"] == "val"
    assert cfg["training"]["early_stopping_patience"] == 5
    assert cfg["training"]["final_finetune"]["enabled"] is False


def test_cdnow_keras_exact_config_parses_expected_ablation_protocol():
    cfg_path = Path("experiments/configs/lstm_base_cdnow_replication_keras_exact.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["dataset"]["protocol_name"] == "valendin_cdnow_master_39x39"
    assert cfg["dataset"]["validation_split_mode"] == "shuffled_tail"
    assert cfg["dataset"]["calibration_weeks"] == 39
    assert cfg["dataset"]["holdout_weeks"] == 39
    assert cfg["dataset"]["freq_cap"] == "observed"
    assert cfg["model"]["dense_activation"] == "linear"
    assert cfg["model"]["keras_initialization"] is True
    assert cfg["model"]["embedding_size_mode"] == "keras_cardinality"
    assert cfg["model"]["keras_lstm_init_mode"] == "whole_matrix"
    assert cfg["model"]["freeze_lstm_bias_hh"] is True
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["val_batch_size"] == "full"
    assert cfg["training"]["inference_batch_size"] == "val"
    assert cfg["training"]["drop_last_train"] is True
    assert cfg["training"]["amp_enabled"] is False
    assert cfg["training"]["repair_on_failure"] is False
    assert cfg["training"]["max_grad_norm"] == pytest.approx(0.0)
    assert cfg["training"]["scheduler"]["type"] == "none"
    assert cfg["training"]["scheduled_sampling"]["enabled"] is False
    assert cfg["training"]["final_finetune"]["enabled"] is False
    assert cfg["calibration"]["temperature_scaling"] is False
    assert cfg["calibration"]["aggregate_scaling"] is False


def test_cdnow_paper_finetune_config_preserves_strict_base_protocol():
    cfg_path = Path("experiments/configs/lstm_base_cdnow_replication_paper_finetune.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["dataset"]["protocol_name"] == "valendin_cdnow_master_39x39"
    assert cfg["training"]["batch_size"] == 32
    assert cfg["training"]["val_batch_size"] == "full"
    assert cfg["training"]["drop_last_train"] is True
    assert cfg["training"]["amp_enabled"] is False
    assert cfg["training"]["repair_on_failure"] is False
    assert cfg["training"]["max_grad_norm"] == pytest.approx(0.0)
    assert cfg["training"]["final_finetune"]["enabled"] is True
    assert cfg["training"]["final_finetune"]["epochs"] == 5
    assert cfg["training"]["final_finetune"]["lr"] == pytest.approx(1e-4)
    assert cfg["training"]["final_finetune"]["batch_size"] == 1024
    assert cfg["calibration"]["temperature_scaling"] is False
    assert cfg["calibration"]["aggregate_scaling"] is False


def test_cdnow_fullbatch_sensitivity_configs_are_explicit_audit_variants():
    strict = yaml.safe_load(Path("experiments/configs/lstm_base_cdnow_replication.yaml").read_text())
    fullbatch = yaml.safe_load(
        Path("experiments/configs/lstm_base_cdnow_replication_fullbatch.yaml").read_text()
    )
    paper = yaml.safe_load(
        Path("experiments/configs/lstm_base_cdnow_replication_paper_finetune.yaml").read_text()
    )
    paper_full = yaml.safe_load(
        Path("experiments/configs/lstm_base_cdnow_replication_paper_finetune_fullbatch.yaml").read_text()
    )

    assert strict["training"]["batch_size"] == 32
    assert fullbatch["training"]["batch_size"] == "full"
    assert fullbatch["training"]["val_batch_size"] == "full"
    assert fullbatch["training"]["repair_on_failure"] is False
    assert fullbatch["inference"]["mode"] == "sample"
    assert fullbatch["evaluation"]["diagnostic_only"] is True

    strict_cmp = copy.deepcopy(strict)
    full_cmp = copy.deepcopy(fullbatch)
    strict_cmp["training"]["batch_size"] = "full"
    strict_cmp["output"]["run_name"] = full_cmp["output"]["run_name"]
    strict_cmp.setdefault("evaluation", {})["diagnostic_only"] = True
    assert full_cmp == strict_cmp

    assert paper["training"]["batch_size"] == 32
    assert paper["training"]["final_finetune"]["batch_size"] == 1024
    assert paper_full["training"]["batch_size"] == "full"
    assert paper_full["training"]["final_finetune"]["batch_size"] == "full"
    assert paper_full["evaluation"]["diagnostic_only"] is True


def test_symbolic_full_batch_size_resolution():
    refs = {"train": 21, "val": 3, "inference": 24}

    assert _resolve_loader_batch_size(
        "full",
        default=32,
        dataset_len=refs["train"],
        name="training.batch_size",
        reference_sizes=refs,
    ) == 21
    assert _resolve_loader_batch_size(
        "val",
        default=32,
        dataset_len=refs["train"],
        name="training.batch_size",
        reference_sizes=refs,
    ) == 3
    assert _resolve_final_finetune_batch_size(
        {"final_finetune": {"enabled": True, "batch_size": "full"}},
        base_batch_size=32,
        full_dataset_len=23570,
    ) == 23570


def test_cdnow_unseen_week_neutralized_config_is_one_switch_sensitivity():
    strict = yaml.safe_load(Path("experiments/configs/lstm_base_cdnow_replication.yaml").read_text())
    sensitivity = yaml.safe_load(
        Path("experiments/configs/lstm_base_cdnow_replication_unseen_week_neutralized.yaml").read_text()
    )

    assert strict["model"]["neutralize_unseen_week_embeddings"] is False
    assert sensitivity["model"]["neutralize_unseen_week_embeddings"] is True
    assert sensitivity["output"]["run_name"] == "lstm_base_cdnow_replication_unseen_week_neutralized"

    strict_cmp = copy.deepcopy(strict)
    sens_cmp = copy.deepcopy(sensitivity)
    strict_cmp["model"]["neutralize_unseen_week_embeddings"] = True
    strict_cmp["output"]["run_name"] = sens_cmp["output"]["run_name"]
    assert sens_cmp == strict_cmp


def test_unseen_week_neutralization_uses_teacher_forced_input_rows():
    assert _teacher_forced_week_rows(calibration_weeks=39, num_rows=52) == set(range(38))
    assert 38 not in _teacher_forced_week_rows(calibration_weeks=39, num_rows=52)
    assert _teacher_forced_week_rows(calibration_weeks=53, num_rows=52) == set(range(52))


def test_cdnow_replication_pipeline_preserves_valendin_contract():
    raw_path = Path("data/raw/CDNOW_master.txt")
    if not raw_path.exists():
        pytest.skip("CDNOW_master.txt is not available in this checkout")

    cfg = yaml.safe_load(Path("experiments/configs/lstm_base_cdnow_replication.yaml").read_text())
    train_ds, val_ds, inference_ds, holdout_gt, _ = CDNOWPipeline().run(copy.deepcopy(cfg))

    assert len(inference_ds) == 23570
    assert len(train_ds) + len(val_ds) == 23570
    assert inference_ds.week.shape[1] == 38
    assert inference_ds.seed_week.shape[1] == 39
    assert inference_ds.y_freq.shape[1] == 38
    assert holdout_gt["raw_freq"].shape == (23570, 39)
    assert inference_ds.max_trans == int(inference_ds.seed_raw_trans.max().item())
    assert np.allclose(inference_ds.mask.numpy(), 1.0)


def test_cdnow_replication_benchmark_holdout_uses_same_calendar_protocol():
    raw_path = Path("data/raw/CDNOW_master.txt")
    if not raw_path.exists():
        pytest.skip("CDNOW_master.txt is not available in this checkout")

    cfg = yaml.safe_load(Path("experiments/configs/lstm_base_cdnow_replication.yaml").read_text())
    pipe = CDNOWPipeline()
    _, _, inference_ds, holdout_gt, _ = pipe.run(copy.deepcopy(cfg))
    rfm_calib, _ = pipe.get_rfm_summary(copy.deepcopy(cfg))
    true_freq_per_week, _ = _holdout_weekly_matrices(
        pipe,
        copy.deepcopy(cfg),
        rfm_calib["customer_id"].values,
    )

    assert np.array_equal(
        inference_ds.customer_ids.numpy(),
        rfm_calib["customer_id"].values,
    )
    assert true_freq_per_week.shape == holdout_gt["raw_freq"].shape
    assert true_freq_per_week.sum() == pytest.approx(float(holdout_gt["raw_freq"].sum()))
    assert np.allclose(true_freq_per_week, holdout_gt["raw_freq"])


def test_validation_split_mode_shuffled_tail_uses_demo_tail():
    pipe = CDNOWPipeline()
    cfg = {"split_seed": 123, "validation_split_mode": "shuffled_tail"}

    train_idx, val_idx = pipe.train_val_indices(10, 0.3, cfg)
    perm = np.random.RandomState(123).permutation(10)

    assert val_idx.tolist() == perm[-3:].tolist()
    assert train_idx.tolist() == perm[:-3].tolist()


def test_cdnow_clean_retains_zero_amount_transactions_for_replication():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "date": [19970101, 19970102, 19970103, 19970104],
        "num_cds": [1, 2, 0, 1],
        "amount": [0.0, 12.5, 99.0, -2.0],
    })

    default_clean = CDNOWPipeline().clean(df)
    assert default_clean["customer_id"].tolist() == [2]

    pipe = CDNOWPipeline()
    pipe._current_dataset_cfg = {"keep_zero_amount_transactions": True}
    clean = pipe.clean(df)
    assert clean["customer_id"].tolist() == [1, 2]
    assert clean.loc[clean["customer_id"] == 1, "transaction_amount"].item() == 0.0


def test_cdnow_requested_master_missing_error_is_actionable(tmp_path):
    (tmp_path / "CDNOW_sample.txt").write_text("00004 0001 19970101 2 29.33\n")

    pipe = CDNOWPipeline()
    pipe._current_dataset_cfg = {"raw_file": "CDNOW_master.txt"}
    with pytest.raises(FileNotFoundError) as excinfo:
        pipe.load_raw(str(tmp_path))

    message = str(excinfo.value)
    assert "CDNOW_master.txt" in message
    assert "CDNOW_sample.txt" in message
    assert "upload_data_to_kaggle.sh" in message


def test_uci_final_validation_rejects_one_year_online_retail(tmp_path):
    path = tmp_path / "Online Retail.csv"
    path.write_text(
        "InvoiceNo;StockCode;Description;Quantity;InvoiceDate;UnitPrice;CustomerID;Country\n"
        "536365;85123A;WHITE HANGING HEART;6;01/12/2010 08:26;2,55;17850;United Kingdom\n"
        "581587;22631;CIRCUS PARADE LUNCH BOX;12;09/12/2011 12:50;1,95;12680;France\n"
    )

    pipe = UCIRetailPipeline()
    pipe._current_dataset_cfg = {"require_online_retail_ii": True}
    clean = pipe.clean(pipe.load_raw(str(tmp_path)))
    with pytest.raises(ValueError, match="two-year UCI Online Retail II"):
        pipe._validate_online_retail_ii_identity(clean, str(tmp_path))


def test_sequence_builder_uses_calendar_week_and_absolute_position():
    df = pd.DataFrame({
        "customer_id": [1, 1, 1],
        "week": [0, 52, 59],
        "weekly_freq": [1, 2, 1],
        "log_spend": [1.0, 2.0, 1.5],
    })

    data = SequenceBuilder(calibration_weeks=60, min_active_weeks=1).build(df)
    assert data["week_input"][0, 0] == 0
    assert data["week_input"][0, 52] == 0
    assert data["week_input"][0, 58] == 6
    assert data["position_input"][0, 52] == 52
    assert data["seed_position"][0, -1] == 59
    assert data["active_mask"].shape == data["y_freq"].shape
    assert data["state_features"].shape == (1, 59, 1)
    assert data["seed_state_features"].shape == (1, 60, 1)


def test_sequence_builder_all_loss_mask_matches_valendin_demo():
    df = pd.DataFrame({
        "customer_id": [1, 1],
        "week": [2, 4],
        "weekly_freq": [1, 1],
        "log_spend": [1.0, 1.5],
    })

    default_data = SequenceBuilder(
        calibration_weeks=6,
        min_active_weeks=1,
        freq_bins=[0, 1, 2, 3],
    ).build(df)
    demo_data = SequenceBuilder(
        calibration_weeks=6,
        min_active_weeks=1,
        freq_bins=[0, 1, 2, 3],
        loss_mask_mode="all",
    ).build(df)

    assert default_data["mask"].tolist() == [[0.0, 0.0, 1.0, 1.0, 1.0]]
    assert demo_data["mask"].tolist() == [[1.0, 1.0, 1.0, 1.0, 1.0]]


def test_robust_spend_scaler_fits_calibration_only_and_preserves_zero_sentinel():
    scaler = SpendScaler(method="robust")
    train = np.array([0.0, 10.0, 20.0, 1000.0], dtype=np.float32)
    transformed = scaler.fit_transform(train)
    center_before = scaler.center_
    scale_before = scaler.scale_

    holdout = scaler.transform(np.array([0.0, 999999.0], dtype=np.float32))
    assert transformed[0] == pytest.approx(0.0)
    assert holdout[0] == pytest.approx(0.0)
    assert scaler.center_ == pytest.approx(center_before)
    assert scaler.scale_ == pytest.approx(scale_before)
    assert scaler.inverse_transform_spend(transformed[1]) == pytest.approx(10.0, rel=1e-5)


class TinyDunnhumbyPipeline(DunnhumbyPipeline):
    def load_raw(self, raw_dir: str):
        return pd.DataFrame({
            "customer_id": [10, 20, 30],
            "date": pd.to_datetime(["2000-01-01", "2000-01-08", "2000-01-15"]),
            "transaction_amount": [5.0, 6.0, 7.0],
        })

    def clean(self, df):
        return df

    def build_covariates(self, raw_dir, calibration_customer_ids, calibration_weeks, holdout_weeks):
        n = len(calibration_customer_ids)
        static = np.ones((n, 2), dtype=np.float32)
        dynamic = np.ones((n, calibration_weeks + holdout_weeks, 2), dtype=np.float32)
        return static, dynamic


def _tiny_dunnhumby_config(mode: str):
    return {
        "dataset": {
            "name": "dunnhumby",
            "raw_dir": "",
            "calibration_weeks": 4,
            "holdout_weeks": 2,
            "min_active_weeks": 1,
            "val_fraction": 0.0,
            "freq_bins": [0, 1, 2, 3],
            "covariate_mode": mode,
            "include_covariates": mode != "none",
        },
        "model": {},
        "training": {"seed": 42},
    }


def test_dunnhumby_covariate_modes_attach_expected_views():
    static_train, _, _, _, _ = TinyDunnhumbyPipeline().run(_tiny_dunnhumby_config("static"))
    dynamic_train, _, _, _, _ = TinyDunnhumbyPipeline().run(_tiny_dunnhumby_config("dynamic"))
    full_train, _, _, _, _ = TinyDunnhumbyPipeline().run(_tiny_dunnhumby_config("full"))
    none_train, _, _, _, _ = TinyDunnhumbyPipeline().run(_tiny_dunnhumby_config("none"))

    assert "static_covariates" in static_train[0]
    assert "dynamic_covariates" not in static_train[0]
    assert "dynamic_covariates" in dynamic_train[0]
    assert "static_covariates" not in dynamic_train[0]
    assert {"static_covariates", "dynamic_covariates"} <= set(full_train[0])
    assert "static_covariates" not in none_train[0]
    assert "dynamic_covariates" not in none_train[0]


def test_split_seed_makes_validation_ids_invariant_to_training_seed(tmp_path):
    rows = []
    for cid in range(1, 9):
        rows.append(f"{cid:05d} {cid:04d} 19970101 1 {10.0 + cid:.2f}\n")
        rows.append(f"{cid:05d} {cid:04d} 19970108 1 {11.0 + cid:.2f}\n")
    (tmp_path / "CDNOW_sample.txt").write_text("".join(rows))

    base_cfg = {
        "dataset": {
            "name": "cdnow",
            "raw_dir": str(tmp_path),
            "calibration_weeks": 3,
            "holdout_weeks": 1,
            "origin_date": "1997-01-01",
            "min_active_weeks": 1,
            "prefer_sample_file": True,
            "val_fraction": 0.25,
            "split_seed": 999,
            "freq_bins": [0, 1, 2, 3],
        },
        "model": {},
        "training": {"seed": 7},
    }
    cfg_other_seed = copy.deepcopy(base_cfg)
    cfg_other_seed["training"]["seed"] = 123

    _, val_a, _, _, _ = CDNOWPipeline().run(copy.deepcopy(base_cfg))
    _, val_b, _, _, _ = CDNOWPipeline().run(cfg_other_seed)

    assert val_a.customer_ids.tolist() == val_b.customer_ids.tolist()


def test_rolling_origin_validation_rejects_holdout_leakage():
    dataset_cfg = {"calibration_weeks": 10}
    validate_rolling_origins(dataset_cfg, [(6, 4)])
    with pytest.raises(ValueError, match="must not touch the final holdout"):
        validate_rolling_origins(dataset_cfg, [(8, 3)])


def test_rolling_origin_truth_uses_raw_unclipped_counts():
    df = pd.DataFrame({
        "customer_id": [1, 1, 1, 1],
        "week": [0, 1, 2, 3],
        "weekly_freq": [1, 2, 5, 4],
        "log_spend": [1.0, 1.1, 1.2, 1.3],
    })
    data = SequenceBuilder(
        calibration_weeks=4,
        min_active_weeks=1,
        freq_bins=[0, 1, 2, 3],
    ).build(df)
    ds = CustomerDataset(data, include_seed=True)
    ro_ds = RollingOriginInferenceDataset(ds, train_weeks=2, validation_weeks=2)

    assert ro_ds.true_freq().tolist() == [[5.0, 4.0]]
    assert ds.seed_trans[:, 2:].tolist() == [[3, 3]]


def test_raw_frequency_target_survives_sequence_dataset_and_collate():
    df = pd.DataFrame({
        "customer_id": [1, 1, 1, 1],
        "week": [0, 1, 2, 3],
        "weekly_freq": [1, 5, 0, 4],
        "log_spend": [1.0, 1.5, 0.0, 1.2],
    })
    data = SequenceBuilder(
        calibration_weeks=4,
        min_active_weeks=1,
        freq_bins=[0, 1, 2, 3],
    ).build(df)
    assert data["y_freq"].tolist() == [[3, 0, 3]]
    assert data["y_freq_raw"].tolist() == [[5, 0, 4]]

    ds = CustomerDataset(data)
    item = ds[0]
    assert item["y_freq"].tolist() == [3, 0, 3]
    assert item["y_freq_raw"].tolist() == [5.0, 0.0, 4.0]

    batch = collate_fn([item])
    assert batch["y_freq"].tolist() == [[3, 0, 3]]
    assert batch["y_freq_raw"].tolist() == [[5.0, 0.0, 4.0]]
