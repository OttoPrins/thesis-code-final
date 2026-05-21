import numpy as np
import pandas as pd
import pytest
import copy
import yaml
from pathlib import Path

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
    cfg_path = Path("experiments/configs/lstm_base_cdnow_valendin_master_39x39.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())

    assert cfg["dataset"]["protocol_name"] == "valendin_cdnow_master_39x39"
    assert cfg["dataset"]["raw_file"] == "CDNOW_master.txt"
    assert cfg["dataset"]["prefer_sample_file"] is False
    assert cfg["dataset"]["calibration_weeks"] == 39
    assert cfg["dataset"]["holdout_weeks"] == 39
    assert cfg["dataset"]["freq_cap"] == "observed"
    assert cfg["inference"]["mode"] == "sample"
    assert cfg["inference"]["n_scenarios"] >= 30


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
