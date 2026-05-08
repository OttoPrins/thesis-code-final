import numpy as np
import pandas as pd
import pytest

from src.data.datasets.cdnow import CDNOWPipeline
from src.data.datasets.dunnhumby import DunnhumbyPipeline
from src.data.datasets.uci_retail import UCIRetailPipeline
from src.data.transforms import SequenceBuilder


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
