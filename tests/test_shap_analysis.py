import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.evaluation.shap_analysis import (
    CovariateShapWrapper,
    additivity_diagnostics,
    aggregate_dynamic_signed,
    eligible_customer_ids,
    make_disjoint_sample,
    normalise_shap_values,
    _prepare_wrapper_for_gradients,
    run_output_dir,
)
from src.models import LSTMModel


class _HistoryAwareLSTM(LSTMModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.joint = True
        self.state_feature_dim = 0
        self.lstm = nn.LSTM(1, 1, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(
        self,
        week,
        trans,
        hidden=None,
        spend=None,
        state_features=None,
        static_covariates=None,
        dynamic_covariates=None,
    ):
        score = trans.float().sum(dim=1)
        score = score + static_covariates.sum(dim=1)
        score = score + dynamic_covariates.sum(dim=(1, 2))
        logits = torch.zeros(
            (len(trans), trans.shape[1], 2), dtype=torch.float32
        )
        logits[:, -1, 1] = score
        spend_mu = score[:, None].expand(-1, trans.shape[1])
        spend_log_var = torch.zeros_like(spend_mu)
        return logits, spend_mu, spend_log_var, hidden


def _wrapper(trans_values, head="freq", spend_center=0.0, spend_scale=1.0):
    sequence_length = len(trans_values)
    return CovariateShapWrapper(
        _HistoryAwareLSTM(),
        seed_week=torch.zeros((1, sequence_length), dtype=torch.long),
        seed_trans=torch.tensor([trans_values], dtype=torch.long),
        seed_spend=torch.zeros((1, sequence_length)),
        seed_delta_t=torch.zeros((1, sequence_length)),
        head=head,
        spend_center=spend_center,
        spend_scale=spend_scale,
    )


def test_customer_specific_wrapper_preserves_each_history():
    static = torch.tensor([[0.2, 0.4]])
    dynamic = torch.zeros((1, 3, 2))
    quiet_history = _wrapper([0, 0, 0])
    active_history = _wrapper([1, 2, 1])

    quiet_prediction = quiet_history(static, dynamic).item()
    active_prediction = active_history(static, dynamic).item()

    assert active_prediction > quiet_prediction


def test_hurdle_spend_is_returned_in_unscaled_log1p_units():
    static = torch.tensor([[0.2, 0.4]])
    dynamic = torch.zeros((1, 3, 2))
    wrapper = _wrapper(
        [1, 0, 0],
        head="spend",
        spend_center=3.0,
        spend_scale=1.5,
    )

    # Model-space mu = trans sum + static sum = 1.6.
    assert wrapper(static, dynamic).item() == pytest.approx(1.6 * 1.5 + 3.0)


def test_cuda_gradient_mode_survives_shap_eval_without_enabling_dropout():
    wrapper = _wrapper([1, 0, 0])

    _prepare_wrapper_for_gradients(wrapper, torch.device("cuda"))
    wrapper.eval()  # GradientExplainer does this in its constructor.

    assert wrapper.training is False
    assert wrapper.model.training is False
    assert wrapper.model.dropout.training is False
    assert wrapper.model.lstm.training is True


def test_normalise_shap_values_supports_current_and_legacy_layouts():
    static = np.ones((2, 3, 1))
    dynamic = np.ones((2, 4, 5, 1))

    current_static, current_dynamic = normalise_shap_values([static, dynamic])
    legacy_static, legacy_dynamic = normalise_shap_values(
        [[static, dynamic]]
    )

    assert current_static.shape == (2, 3)
    assert current_dynamic.shape == (2, 4, 5)
    assert np.array_equal(current_static, legacy_static)
    assert np.array_equal(current_dynamic, legacy_dynamic)


def test_dynamic_grouping_sums_signed_values_before_absolute_importance():
    values = np.array(
        [
            [[2.0], [-2.0], [1.0]],
            [[-3.0], [1.0], [1.0]],
        ]
    )
    grouped = aggregate_dynamic_signed(values)

    assert grouped[:, 0].tolist() == pytest.approx([1.0, -1.0])
    assert np.mean(np.abs(grouped[:, 0])) == pytest.approx(1.0)
    assert np.mean(np.abs(values).sum(axis=1)[:, 0]) == pytest.approx(5.0)


def test_additivity_diagnostics_reports_exact_decomposition():
    static = np.array([[0.2, 0.3], [0.1, -0.1]])
    dynamic = np.array(
        [
            [[0.1], [0.4]],
            [[0.2], [0.3]],
        ]
    )
    attributed = static.sum(axis=1) + dynamic.sum(axis=(1, 2))
    baseline = np.array([1.0, 2.0])
    prediction = baseline + attributed

    diagnostics = additivity_diagnostics(
        prediction, baseline, static, dynamic
    )

    assert diagnostics["normalized_error"] == pytest.approx(0.0)
    assert diagnostics["residual"] == pytest.approx(np.zeros(2))


def test_primary_cohort_requires_observed_demographic_row(tmp_path):
    pd.DataFrame({"household_key": [2, 4]}).to_csv(
        tmp_path / "hh_demographic.csv", index=False
    )

    class Dataset:
        customer_ids = torch.tensor([1, 2, 3, 4])

    primary = eligible_customer_ids(
        Dataset(), raw_dir=tmp_path, cohort="observed_demographics"
    )
    sensitivity = eligible_customer_ids(
        Dataset(), raw_dir=tmp_path, cohort="all_households"
    )

    assert primary.tolist() == [2, 4]
    assert sensitivity.tolist() == [1, 2, 3, 4]


def test_shared_sample_is_disjoint_and_reproducible():
    first = make_disjoint_sample(
        range(500),
        n_background=100,
        n_explain=100,
        analysis_seed=42,
        cohort="observed_demographics",
    )
    second = make_disjoint_sample(
        range(500),
        n_background=100,
        n_explain=100,
        analysis_seed=42,
        cohort="observed_demographics",
    )

    assert not set(first.background_ids) & set(first.explain_ids)
    assert np.array_equal(first.background_ids, second.background_ids)
    assert np.array_equal(first.explain_ids, second.explain_ids)


def test_namespaced_output_paths_cannot_collide():
    lstm = run_output_dir(
        "results",
        cohort="observed_demographics",
        architecture="lstm",
        seed=42,
        n_integration_samples=64,
    )
    transformer = run_output_dir(
        "results",
        cohort="observed_demographics",
        architecture="transformer",
        seed=42,
        n_integration_samples=64,
    )
    different_seed = run_output_dir(
        "results",
        cohort="observed_demographics",
        architecture="lstm",
        seed=7,
        n_integration_samples=64,
    )

    assert lstm != transformer
    assert lstm != different_seed
