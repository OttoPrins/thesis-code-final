"""
Synthetic recovery test for GammaPoissonPropensityModel.

Generates customers from a known hierarchical Gamma-Poisson model, fits the
non-thesis Gamma-Poisson diagnostic model,
and verifies that the recovered per-customer rates Spearman-correlate ≥ 0.80 with
the true latent rates.

Run:
    pytest tests/test_gppm_recovery.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmarks import GammaPoissonPropensityModel


def _simulate_customers(
    n_customers: int = 500,
    alpha_true: float = 1.5,
    beta_true: float = 4.0,
    calib_weeks: int = 39,
    holdout_weeks: int = 39,
    rng: np.random.RandomState = None,
) -> tuple:
    """
    Simulate customers from:
        lambda_i ~ Gamma(alpha_true, beta_true)
        x_i | lambda_i, T_i ~ Poisson(lambda_i * T_i)
        y_i | lambda_i, T*  ~ Poisson(lambda_i * T*)

    T_i is uniform in [calib_weeks/2, calib_weeks] to mimic varying observation windows.

    Returns:
        rfm_calib:    DataFrame with columns [customer_id, frequency, recency, T, monetary_value, litt]
        true_rates:   (N,) float — latent lambda_i per customer
        true_holdout: (N,) float — E[y_i | lambda_i] = lambda_i * holdout_weeks
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Latent rates
    true_rates = rng.gamma(alpha_true, 1.0 / beta_true, size=n_customers)

    # Observation windows (weeks)
    T_i = rng.uniform(calib_weeks / 2, calib_weeks, size=n_customers)

    # Observed counts
    x_i = rng.poisson(true_rates * T_i)

    # Recency: fraction of T_i (simplified — not critical for this model which ignores recency)
    recency = rng.uniform(0, T_i)

    # Monetary value: positively correlated with rate (simplistic proxy)
    monetary_value = np.maximum(rng.normal(true_rates * 20, 5), 0.01)

    # litt: log mean inter-transaction time (approx; this model ignores this field)
    litt = np.log(np.maximum(T_i / np.maximum(x_i, 1), 0.01))

    rfm = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "frequency": x_i.astype(float),
        "recency": recency,
        "T": T_i,
        "monetary_value": monetary_value,
        "litt": litt,
    })

    true_holdout = true_rates * holdout_weeks
    return rfm, true_rates, true_holdout


def test_gamma_poisson_recovers_latent_rates():
    """
    The posterior rate estimates should correlate strongly with the true
    latent Gamma rates used to generate the synthetic data.
    """
    from scipy import stats

    rng = np.random.RandomState(0)
    rfm, true_rates, _ = _simulate_customers(n_customers=500, rng=rng)

    model = GammaPoissonPropensityModel()
    model.fit(rfm)

    # Posterior rate estimate: (alpha + x_i) / (beta + T_i)
    x = rfm["frequency"].values
    T = rfm["T"].values
    posterior_rates = (model.alpha + x) / (model.beta + T)

    rho, pval = stats.spearmanr(true_rates, posterior_rates)
    assert rho >= 0.80, (
        f"GPPM synthetic recovery Spearman = {rho:.3f} < 0.80 threshold. "
        f"Model may not be recovering the latent Gamma-Poisson structure."
    )


def test_gamma_poisson_predict_freq_positive():
    """Predicted frequencies must be non-negative for all customers."""
    rfm, _, true_holdout = _simulate_customers(n_customers=200)
    model = GammaPoissonPropensityModel()
    model.fit(rfm)
    preds = model.predict_freq(rfm, holdout_weeks=39)
    assert (preds >= 0).all(), "GammaPoissonPropensityModel produced negative predictions"


def test_gamma_poisson_predict_total_close_to_truth():
    """
    The total predicted transactions should be within 30% of the true expected total.
    This is a weak test — just checks the model isn't wildly off at the cohort level.
    """
    rng = np.random.RandomState(99)
    rfm, _, true_holdout = _simulate_customers(n_customers=1000, rng=rng)

    model = GammaPoissonPropensityModel()
    model.fit(rfm)
    preds = model.predict_freq(rfm, holdout_weeks=39)

    pred_total = preds.sum()
    true_total = true_holdout.sum()
    ratio = abs(pred_total - true_total) / true_total
    assert ratio < 0.30, (
        f"GPPM cohort-level bias = {ratio:.1%} > 30%. "
        f"pred_total={pred_total:.1f}, true_total={true_total:.1f}"
    )


def test_gamma_poisson_predict_spend_returns_none():
    """Gamma-Poisson diagnostic does not model spend — predict_spend returns None."""
    rfm, _, _ = _simulate_customers(n_customers=100)
    model = GammaPoissonPropensityModel()
    model.fit(rfm)
    assert model.predict_spend(rfm, holdout_weeks=39) is None


def test_gamma_poisson_name():
    assert GammaPoissonPropensityModel().name() == "gamma_poisson"
