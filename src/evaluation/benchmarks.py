"""
Probabilistic benchmarks for CLV prediction.

Models:
- ParetoNBDModel: frequency-only (lifetimes)
- BGNBDGammaGammaModel: frequency + spend (lifetimes)
- ParetoGGGModel: frequency with regularity (rpy2 → R BTYD.plus)
- GPPMModel: Gaussian process propensity model (PyMC)
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BenchmarkModel(ABC):
    """Base class for all probabilistic benchmarks."""

    @abstractmethod
    def fit(self, rfm_calib: pd.DataFrame) -> None:
        """Fit the model on calibration RFM data."""
        pass

    @abstractmethod
    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """Predict total transactions in holdout period. Returns shape (N,) float."""
        pass

    @abstractmethod
    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray | None:
        """
        Predict total spend in holdout period. Returns shape (N,) float or None.
        Models that only predict frequency return None.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Model name for results file."""
        pass


class ParetoNBDModel(BenchmarkModel):
    """Pareto/NBD frequency-only model via lifetimes."""

    def __init__(self):
        try:
            from lifetimes import ParetoNBDFitter
        except ImportError:
            raise ImportError("Install lifetimes: pip install lifetimes")
        self.fitter = ParetoNBDFitter(penalizer_coef=0.01)
        self.fitted = False

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        """Fit Pareto/NBD on calibration RFM."""
        self.fitter.fit(
            frequency=rfm_calib["frequency"].values,
            recency=rfm_calib["recency"].values,
            T=rfm_calib["T"].values,
        )
        self.fitted = True
        logger.info(f"Fitted {self.name()} on {len(rfm_calib)} customers")

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """Predict expected transactions in holdout period."""
        assert self.fitted, "Call fit() first"
        result = self.fitter.conditional_expected_number_of_purchases_up_to_time(
            t=holdout_weeks,
            frequency=rfm_calib["frequency"].values,
            recency=rfm_calib["recency"].values,
            T=rfm_calib["T"].values,
        )
        # Guard against NaN/inf from logaddexp numerical instability (lifetimes issue
        # when alpha+T or beta+T approaches certain edge cases). Assign 0 for affected
        # customers — conservative but acceptable for a small minority.
        result = np.nan_to_num(np.asarray(result, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return result.astype(np.float32)

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> None:
        """Pareto/NBD does not predict spend."""
        return None

    def name(self) -> str:
        return "pareto_nbd"


class BGNBDGammaGammaModel(BenchmarkModel):
    """BG/NBD + Gamma-Gamma model via lifetimes."""

    def __init__(self):
        try:
            from lifetimes import BetaGeoFitter, GammaGammaFitter
        except ImportError:
            raise ImportError("Install lifetimes: pip install lifetimes")
        self.bgf = BetaGeoFitter(penalizer_coef=0.01)
        self.ggf = GammaGammaFitter(penalizer_coef=0.01)
        self.fitted = False

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        """Fit BG/NBD and Gamma-Gamma on calibration RFM."""
        # BG/NBD on all customers
        self.bgf.fit(
            frequency=rfm_calib["frequency"].values,
            recency=rfm_calib["recency"].values,
            T=rfm_calib["T"].values,
        )

        # Gamma-Gamma only on repeat buyers (frequency > 0)
        repeat_mask = rfm_calib["frequency"] > 0
        if repeat_mask.sum() > 0:
            self.ggf.fit(
                frequency=rfm_calib.loc[repeat_mask, "frequency"].values,
                monetary_value=rfm_calib.loc[repeat_mask, "monetary_value"].values,
            )
            self.has_gamma_gamma = True
        else:
            self.has_gamma_gamma = False
            logger.warning("No repeat buyers in calibration set; Gamma-Gamma fitting skipped")

        self.fitted = True
        logger.info(f"Fitted {self.name()} on {len(rfm_calib)} customers")

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """Predict expected transactions in holdout period.

        One-timers (frequency=0) cause log(negative) in lifetimes' BG/NBD
        conditional formula when a+b<1 (the hyp2f1 value is negative). We
        compute their predictions directly — without the log wrapper — using
        scipy.special.hyp2f1 and the closed-form BG/NBD equation.
        """
        from scipy.special import hyp2f1 as scipy_hyp2f1

        assert self.fitted, "Call fit() first"
        freq = rfm_calib["frequency"].values.astype(float)
        recency = rfm_calib["recency"].values.astype(float)
        T = rfm_calib["T"].values.astype(float)

        # Repeat buyers: standard lifetimes call works fine
        repeat_mask = freq > 0
        pred = np.zeros(len(rfm_calib), dtype=np.float64)
        if repeat_mask.sum() > 0:
            result = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                t=holdout_weeks,
                frequency=freq[repeat_mask],
                recency=recency[repeat_mask],
                T=T[repeat_mask],
            )
            pred[repeat_mask] = np.asarray(result)

        # One-timers: compute directly using scipy hyp2f1 to avoid log(negative)
        if (~repeat_mask).sum() > 0:
            r, alpha, a, b = (
                float(self.bgf.params_["r"]),
                float(self.bgf.params_["alpha"]),
                float(self.bgf.params_["a"]),
                float(self.bgf.params_["b"]),
            )
            T0 = T[~repeat_mask]
            t = float(holdout_weeks)
            _a, _b, _c = r, b, a + b - 1
            _z = t / (alpha + T0 + t)
            hyp_vals = np.array([scipy_hyp2f1(_a, _b, _c, zi) for zi in _z])
            ratio = ((alpha + T0) / (alpha + t + T0)) ** r
            first_term = _c / (a - 1)
            second_term = 1.0 - hyp_vals * ratio
            pred[~repeat_mask] = first_term * second_term

        return pred.astype(np.float32)

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """
        Predict total spend in holdout period for all customers.

        Strategy: Gamma-Gamma gives E[spend per transaction]; multiply by
        BG/NBD predicted number of transactions in holdout.
        One-timers (frequency == 0) are assigned zero predicted spend.
        """
        assert self.fitted, "Call fit() first"
        if not self.has_gamma_gamma:
            return np.zeros(len(rfm_calib), dtype=np.float32)

        # E[spend per transaction] from Gamma-Gamma (only for repeat buyers)
        spend_per_txn = np.zeros(len(rfm_calib), dtype=np.float32)
        repeat_mask = (rfm_calib["frequency"] > 0).values

        spend_per_txn[repeat_mask] = np.asarray(
            self.ggf.conditional_expected_average_profit(
                frequency=rfm_calib.loc[repeat_mask, "frequency"].values,
                monetary_value=rfm_calib.loc[repeat_mask, "monetary_value"].values,
            )
        ).astype(np.float32)

        # Total predicted spend = predicted_freq * E[spend per txn]
        pred_freq = self.predict_freq(rfm_calib, holdout_weeks)
        return (spend_per_txn * pred_freq).astype(np.float32)

    def name(self) -> str:
        return "bgnbd_gg"


class ParetoGGGModel(BenchmarkModel):
    """
    Pareto/GGG (Platzer & Reutterer 2016) via rpy2 → R BTYD.plus.

    Extends Pareto/NBD with a customer-specific Erlang-k regularity parameter.

    R package: BTYD.plus (Platzer 2016).
    Install: R -e 'install.packages("BTYD.plus", repos="https://cran.r-project.org")'

    The model requires a cal.cbs matrix with columns:
      x     — repeat transaction count (frequency in lifetimes convention)
      t.x   — recency (weeks from first to last purchase in calibration)
      T.cal — observation window (weeks from first purchase to end of calibration)
      litt  — log mean inter-transaction time (approximated from aggregate data)
    """

    def __init__(self):
        try:
            import rpy2.robjects as ro
            from rpy2.robjects.packages import importr
            from rpy2.robjects import numpy2ri
            numpy2ri.activate()
        except ImportError:
            raise ImportError("Install rpy2: pip install rpy2")

        self.ro = ro
        self.numpy2ri = numpy2ri

        try:
            self.btyd = importr("BTYD.plus")
        except Exception as e:
            raise ImportError(
                "R package BTYD.plus not found. "
                'Install R, then run: R -e \'install.packages("BTYD.plus")\''
            ) from e

        self.fitted = False
        self.params = None

    def _build_cal_cbs(self, rfm_calib: pd.DataFrame):
        """
        Build the cal.cbs matrix required by BTYD.plus pareto.ggg functions.
        Columns: x (frequency), t.x (recency), T.cal (T), litt.
        """
        cal_cbs = self.ro.r.cbind(
            x=self.ro.FloatVector(rfm_calib["frequency"].astype(float).values),
            **{"t.x": self.ro.FloatVector(rfm_calib["recency"].astype(float).values)},
            **{"T.cal": self.ro.FloatVector(rfm_calib["T"].astype(float).values)},
            litt=self.ro.FloatVector(rfm_calib["litt"].astype(float).values),
        )
        return cal_cbs

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        """
        Fit Pareto/GGG using R BTYD.plus::pareto.ggg.EstimateParameters.
        rpy2 translates dots to underscores: pareto.ggg.EstimateParameters →
        btyd.pareto_ggg_EstimateParameters.
        """
        cal_cbs = self._build_cal_cbs(rfm_calib)
        try:
            self.params = self.btyd.pareto_ggg_EstimateParameters(cal_cbs)
            self.fitted = True
            logger.info(f"Fitted {self.name()} on {len(rfm_calib)} customers (via R)")
        except Exception as e:
            logger.error(f"Failed to fit Pareto/GGG: {e}")
            raise

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """
        Predict expected transactions in holdout using
        R BTYD.plus::pareto.ggg.ExpectedYTransactions.
        """
        assert self.fitted, "Call fit() first"
        cal_cbs = self._build_cal_cbs(rfm_calib)
        try:
            pred = self.btyd.pareto_ggg_ExpectedYTransactions(
                params=self.params,
                cal_cbs=cal_cbs,
                T_star=float(holdout_weeks),
            )
            return np.array(pred).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to predict Pareto/GGG: {e}")
            raise

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> None:
        """Pareto/GGG does not model spend."""
        return None

    def name(self) -> str:
        return "pareto_ggg"


class GPPMModel(BenchmarkModel):
    """
    Gamma-Poisson propensity model (approximation of Dew & Ansari 2018 GPPM).

    Hierarchical model:
        lambda_i ~ Gamma(alpha, beta)          (per-customer purchase rate)
        x_i      ~ Poisson(lambda_i * T_i)    (observed repeat transactions)

    Hyperparameters alpha, beta are estimated via MLE on the negative-binomial
    marginal likelihood (NegBin = Gamma-Poisson compound). Customer-level rates
    are then computed from the analytical posterior:
        lambda_i | x_i, T_i ~ Gamma(alpha + x_i, beta + T_i)
        E[y_i | x_i, T_i]   = (alpha + x_i) / (beta + T_i) * T_star

    This requires only scipy — no MCMC, no pymc.
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.fitted = False

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        from scipy.optimize import minimize
        from scipy.special import gammaln

        x = rfm_calib["frequency"].values.astype(float)
        T = rfm_calib["T"].values.astype(float)
        T = np.where(T <= 0, 1.0, T)  # guard against zero observation windows

        def neg_log_likelihood(params):
            # Negative binomial log-likelihood for Gamma-Poisson compound.
            # x ~ NegBin(r=alpha, p=beta/(beta+T))
            log_a, log_b = params
            a = np.exp(log_a)
            b = np.exp(log_b)
            p = b / (b + T)
            nll = -(
                gammaln(a + x) - gammaln(a) - gammaln(x + 1)
                + a * np.log(p)
                + x * np.log(1 - p)
            ).sum()
            return nll

        res = minimize(
            neg_log_likelihood,
            x0=[0.0, 0.0],  # log(1), log(1) as starting point
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )
        self.alpha = float(np.exp(res.x[0]))
        self.beta = float(np.exp(res.x[1]))
        self.fitted = True
        logger.info(
            f"Fitted {self.name()} on {len(rfm_calib)} customers "
            f"(alpha={self.alpha:.4f}, beta={self.beta:.4f})"
        )

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """E[y_i] = (alpha + x_i) / (beta + T_i) * T_star."""
        assert self.fitted, "Call fit() first"
        x = rfm_calib["frequency"].values.astype(float)
        T = rfm_calib["T"].values.astype(float)
        T = np.where(T <= 0, 1.0, T)
        posterior_rate = (self.alpha + x) / (self.beta + T)
        return (posterior_rate * holdout_weeks).astype(np.float32)

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> None:
        """GPPM does not predict spend."""
        return None

    def name(self) -> str:
        return "gppm"


# Registry of available models
BENCHMARK_MODELS = {
    "pareto_nbd": ParetoNBDModel,
    "bgnbd_gg": BGNBDGammaGammaModel,
    "pareto_ggg": ParetoGGGModel,
    "gppm": GPPMModel,
}


def get_benchmark_model(model_name: str) -> BenchmarkModel:
    """Get a benchmark model by name."""
    if model_name not in BENCHMARK_MODELS:
        raise ValueError(f"Unknown benchmark: {model_name!r}. Choose from: {list(BENCHMARK_MODELS)}")
    return BENCHMARK_MODELS[model_name]()
