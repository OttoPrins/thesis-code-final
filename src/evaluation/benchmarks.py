"""
Probabilistic benchmarks for CLV prediction.

Models:
- ParetoNBDModel: frequency-only (lifetimes)
- BGNBDGammaGammaModel: frequency + spend (lifetimes)
- ParetoGGGModel: frequency with regularity (rpy2 → R BTYDplus)
- GPPMModel: true Gaussian process propensity model (Stan/cmdstanpy)
- GammaPoissonPropensityModel: non-thesis diagnostic proxy
"""

from abc import ABC, abstractmethod
from pathlib import Path
import glob
import numpy as np
import os
import pandas as pd
import platform
import logging
import re

logger = logging.getLogger(__name__)


def _count_stan_divergences(diagnose_text: str) -> int:
    """Parse CmdStan diagnose text for divergent transition counts."""
    text = diagnose_text or ""
    if re.search(r"\bno divergent transitions\b", text, flags=re.IGNORECASE):
        return 0
    counts = [
        int(match.group(1))
        for match in re.finditer(
            r"(\d+)\s+of\s+\d+\s+(?:\([^)]+\)\s+)?transitions ended with a divergence",
            text,
            flags=re.IGNORECASE,
        )
    ]
    counts.extend(
        int(match.group(1))
        for match in re.finditer(r"(\d+)\s+divergent transitions", text, flags=re.IGNORECASE)
    )
    return int(sum(counts)) if counts else 0


def stan_sampler_diagnostics(fit_result, max_rhat_allowed: float = 1.05) -> dict:
    """
    Extract minimal validity diagnostics from a CmdStan fit.

    The thesis comparison should only include GPPM when Stan reports no
    divergent transitions and R-hat is within a conventional tolerance.
    """
    diagnose_text = ""
    diagnose_available = False
    try:
        diagnose_text = fit_result.diagnose()
        diagnose_available = True
    except Exception as exc:
        logger.warning("CmdStan diagnose() failed: %s", exc)

    divergent_transitions = _count_stan_divergences(diagnose_text)
    max_rhat = float("nan")
    try:
        summary = fit_result.summary()
        if "R_hat" in summary.columns:
            rhats = pd.to_numeric(summary["R_hat"], errors="coerce")
            rhats = rhats.replace([np.inf, -np.inf], np.nan).dropna()
            if not rhats.empty:
                max_rhat = float(rhats.max())
    except Exception as exc:
        logger.warning("CmdStan summary() failed: %s", exc)

    rhat_ok = (not np.isfinite(max_rhat)) or (max_rhat <= max_rhat_allowed)
    diagnostics_ok = bool(diagnose_available and divergent_transitions == 0 and rhat_ok)
    return {
        "benchmark_valid": diagnostics_ok,
        "stan_diagnostics_ok": diagnostics_ok,
        "stan_diagnostics_text_available": diagnose_available,
        "stan_divergent_transitions": divergent_transitions,
        "stan_max_rhat": max_rhat,
        "stan_max_rhat_allowed": float(max_rhat_allowed),
    }


def _ensure_macos_libcxx_headers() -> None:
    """
    Make CmdStan/R source builds find libc++ headers on macOS CLT installs.

    Some Command Line Tools installations expose headers only under versioned
    SDKs, while clang's default `/usr/include/c++/v1` search path is empty. Stan
    compilation then fails on standard headers such as <stdexcept>/<cstddef>.
    """
    if platform.system() != "Darwin":
        return
    patterns = [
        "/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk/usr/include/c++/v1/cstddef",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/"
        "Developer/SDKs/MacOSX*.sdk/usr/include/c++/v1/cstddef",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in glob.glob(pattern))
    if not candidates:
        return

    non_beta = [p for p in candidates if "MacOSX26" not in str(p)]
    chosen = sorted(non_beta or candidates)[-1].parent
    current = os.environ.get("CPLUS_INCLUDE_PATH", "")
    paths = [p for p in current.split(os.pathsep) if p]
    if str(chosen) not in paths:
        os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join([str(chosen), *paths])
        logger.info("Using libc++ header path for native builds: %s", chosen)


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

        # Gamma-Gamma requires frequency > 0 AND monetary_value > 0
        repeat_mask = (rfm_calib["frequency"] > 0) & (rfm_calib["monetary_value"] > 0)
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
    Pareto/GGG (Platzer & Reutterer 2016) via rpy2 → R BTYDplus.

    Extends Pareto/NBD with a customer-specific Erlang-k regularity parameter.

    R package: BTYDplus (Platzer 2016).
    Install: remotes::install_github("mplatzer/BTYDplus", dependencies=TRUE)

    The model requires a cal.cbs data frame with columns:
      x     — repeat transaction count (frequency in lifetimes convention)
      t.x   — recency (weeks from first to last purchase in calibration)
      T.cal — observation window (weeks from first purchase to end of calibration)
      litt  — log mean inter-transaction time (approximated from aggregate data)
    """

    def __init__(
        self,
        mcmc: int = 2500,
        burnin: int = 500,
        thin: int = 50,
        chains: int = 2,
        mc_cores: int = 1,
        trace: int = 100,
    ):
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
            self.btyd = importr("BTYDplus")
        except Exception as e:
            raise ImportError(
                "R package BTYDplus not found. "
                "Install R, then run: R -e "
                '\'remotes::install_github("mplatzer/BTYDplus", dependencies=TRUE)\''
            ) from e

        self.fitted = False
        self.params = None
        self.mcmc = mcmc
        self.burnin = burnin
        self.thin = thin
        self.chains = chains
        self.mc_cores = mc_cores
        self.trace = trace

    def _build_cal_cbs(self, rfm_calib: pd.DataFrame, holdout_weeks: int | None = None):
        """
        Build the cal.cbs data frame required by BTYDplus Pareto/GGG functions.
        Columns: x (frequency), t.x (recency), T.cal (T), litt, optional T.star.
        """
        cols = {
            "x": self.ro.FloatVector(rfm_calib["frequency"].astype(float).values),
            "t.x": self.ro.FloatVector(rfm_calib["recency"].astype(float).values),
            "T.cal": self.ro.FloatVector(rfm_calib["T"].astype(float).values),
            "litt": self.ro.FloatVector(rfm_calib["litt"].astype(float).values),
        }
        if holdout_weeks is not None:
            cols["T.star"] = self.ro.FloatVector(
                np.full(len(rfm_calib), float(holdout_weeks))
            )
        return self.ro.DataFrame(cols)

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        """
        Fit Pareto/GGG using R BTYDplus::pggg.mcmc.DrawParameters.
        rpy2 translates dots to underscores: pggg.mcmc.DrawParameters →
        btyd.pggg_mcmc_DrawParameters.
        """
        cal_cbs = self._build_cal_cbs(rfm_calib)
        try:
            self.params = self.btyd.pggg_mcmc_DrawParameters(
                cal_cbs,
                mcmc=self.mcmc,
                burnin=self.burnin,
                thin=self.thin,
                chains=self.chains,
                mc_cores=self.mc_cores,
                trace=self.trace,
            )
            self.fitted = True
            logger.info(f"Fitted {self.name()} on {len(rfm_calib)} customers (via R)")
        except Exception as e:
            logger.error(f"Failed to fit Pareto/GGG: {e}")
            raise

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        """
        Predict expected transactions in holdout using
        R BTYDplus::mcmc.DrawFutureTransactions and column means.
        """
        assert self.fitted, "Call fit() first"
        cal_cbs = self._build_cal_cbs(rfm_calib, holdout_weeks=holdout_weeks)
        try:
            xstar_draws = self.btyd.mcmc_DrawFutureTransactions(
                cal_cbs,
                self.params,
                self.ro.FloatVector(np.full(len(rfm_calib), float(holdout_weeks))),
            )
            pred = self.ro.r["apply"](xstar_draws, 2, self.ro.r["mean"])
            return np.array(pred).astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to predict Pareto/GGG: {e}")
            raise

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> None:
        """Pareto/GGG does not model spend."""
        return None

    def name(self) -> str:
        return "pareto_ggg"


class GammaPoissonPropensityModel(BenchmarkModel):
    """
    Gamma-Poisson propensity model.

    This is a fast non-thesis diagnostic proxy. It is NOT the Dew & Ansari
    (2018) Gaussian Process Propensity Model and must not be reported as GPPM
    in final thesis comparison tables.

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
        return "gamma_poisson"


class GPPMModel(BenchmarkModel):
    """
    Stan-backed Gaussian Process Propensity Model for CDNOW replication.

    This wrapper intentionally requires customer-week event counts, not only RFM
    aggregates. `run_benchmarks.py` calls `fit_weekly_counts()` for CDNOW. A plain
    `fit(rfm_calib)` call fails clearly because an RFM table has already thrown
    away the temporal event pattern the GPPM is designed to model.
    """

    def __init__(
        self,
        iter_sampling: int = 500,
        iter_warmup: int = 500,
        chains: int = 4,
        seed: int = 42,
        adapt_delta: float = 0.9,
        max_treedepth: int = 12,
    ):
        try:
            from cmdstanpy import CmdStanModel, cmdstan_path
        except ImportError as exc:
            raise ImportError(
                "True Dew/Ansari-style GPPM requires cmdstanpy and CmdStan. "
                "Install with `pip install cmdstanpy` and run "
                "`python -m cmdstanpy.install_cmdstan`. The old Gamma-Poisson "
                "diagnostic is available as `gamma_poisson`, but it is excluded "
                "from thesis GPPM claims."
            ) from exc

        try:
            cmdstan_path()
        except Exception as exc:
            raise ImportError(
                "cmdstanpy is installed, but CmdStan itself is not available. "
                "Run `python -m cmdstanpy.install_cmdstan` before fitting the "
                "true Stan-backed GPPM."
            ) from exc

        _ensure_macos_libcxx_headers()
        self.CmdStanModel = CmdStanModel
        self.iter_sampling = iter_sampling
        self.iter_warmup = iter_warmup
        self.chains = chains
        self.seed = seed
        self.adapt_delta = adapt_delta
        self.max_treedepth = max_treedepth
        self.fitted = False
        self.fit_result = None
        self.prediction_customer_ids: np.ndarray | None = None
        self.pred_freq: np.ndarray | None = None
        self.diagnostics: dict = {
            "benchmark_valid": False,
            "stan_diagnostics_ok": False,
        }

    def fit(self, rfm_calib: pd.DataFrame) -> None:
        raise RuntimeError(
            "True GPPM cannot be fit from RFM aggregates. Use "
            "GPPMModel.fit_weekly_counts(calib_weekly_df, customer_ids, "
            "calibration_weeks, holdout_weeks), which preserves the weekly event log."
        )

    def fit_weekly_counts(
        self,
        calib_weekly_df: pd.DataFrame,
        customer_ids: np.ndarray,
        calibration_weeks: int,
        holdout_weeks: int,
    ) -> None:
        """
        Fit a GP-Poisson propensity model on dense customer-week counts.

        Args:
            calib_weekly_df: aggregated weekly calibration rows with columns
                             customer_id, week, weekly_freq
            customer_ids: ordered customer IDs used for downstream metric alignment
            calibration_weeks: number of observed calibration weeks
            holdout_weeks: forecast horizon
        """
        customer_ids = np.asarray(customer_ids, dtype=np.int64)
        T_calib = int(calibration_weeks)
        T_total = int(calibration_weeks + holdout_weeks)
        x = np.zeros((len(customer_ids), T_calib), dtype=np.int32)
        cid_to_idx = {cid: i for i, cid in enumerate(customer_ids)}

        df = calib_weekly_df.copy()
        df = df[df["customer_id"].isin(cid_to_idx)]
        df = df[(df["week"] >= 0) & (df["week"] < T_calib)]
        if not df.empty:
            row_idx = df["customer_id"].map(cid_to_idx).values.astype(int)
            col_idx = df["week"].values.astype(int)
            x[row_idx, col_idx] = df["weekly_freq"].values.astype(np.int32)

        stan_file = Path(__file__).with_name("stan") / "gppm_cdnow.stan"
        model = self.CmdStanModel(stan_file=str(stan_file))
        data = {
            "N": int(len(customer_ids)),
            "T_calib": T_calib,
            "T_total": T_total,
            "x": x,
            "jitter": 1e-6,
        }
        inits = {
            "mu": -3.0,
            "z_customer": np.zeros(len(customer_ids), dtype=np.float64),
            "sigma_customer": 0.5,
            "eta_time": np.zeros(T_total, dtype=np.float64),
            "sigma_gp": 0.2,
            "rho": 8.0,
        }
        self.fit_result = model.sample(
            data=data,
            iter_sampling=self.iter_sampling,
            iter_warmup=self.iter_warmup,
            chains=self.chains,
            seed=self.seed,
            inits=inits,
            adapt_delta=self.adapt_delta,
            max_treedepth=self.max_treedepth,
            show_progress=True,
        )
        self.diagnostics = stan_sampler_diagnostics(self.fit_result)
        if not self.diagnostics["stan_diagnostics_ok"]:
            logger.warning(
                "GPPM Stan diagnostics failed; metrics will be marked invalid "
                "(divergences=%s, max_rhat=%s).",
                self.diagnostics["stan_divergent_transitions"],
                self.diagnostics["stan_max_rhat"],
            )
        draws = self.fit_result.stan_variable("pred_freq_expected")
        self.pred_freq = draws.mean(axis=0).astype(np.float32)
        self.prediction_customer_ids = customer_ids.copy()
        self.fitted = True
        logger.info(f"Fitted {self.name()} on {len(customer_ids)} customers (Stan)")

    def predict_freq(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> np.ndarray:
        assert self.fitted, "Call fit_weekly_counts() first"
        requested_ids = rfm_calib["customer_id"].values.astype(np.int64)
        if np.array_equal(requested_ids, self.prediction_customer_ids):
            return self.pred_freq.copy()
        pred_lookup = {
            cid: pred for cid, pred in zip(self.prediction_customer_ids, self.pred_freq)
        }
        return np.array([pred_lookup.get(cid, 0.0) for cid in requested_ids], dtype=np.float32)

    def predict_spend(self, rfm_calib: pd.DataFrame, holdout_weeks: int) -> None:
        return None

    def name(self) -> str:
        return "gppm"


# Registry of available models
BENCHMARK_MODELS = {
    "pareto_nbd": ParetoNBDModel,
    "bgnbd_gg": BGNBDGammaGammaModel,
    "pareto_ggg": ParetoGGGModel,
    "gppm": GPPMModel,
    "gamma_poisson": GammaPoissonPropensityModel,
}


def get_benchmark_model(model_name: str) -> BenchmarkModel:
    """Get a benchmark model by name."""
    if model_name not in BENCHMARK_MODELS:
        raise ValueError(f"Unknown benchmark: {model_name!r}. Choose from: {list(BENCHMARK_MODELS)}")
    return BENCHMARK_MODELS[model_name]()
