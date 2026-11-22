"""
models.py
---------
Forecasting models for daily hospital count series.

Main model, GP-AR (Generalized Poisson autoregression), a faithful
observation-driven port of the upstream GAR model
(tufts-ml/single-hospital-count-forecasting, src/GenPoissonAutoregression.py):

  upstream (parameter-driven, PyMC3/NUTS):
      f_t = beta_0 + sum_w beta_w f_{t-w} + eps_t,  eps_t ~ N(0, tau^2)
      y_t ~ GenPoisson(theta = exp(f_t), lam),  lam ~ TruncNormal(0, 0.3)

  this port (observation-driven, MLE via scipy, "route B"):
      f_t = beta_0 + sum_{w=1..W} beta_w * log(1 + y_{t-w})
      mu_t = exp(f_t)                         (conditional mean)
      y_t | past ~ GenPoisson(theta_t = (1-lam) * mu_t,  lam)

Why the change: the upstream latent AR log-rate makes the likelihood an
integral over the latent path, which PyMC3 handles by MCMC. Without PyMC3,
the observation-driven form (a log-linear INGARCH-type recursion, cf. Famoye
1993's restricted GP regression) has a closed-form conditional likelihood
maximizable with scipy.optimize, while keeping the two ingredients the paper
actually tests: (i) the Generalized Poisson likelihood and (ii) an
autoregressive mean. Parameterizing theta_t = (1-lam)*mu_t makes exp(f_t)
the conditional mean exactly, so lam is a pure dispersion parameter and the
GP-vs-Poisson comparison is a like-for-like test with identical mean models.

The standard-Poisson baseline (upstream's poisson_vs_genpoisson.py) is the
same model with lam pinned at 0. Forecast uncertainty comes from Monte Carlo
simulation of future trajectories, optionally with parameters drawn from the
asymptotic normal N(MLE, I^{-1}), a Laplace-style stand-in for the upstream
posterior sampling.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from statsmodels.tools.numdiff import approx_hess1

from genpoisson import (gp_logpmf, gp_rvs, gp_rvs_vec, gp_logpmf_batch,
                        gp_cdf_pmf_at, _LAM_MAX)


class CountARModel:
    """GP-AR when genpoisson=True; standard Poisson-AR when False (lam=0)."""

    def __init__(self, window=1, genpoisson=True):
        self.W = int(window)
        self.genpoisson = bool(genpoisson)
        self.beta = None
        self.lam = 0.0
        self.cov = None      # asymptotic covariance of the raw parameter vector
        self.raw = None      # raw (unconstrained) MLE vector

    # ----- internals -------------------------------------------------------
    def _design(self, y):
        """X_t rows [1, log1p(y_{t-1}), ..., log1p(y_{t-W})] for t = W..T-1."""
        y = np.asarray(y, dtype=float)
        T = len(y)
        X = np.ones((T - self.W, self.W + 1))
        for w in range(1, self.W + 1):
            X[:, w] = np.log1p(y[self.W - w: T - w])
        return X, y[self.W:]

    @staticmethod
    def _unpack(raw, genpoisson):
        if genpoisson:
            return raw[:-1], _LAM_MAX * np.tanh(raw[-1])
        return raw, 0.0

    def _nll(self, raw, X, yt):
        beta, lam = self._unpack(raw, self.genpoisson)
        f = X @ beta
        # Explosion guard: a conditional mean far above the data scale is a bad
        # region the optimizer should leave immediately. Returning a penalty
        # here (instead of evaluating the likelihood) avoids building a giant
        # normalization grid for absurd theta when lam < 0, the source of a
        # ~70s-per-fit slowdown otherwise.
        fmax = np.log(50.0 * (yt.max() + 1.0))
        if np.max(f) > fmax:
            return 1e12 + np.max(f)
        mu = np.exp(np.clip(f, -20, 20))
        theta = (1.0 - lam) * mu
        lp = gp_logpmf(yt, theta, lam)
        if not np.all(np.isfinite(lp)):
            return 1e12
        return -lp.sum()

    # ----- API -------------------------------------------------------------
    def fit(self, y):
        X, yt = self._design(y)
        ybar = max(np.mean(yt), 0.1)
        x0 = np.zeros(self.W + 1 + (1 if self.genpoisson else 0))
        x0[0] = np.log(ybar) * 0.5
        x0[1] = 0.5 * np.log(ybar) / max(np.log1p(ybar), 1e-6)
        res = minimize(self._nll, x0, args=(X, yt), method="Nelder-Mead",
                       options={"maxiter": 20000, "maxfev": 40000,
                                "xatol": 1e-7, "fatol": 1e-9})
        res2 = minimize(self._nll, res.x, args=(X, yt), method="BFGS",
                        options={"maxiter": 500})
        best = res2 if res2.fun <= res.fun else res
        self.raw = best.x
        self.beta, self.lam = self._unpack(best.x, self.genpoisson)
        self.nll_ = best.fun
        try:
            H = approx_hess1(best.x, self._nll, args=(X, yt))
            cov = np.linalg.pinv(H)
            # guard against a non-PSD numerical Hessian
            evals, evecs = np.linalg.eigh((cov + cov.T) / 2)
            self.cov = (evecs * np.maximum(evals, 1e-10)) @ evecs.T
        except Exception:
            self.cov = None
        return self

    def cond_theta(self, y):
        """Conditional (theta_t, lam) along an observed series."""
        X, yt = self._design(y)
        mu = np.exp(np.clip(X @ self.beta, -20, 20))
        return (1.0 - self.lam) * mu, yt

    def loglik_per_obs(self, y):
        """Exact one-step-ahead log predictive for each y_t, t = W..T-1."""
        theta, yt = self.cond_theta(y)
        return gp_logpmf(yt, theta, self.lam)

    def _draw_param_matrix(self, S, rng, param_uncertainty):
        """Return (betas[S, W+1], lams[S]) drawn from N(MLE, cov) or fixed."""
        if param_uncertainty and self.cov is not None:
            raws = rng.multivariate_normal(self.raw, self.cov, size=S)
        else:
            raws = np.tile(self.raw, (S, 1))
        if self.genpoisson:
            betas = raws[:, :-1]
            lams = _LAM_MAX * np.tanh(raws[:, -1])
        else:
            betas = raws
            lams = np.zeros(S)
        return betas, np.clip(lams, -0.95, 0.95)

    def forecast_samples(self, history, F, S, rng, param_uncertainty=True):
        """
        Simulate S trajectories of the next F days given `history`, vectorized
        across trajectories. Returns (samples[S, F], theta_paths[S, F], lams[S]);
        theta paths are kept so heldout predictive log-likelihoods can be
        Rao-Blackwellized.
        """
        history = np.asarray(history, dtype=float)
        betas, lams = self._draw_param_matrix(S, rng, param_uncertainty)
        samples = np.zeros((S, F), dtype=int)
        thetas = np.zeros((S, F))
        # Bound the conditional mean to the data scale. A drawn AR coefficient
        # > 1 makes the log-linear recursion explosive; feeding sampled counts
        # back in can then run theta off to infinity and blow up the sampler's
        # support grid. Capping f keeps every trajectory on a sane count scale.
        fmax = np.log(50.0 * (history.max() + 1.0))
        # buf[:, 0] is the most recent lag, buf[:, W-1] the oldest.
        buf = np.tile(history[-self.W:][::-1], (S, 1))    # S x W
        b0 = betas[:, 0]
        bw = betas[:, 1:]                                  # S x W
        for h in range(F):
            f = np.clip(b0 + np.sum(bw * np.log1p(buf), axis=1), -20, fmax)
            mu = np.exp(f)
            theta = np.maximum((1.0 - lams) * mu, 1e-8)
            thetas[:, h] = theta
            ynew = gp_rvs_vec(theta, lams, rng)
            samples[:, h] = ynew
            buf = np.column_stack([ynew.astype(float), buf[:, :-1]])
        return samples, thetas, lams

    def heldout_loglik(self, thetas, lams, y_obs):
        """
        Upstream-style heldout score: log mean over trajectories of the joint
        F-day likelihood, divided by F  (log-mean-exp over MC paths), plus the
        Rao-Blackwellized per-day version.
        """
        y_obs = np.asarray(y_obs, dtype=float)
        S, F = thetas.shape
        lp = gp_logpmf_batch(np.broadcast_to(y_obs, (S, F)),
                             thetas, lams[:, None])
        joint = (logsumexp(lp.sum(axis=1)) - np.log(S)) / F
        per_day = logsumexp(lp, axis=0) - np.log(S)   # log p(y_d | history)
        return float(joint), per_day

    def pit_one_step(self, y, rng):
        """Randomized PIT values of the one-step predictive on series y."""
        theta, yt = self.cond_theta(y)
        u = np.zeros(len(yt))
        for i, (th, yy) in enumerate(zip(theta, yt)):
            below, at = gp_cdf_pmf_at(int(yy), th, self.lam)
            u[i] = below + rng.random() * at
        return u


# ---------------------------------------------------------------------------
# Baselines (non-likelihood): seasonal naive and moving average, with
# empirical residual-bootstrap predictive samples so that MAE/RMSE/coverage
# are computed identically to the AR models.
# ---------------------------------------------------------------------------

class _EmpiricalBaseline:
    def __init__(self):
        self.resid_by_h = None

    def _point(self, history, F):
        raise NotImplementedError

    def fit(self, y, F):
        """Collect historical h-step-ahead errors of the rule on `y`."""
        y = np.asarray(y, dtype=float)
        self.resid_by_h = {h: [] for h in range(F)}
        start = max(self.min_history, 1)
        for t in range(start, len(y) - F):
            pts = self._point(y[:t], F)
            for h in range(F):
                self.resid_by_h[h].append(y[t + h] - pts[h])
        self.resid_by_h = {h: np.asarray(v) for h, v in self.resid_by_h.items()}
        return self

    def forecast_samples(self, history, F, S, rng, param_uncertainty=None):
        pts = self._point(np.asarray(history, dtype=float), F)
        out = np.zeros((S, F), dtype=int)
        for h in range(F):
            res = self.resid_by_h[h]
            draw = res[rng.integers(0, len(res), size=S)]
            out[:, h] = np.maximum(np.rint(pts[h] + draw), 0).astype(int)
        return out, None, None


class SeasonalNaive(_EmpiricalBaseline):
    """y_hat(t+h) = y(t+h-7k) for the smallest k putting the lag in history."""
    min_history = 7

    def _point(self, history, F):
        pts = np.zeros(F)
        for h in range(F):
            lag = 7 * int(np.ceil((h + 1) / 7.0))
            pts[h] = history[len(history) + h - lag]
        return pts


class MovingAverage(_EmpiricalBaseline):
    """y_hat(t+h) = mean of the last `window` observations (flat)."""
    min_history = 7

    def __init__(self, window=7):
        super().__init__()
        self.window = int(window)
        self.min_history = self.window

    def _point(self, history, F):
        return np.full(F, history[-self.window:].mean())
