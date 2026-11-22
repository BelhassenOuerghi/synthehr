"""
genpoisson.py
-------------
Generalized Poisson (GP) distribution of Consul & Jain (1973), as used by
Lee et al. (2021), "Forecasting COVID-19 counts at a single hospital:
A Hierarchical Bayesian approach" (arXiv:2104.09327) and the upstream
implementation tufts-ml/single-hospital-count-forecasting (src/GenPoisson.py,
a PyMC3 custom Discrete). Re-implemented here in pure numpy/scipy because
PyMC3 is unavailable in this environment.

PMF:  P(y | theta, lam) = theta * (theta + lam*y)^(y-1) * exp(-(theta+lam*y)) / y!
      for y = 0, 1, 2, ...   with theta > 0 and -1 <= lam < 1.

Moments (untruncated): E[y] = theta / (1-lam),  Var[y] = theta / (1-lam)^3,
so the dispersion index Var/Mean = (1-lam)^{-2}:
  lam > 0  -> over-dispersion,
  lam = 0  -> exactly the standard Poisson(theta),
  lam < 0  -> under-dispersion.

For lam < 0 the pmf is truncated at m = floor(-theta/lam) and no longer sums
exactly to one. The upstream code silently ignores the (tiny) truncation mass
(GenPoisson.py returns logp = 0 beyond the support); here we do slightly
better and renormalize the truncated pmf numerically, which keeps all
log-likelihood comparisons proper.
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize

_LAM_MAX = 0.999  # keep lam strictly inside (-1, 1) for numerical safety
_YMAX_HARD_CAP = 100_000  # absolute pmf-grid ceiling, OOM backstop (daily hospital
                          # counts are O(10); a grid this large already has ~0 mass)


def _support_max(theta, lam, tail_sd=15.0, pad=30):
    """Largest support value we need to enumerate for a pmf table."""
    theta = float(np.max(theta))
    mean = theta / (1.0 - lam)
    sd = np.sqrt(theta / (1.0 - lam) ** 3)
    tail = int(np.ceil(mean + tail_sd * sd)) + pad
    if lam < 0:
        # True support ends at floor(-theta/lam); but for |lam| small that point
        # is astronomically far in the tail where the pmf mass is ~0. Enumerating
        # it would build a giant grid and dominate runtime, so cap at the tail
        # bound (the omitted mass is negligible and renormalization is ~0).
        tail = min(int(np.floor(-theta / lam)), tail)
    # Absolute backstop: a pathological parameter draw can send theta huge
    # (mu saturates at exp(20)), which would allocate a multi-GB grid and OOM.
    return max(1, min(tail, _YMAX_HARD_CAP))


def gp_logpmf_table(theta, lam, ymax=None):
    """Normalized log-pmf over y = 0..ymax for scalar theta, lam."""
    if ymax is None:
        ymax = _support_max(theta, lam)
    y = np.arange(ymax + 1)
    arg = theta + lam * y
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(theta) + (y - 1) * np.log(np.maximum(arg, 1e-300)) - arg - gammaln(y + 1)
    lp = np.where(arg > 0, lp, -np.inf)   # truncation for lam < 0
    lp = lp - logsumexp(lp)               # renormalize (no-op up to fp error for lam >= 0)
    return lp


def _gp_lognorm_vec(theta, lam):
    """
    log normalization constant log(sum_y unnorm-pmf) for each theta, vectorized.
    Only needed for lam < 0 (truncated support); for lam >= 0 it is ~0.
    theta : 1-D array. Returns array of same shape.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    # truncation point m_i = floor(-theta_i/lam); global grid covers the largest,
    # but capped at the tail bound so tiny |lam| cannot explode the grid.
    ymax = _support_max(theta.max(), lam)
    y = np.arange(ymax + 1)[None, :]           # 1 x (ymax+1)
    th = theta[:, None]                        # n x 1
    arg = th + lam * y
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(th) + (y - 1) * np.log(np.maximum(arg, 1e-300)) - arg - gammaln(y + 1)
    lp = np.where(arg > 0, lp, -np.inf)
    return logsumexp(lp, axis=1)               # n


def gp_logpmf(y, theta, lam):
    """
    Elementwise normalized log-pmf. `y` and `theta` broadcast; `lam` scalar.
    For lam >= 0 the series sums to one analytically; for lam < 0 the pmf is
    truncated, so a per-theta normalization constant is subtracted (computed
    in one vectorized pass over the broadcast thetas).
    """
    y = np.asarray(y, dtype=float)
    theta = np.asarray(theta, dtype=float)
    arg = theta + lam * y
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(theta) + (y - 1.0) * np.log(np.maximum(arg, 1e-300)) - arg - gammaln(y + 1.0)
    lp = np.where(arg > 0, lp, -np.inf)
    if lam < 0:
        th_b = np.broadcast_to(theta, lp.shape).ravel()
        logZ = _gp_lognorm_vec(th_b, lam).reshape(lp.shape)
        lp = lp - logZ
    return lp


def gp_logpmf_batch(y, theta, lam):
    """
    Normalized log-pmf with ELEMENTWISE lam (unlike gp_logpmf's scalar lam).
    y, theta, lam are arrays broadcast to a common shape; returns that shape.
    Normalization for the lam < 0 entries is done in one vectorized grid pass.
    """
    y, theta, lam = np.broadcast_arrays(np.asarray(y, float),
                                        np.asarray(theta, float),
                                        np.asarray(lam, float))
    shape = y.shape
    yf, thf, lf = y.ravel(), theta.ravel(), lam.ravel()
    arg = thf + lf * yf
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(thf) + (yf - 1.0) * np.log(np.maximum(arg, 1e-300)) - arg - gammaln(yf + 1.0)
    lp = np.where(arg > 0, lp, -np.inf)
    neg = lf < 0
    if np.any(neg):
        th_n, l_n = thf[neg], lf[neg]
        # cap the grid at the tail bound (see _support_max): the true truncation
        # point can be astronomically far for tiny |lam| and carries ~0 mass.
        m = min(int(np.floor((-th_n / l_n).max())) + 1,
                _support_max(float(th_n.max()), float(l_n.min())))
        yg = np.arange(m + 1)[None, :]
        a = th_n[:, None] + l_n[:, None] * yg
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(th_n)[:, None] + (yg - 1) * np.log(np.maximum(a, 1e-300)) - a - gammaln(yg + 1)
        lg = np.where(a > 0, lg, -np.inf)
        lp[neg] = lp[neg] - logsumexp(lg, axis=1)
    return lp.reshape(shape)


def gp_rvs_vec(theta, lam, rng):
    """
    Draw ONE Generalized Poisson sample for each (theta_i, lam_i) pair.
    theta, lam : 1-D arrays of equal length. Returns int array of that length.
    Fully vectorized: builds a shared support grid, tabulates the (renormalized)
    per-row pmf, and inverts each row's CDF with a single uniform draw.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    lam = np.asarray(lam, dtype=float).ravel()
    n = theta.size
    mean = theta / (1.0 - lam)
    sd = np.sqrt(np.maximum(theta, 1e-9) / (1.0 - lam) ** 3)
    ymax = int(np.ceil((mean + 15.0 * sd).max())) + 20
    # Hard cap: never let a pathological (theta, lam) build a giant grid that
    # would exhaust memory. Callers bound theta to the data scale, so this cap
    # is only a last-resort guard and is never reached in normal operation.
    ymax = min(ymax, 200000)
    y = np.arange(ymax + 1)[None, :]                 # 1 x (ymax+1)
    arg = theta[:, None] + lam[:, None] * y          # n x (ymax+1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = (np.log(theta)[:, None] + (y - 1) * np.log(np.maximum(arg, 1e-300))
              - arg - gammaln(y + 1))
    lp = np.where(arg > 0, lp, -np.inf)
    lp = lp - logsumexp(lp, axis=1, keepdims=True)   # renormalize each row
    cdf = np.cumsum(np.exp(lp), axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(n)
    return (cdf < u[:, None]).sum(axis=1).astype(int)


def gp_logpmf_table_unnorm(theta, lam):
    ymax = _support_max(theta, lam)
    y = np.arange(ymax + 1)
    arg = theta + lam * y
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(theta) + (y - 1) * np.log(np.maximum(arg, 1e-300)) - arg - gammaln(y + 1)
    return np.where(arg > 0, lp, -np.inf)


def gp_mean(theta, lam):
    return theta / (1.0 - lam)


def gp_var(theta, lam):
    return theta / (1.0 - lam) ** 3


def gp_rvs(theta, lam, size, rng):
    """Sample by inversion from the tabulated (renormalized) pmf."""
    lp = gp_logpmf_table(theta, lam)
    cdf = np.cumsum(np.exp(lp))
    cdf[-1] = 1.0
    u = rng.random(size)
    return np.searchsorted(cdf, u)


def gp_cdf_pmf_at(y, theta, lam):
    """Return (P(Y <= y-1), P(Y = y)) for the randomized PIT."""
    lp = gp_logpmf_table(theta, lam, ymax=max(int(y) + 1, _support_max(theta, lam)))
    pmf = np.exp(lp)
    y = int(y)
    below = pmf[:y].sum() if y > 0 else 0.0
    at = pmf[y] if y < len(pmf) else 0.0
    return below, at


# ---------------------------------------------------------------------------
# iid maximum-likelihood fits (used for the marginal dispersion diagnostic)
# ---------------------------------------------------------------------------

def fit_gp_iid(y):
    """MLE of (theta, lam) for an iid GP sample. Returns (theta, lam, loglik)."""
    y = np.asarray(y, dtype=float)
    mean, var = y.mean(), y.var(ddof=1)
    lam0 = np.clip(1.0 - np.sqrt(mean / max(var, 1e-9)), -0.9, 0.9)  # moment start
    th0 = mean * (1.0 - lam0)

    def nll(p):
        theta = np.exp(p[0])
        lam = _LAM_MAX * np.tanh(p[1])
        lp = gp_logpmf(y, theta, lam)
        if not np.all(np.isfinite(lp)):
            return 1e12
        return -lp.sum()

    x0 = np.array([np.log(max(th0, 1e-3)), np.arctanh(np.clip(lam0 / _LAM_MAX, -0.99, 0.99))])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
    theta = float(np.exp(res.x[0]))
    lam = float(_LAM_MAX * np.tanh(res.x[1]))
    return theta, lam, -res.fun


def fit_poisson_iid(y):
    """MLE of Poisson rate; returns (mu, loglik)."""
    y = np.asarray(y, dtype=float)
    mu = y.mean()
    ll = np.sum(y * np.log(mu) - mu - gammaln(y + 1))
    return float(mu), float(ll)
