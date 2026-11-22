"""
TabDDPM (Kotelnikov, Baranchuk, Rubachev & Babenko, ICML 2023), implemented
from scratch, and a differentially private trainer for it.

The model:

* numeric columns are mapped to a standard normal by a quantile transform and
  diffused with Gaussian noise (Ho et al., 2020); the network predicts the
  noise;
* discrete columns are diffused with multinomial diffusion (Hoogeboom et al.,
  2021) towards the uniform distribution; the network predicts the clean
  category and the loss is the KL divergence between true and model
  posteriors;
* one MLP denoiser sees both, with a sinusoidal timestep embedding added to a
  projection of the input; cosine noise schedule (Nichol & Dhariwal, 2021);
  sampling uses an exponential moving average of the weights, with warm-up.

The private trainer (:class:`DPTabDDPM`) is DP-SGD (Abadi et al., 2016) with
true Poisson subsampling of privacy units, per-example gradients
(``torch.func``), noise multiplicity (Dockhorn et al., 2023: each unit's loss
is averaged over several noise draws before clipping, at no privacy cost),
and a privacy unit that is either one record or one **patient** (all of a
patient's records are sampled together and clipped as one contribution).
Noise is calibrated as in DP-CTGAN (sensitivity 2C, replace-one); against
add/remove adjacency, for which the accountant is exact under Poisson
sampling, the same run satisfies the smaller ``epsilon_add_remove_``.

Model sizes. The research found that the best size depends on privacy:
without it, bigger is better; under DP at epsilon = 10 the order reverses.
The defaults follow that evidence: ``"large"`` for :class:`TabDDPM`,
``"small"`` for :class:`DPTabDDPM`.

Not covered by the privacy guarantee: the quantile transform and the category
lists are fitted on the training data, and they are stored in saved model
files. Share synthetic data, not model files, unless the data are not
sensitive.

The numerics are identical to the research implementation
(``research/stage4-realism/src/tabddpm.py``); the test suite checks it.
"""
from __future__ import annotations

import copy
import math
import time
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from synthehr.generators._base import (TabularGenerator,
                                       infer_discrete_columns, resolve_device)
from synthehr.privacy.accountant import Run, calibrate_sigma, compose

SIZES = {"small": (128, 128), "medium": (256, 256, 256),
         "large": (512, 512, 512, 512)}


def _hidden(size: Union[str, Sequence[int]]) -> tuple:
    if isinstance(size, str):
        if size not in SIZES:
            raise ValueError(f"size must be one of {list(SIZES)} or a tuple")
        return SIZES[size]
    return tuple(size)


# ---------------------------------------------------------------------------
# Encoding: quantile-normal numerics, one-hot discretes
# ---------------------------------------------------------------------------
class _Encoder:
    def fit(self, df, discrete_columns, seed=0):
        self.columns = list(df.columns)
        self.discrete = [c for c in self.columns if c in discrete_columns]
        self.numeric = [c for c in self.columns if c not in discrete_columns]
        self.qt = None
        if self.numeric:
            self.qt = QuantileTransformer(
                output_distribution="normal",
                n_quantiles=min(1000, len(df)), subsample=10 ** 9,
                random_state=seed).fit(df[self.numeric].to_numpy(float))
        self.categories = [sorted(df[c].astype(str).unique().tolist())
                           for c in self.discrete]
        self.sizes = [len(c) for c in self.categories]
        return self

    def transform(self, df):
        x_num = (self.qt.transform(df[self.numeric].to_numpy(float))
                 .astype("float32") if self.numeric
                 else np.zeros((len(df), 0), "float32"))
        blocks = []
        for c, cats in zip(self.discrete, self.categories):
            idx = {v: i for i, v in enumerate(cats)}
            codes = np.array([idx.get(v, 0) for v in df[c].astype(str)])
            oh = np.zeros((len(df), len(cats)), "float32")
            oh[np.arange(len(df)), codes] = 1.0
            blocks.append(oh)
        x_cat = (np.concatenate(blocks, 1) if blocks
                 else np.zeros((len(df), 0), "float32"))
        return x_num, x_cat

    def inverse(self, x_num, x_cat):
        out = {}
        if self.numeric:
            vals = self.qt.inverse_transform(np.clip(x_num, -5.2, 5.2))
            for j, c in enumerate(self.numeric):
                out[c] = vals[:, j]
        pos = 0
        for c, cats in zip(self.discrete, self.categories):
            k = len(cats)
            out[c] = np.array(cats, dtype=object)[
                x_cat[:, pos:pos + k].argmax(1)]
            pos += k
        return pd.DataFrame(out)[self.columns]


# ---------------------------------------------------------------------------
# Denoiser and schedule
# ---------------------------------------------------------------------------
class Denoiser(nn.Module):
    """MLP over [noisy numerics, noisy one-hots] with a timestep embedding."""

    def __init__(self, d_in: int, d_out: int, hidden=(256, 256, 256),
                 emb: int = 128, dropout: float = 0.0):
        super().__init__()
        self.emb = emb
        self.proj = nn.Linear(d_in, emb)
        self.temb = nn.Sequential(nn.Linear(emb, emb), nn.SiLU(),
                                  nn.Linear(emb, emb))
        layers, d = [], emb
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        half = self.emb // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(
            half, device=x.device, dtype=torch.float32) / half)
        a = t.float().unsqueeze(-1) * freqs
        te = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return self.mlp(self.proj(x) + self.temb(te))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def cosine_betas(T: int, s: float = 0.008) -> np.ndarray:
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return np.clip(1.0 - abar[1:] / abar[:-1], 1e-8, 0.999)


class _Schedule:
    """Per-timestep constants, indexed by t in 0..T (t = 0 is the data)."""

    def __init__(self, T: int, device):
        b = np.concatenate([[0.0], cosine_betas(T)])
        a = 1.0 - b
        abar = np.cumprod(a)
        abar_prev = np.concatenate([[1.0], abar[:-1]])
        post_var = np.zeros_like(b)
        post_var[1:] = b[1:] * (1 - abar_prev[1:]) / (1 - abar[1:])

        def flog(x):
            return np.log(np.maximum(x, 1e-30))

        f = dict(beta=b, alpha=a, abar=abar,
                 sqrt_abar=np.sqrt(abar), sqrt_1m_abar=np.sqrt(1 - abar),
                 post_var=post_var, log_alpha=flog(a), log_1m_alpha=flog(b),
                 log_abar=flog(abar), log_1m_abar=flog(1 - abar))
        self.T = T
        for k, v in f.items():
            setattr(self, k, torch.tensor(v, dtype=torch.float32,
                                          device=device))


# ---------------------------------------------------------------------------
# Multinomial-diffusion helpers (log space, per discrete column)
# ---------------------------------------------------------------------------
def _seg_logsumexp(x, sizes):
    out, pos = [], 0
    for k in sizes:
        s = torch.logsumexp(x[..., pos:pos + k], dim=-1, keepdim=True)
        out.append(s.expand(*x.shape[:-1], k))
        pos += k
    return torch.cat(out, dim=-1)


def _seg_log_softmax(x, sizes):
    return x - _seg_logsumexp(x, sizes)


def _seg_logK(sizes, device):
    return torch.tensor(np.concatenate([[math.log(k)] * k for k in sizes]),
                        dtype=torch.float32, device=device)


def _seg_sample_onehot(logp, sizes, gumbel):
    """One-hot draw per segment from log-probabilities (Gumbel trick)."""
    z = logp + gumbel
    out, pos = [], 0
    for k in sizes:
        seg = z[..., pos:pos + k]
        out.append((seg == seg.max(dim=-1, keepdim=True).values).float())
        pos += k
    return torch.cat(out, dim=-1)


def _gumbel(shape, gen, device):
    u = torch.rand(shape, generator=gen, device=device).clamp(1e-10, 1 - 1e-10)
    return -torch.log(-torch.log(u))


# ---------------------------------------------------------------------------
# TabDDPM
# ---------------------------------------------------------------------------
class TabDDPM(TabularGenerator):
    """Diffusion model for mixed-type tables.

    Parameters
    ----------
    size:
        ``"small"`` (2 x 128), ``"medium"`` (3 x 256), ``"large"`` (4 x 512),
        or a tuple of hidden widths.
    steps:
        Optimiser steps (20,000 was best in the research pilot).
    timesteps:
        Diffusion steps; sampling runs all of them.
    """

    def __init__(self, size: Union[str, Sequence[int]] = "large",
                 emb: int = 128, timesteps: int = 1000, steps: int = 20000,
                 batch_size: int = 4096, lr: float = 1e-3,
                 weight_decay: float = 1e-5, ema_decay: float = 0.999,
                 dropout: float = 0.0, device: str = "auto", seed: int = 0,
                 verbose: bool = False):
        self.hidden, self.emb, self.T = _hidden(size), emb, timesteps
        self.steps, self.batch = steps, batch_size
        self.lr, self.wd, self.ema_decay = lr, weight_decay, ema_decay
        self.dropout = dropout
        self.device = resolve_device(device)
        self.seed, self.verbose = seed, verbose
        self.loss_hist: list[float] = []

    # -- setup ---------------------------------------------------------------
    def _setup(self, df, discrete_columns):
        self._check_input(df)
        if discrete_columns is None:
            discrete_columns = infer_discrete_columns(df)
        self.discrete_columns_ = list(discrete_columns)
        self.columns_ = list(df.columns)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed)
        self.enc = _Encoder().fit(df, discrete_columns, seed=self.seed)
        x_num, x_cat = self.enc.transform(df)
        self.d_num, self.sizes = x_num.shape[1], self.enc.sizes
        self.d_cat = x_cat.shape[1]
        self.x_num = torch.from_numpy(x_num).to(self.device)
        self.x_cat = torch.from_numpy(x_cat).to(self.device)
        self.sch = _Schedule(self.T, self.device)
        self.logK = _seg_logK(self.sizes, self.device) if self.sizes else None
        self.model = Denoiser(self.d_num + self.d_cat, self.d_num + self.d_cat,
                              self.hidden, self.emb, self.dropout
                              ).to(self.device)
        self.ema = copy.deepcopy(self.model).requires_grad_(False)

    def _release_training_data(self):
        """Drop the encoded training rows and the raw (non-averaged) model:
        sampling needs neither, and a saved model must not carry the data."""
        self.x_num = self.x_cat = None
        self.model = None

    # -- forward diffusion and loss targets (parameter-free) -----------------
    def _noised(self, idx, reps: int = 1) -> dict:
        s = self.sch
        idx = torch.as_tensor(idx, device=self.device).repeat_interleave(reps)
        r = len(idx)
        t = torch.randint(1, self.T + 1, (r,), generator=self.gen,
                          device=self.device)
        x0n = self.x_num[idx]
        eps = torch.randn(x0n.shape, generator=self.gen, device=self.device)
        xtn = s.sqrt_abar[t, None] * x0n + s.sqrt_1m_abar[t, None] * eps
        pack = {"t": t, "eps": eps}
        if self.sizes:
            log_x0 = torch.log(self.x_cat[idx].clamp(min=1e-30))
            log_q = torch.logaddexp(log_x0 + s.log_abar[t, None],
                                    s.log_1m_abar[t, None] - self.logK)
            xtc = _seg_sample_onehot(log_q, self.sizes,
                                     _gumbel(log_q.shape, self.gen,
                                             self.device))
            log_xt = torch.log(xtc.clamp(min=1e-30))
            fact1 = torch.logaddexp(log_xt + s.log_alpha[t, None],
                                    s.log_1m_alpha[t, None] - self.logK)
            labp = s.log_abar[t - 1, None]
            l1mabp = s.log_1m_abar[t - 1, None]
            fact2 = torch.logaddexp(log_x0 + labp, l1mabp - self.logK)
            log_true = _seg_log_softmax(fact1 + fact2, self.sizes)
            pack.update(fact1=fact1, labp=labp.expand(r, 1),
                        l1mabp=l1mabp.expand(r, 1), log_true=log_true)
            xin = torch.cat([xtn, xtc], 1)
        else:
            xin = xtn
        pack["xin"] = xin
        return pack

    def _loss(self, out, p):
        """Mean loss over the rows in ``out``; usable inside ``vmap``."""
        loss = 0.0
        if self.d_num:
            loss = ((out[..., :self.d_num] - p["eps"]) ** 2).mean()
        if self.sizes:
            log_x0_hat = _seg_log_softmax(out[..., self.d_num:], self.sizes)
            fact2 = torch.logaddexp(log_x0_hat + p["labp"],
                                    p["l1mabp"] - self.logK)
            log_model = _seg_log_softmax(p["fact1"] + fact2, self.sizes)
            kl = (p["log_true"].exp() * (p["log_true"] - log_model)).sum(-1)
            loss = loss + kl.mean() / len(self.sizes)
        return loss

    def _ema_update(self, step: int):
        # warm-up: without it the average keeps decay**steps of the random
        # initialisation (about 22% after 1,500 steps at 0.999)
        d = min(self.ema_decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)

    def _lr_at(self, step: int, total: int) -> float:
        return self.lr * (1.0 - step / total)

    def fit(self, df: pd.DataFrame,
            discrete_columns: Optional[Sequence[str]] = None,
            groups: Optional[Iterable] = None) -> "TabDDPM":
        self._setup(df, discrete_columns)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        n = len(df)
        t0, run = time.time(), 0.0
        for step in range(self.steps):
            for g in opt.param_groups:
                g["lr"] = self._lr_at(step, self.steps)
            p = self._noised(self.rng.integers(0, n, self.batch))
            loss = self._loss(self.model(p["xin"], p["t"]), p)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            self._ema_update(step)
            run += float(loss.detach())
            if (step + 1) % 500 == 0:
                self.loss_hist.append(run / 500)
                if self.verbose:
                    print(f"  step {step+1}/{self.steps} loss {run/500:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                run = 0.0
        self._release_training_data()
        return self

    # -- reverse process -----------------------------------------------------
    @torch.no_grad()
    def sample(self, n: int, batch: int = 8192, seed_offset: int = 7
               ) -> pd.DataFrame:
        """Draw ``n`` rows. ``seed_offset`` selects an independent stream
        (the same fitted model and offset give the same rows)."""
        if getattr(self, "ema", None) is None:
            raise RuntimeError("call fit() before sample()")
        s, model = self.sch, self.ema.eval()
        gen = torch.Generator(device=self.device).manual_seed(
            self.seed + seed_offset)
        nums, cats, done = [], [], 0
        while done < n:
            b = min(batch, n - done)
            xn = torch.randn((b, self.d_num), generator=gen, device=self.device)
            if self.sizes:
                uni = torch.zeros((b, self.d_cat), device=self.device)
                xc = _seg_sample_onehot(uni, self.sizes,
                                        _gumbel(uni.shape, gen, self.device))
            for t in range(self.T, 0, -1):
                tt = torch.full((b,), t, device=self.device, dtype=torch.long)
                xin = torch.cat([xn, xc], 1) if self.sizes else xn
                out = model(xin, tt)
                if self.d_num:
                    eps_hat = out[:, :self.d_num]
                    mean = (xn - s.beta[t] / s.sqrt_1m_abar[t] * eps_hat) \
                        / torch.sqrt(s.alpha[t])
                    xn = (mean + torch.sqrt(s.post_var[t]) * torch.randn(
                        xn.shape, generator=gen, device=self.device)
                        if t > 1 else mean)
                if self.sizes:
                    log_xt = torch.log(xc.clamp(min=1e-30))
                    log_x0_hat = _seg_log_softmax(out[:, self.d_num:],
                                                  self.sizes)
                    fact1 = torch.logaddexp(log_xt + s.log_alpha[t],
                                            s.log_1m_alpha[t] - self.logK)
                    fact2 = torch.logaddexp(log_x0_hat + s.log_abar[t - 1],
                                            s.log_1m_abar[t - 1] - self.logK)
                    logp = _seg_log_softmax(fact1 + fact2, self.sizes)
                    xc = _seg_sample_onehot(logp, self.sizes,
                                            _gumbel(logp.shape, gen,
                                                    self.device))
            nums.append(xn.cpu().numpy())
            if self.sizes:
                cats.append(xc.cpu().numpy())
            done += b
        x_num = np.concatenate(nums) if self.d_num else np.zeros((n, 0))
        x_cat = np.concatenate(cats) if self.sizes else np.zeros((n, 0))
        return self.enc.inverse(x_num, x_cat)


# ---------------------------------------------------------------------------
# DP-TabDDPM
# ---------------------------------------------------------------------------
class DPTabDDPM(TabDDPM):
    """TabDDPM trained by DP-SGD.

    Parameters
    ----------
    epsilon, delta:
        Target budget; sigma is calibrated to meet it.
    privacy_unit:
        ``"record"`` (one row) or ``"patient"`` (pass ``groups`` to ``fit``:
        all rows of a patient are one unit).
    epochs, expected_batch:
        Passes over the privacy units and the expected units per step
        (Poisson sampling at q = expected_batch / units). 200 epochs at 2,048
        were best in the research pilot.
    noise_multiplicity:
        Noise draws averaged per unit before clipping.
    chunk:
        Rows per per-example-gradient chunk (memory only; padded to a fixed
        size so GPU backends do not recompile every step).
    """

    _KEYS = ("xin", "t", "eps", "fact1", "labp", "l1mabp", "log_true")

    def __init__(self, *, epsilon: float = 10.0, delta: float = 1e-5,
                 clip_norm: float = 1.0, epochs: int = 200,
                 expected_batch: int = 2048, noise_multiplicity: int = 4,
                 privacy_unit: str = "record", chunk: int = 512,
                 noise_multiplier: Optional[float] = None,
                 size: Union[str, Sequence[int]] = "small", lr: float = 2e-3,
                 **kw):
        kw.setdefault("batch_size", expected_batch)
        super().__init__(size=size, lr=lr, **kw)
        if privacy_unit not in ("record", "patient"):
            raise ValueError("privacy_unit must be 'record' or 'patient'")
        self.target_epsilon, self.target_delta = epsilon, delta
        self.clip_norm, self.epochs = clip_norm, epochs
        self.expected_batch = expected_batch
        self.m, self.unit, self.chunk = noise_multiplicity, privacy_unit, chunk
        self.noise_multiplier = noise_multiplier
        self.sigma_: Optional[float] = None
        self.epsilon_: Optional[float] = None
        self.epsilon_add_remove_: Optional[float] = None
        self.privacy_: Optional[Run] = None

    def plan_budget(self, n_units: int, runs: int = 1):
        """(q, steps, sigma) for ``n_units`` privacy units. With ``runs > 1``
        sigma is set so that that many identical runs compose to epsilon
        (for pooled releases)."""
        q = min(self.expected_batch / n_units, 1.0)
        steps = int(math.ceil(self.epochs / q))
        sigma = (float(self.noise_multiplier) if self.noise_multiplier
                 else calibrate_sigma(self.target_epsilon, q, steps,
                                      self.target_delta, runs=runs))
        return q, steps, sigma

    def _per_row_grads(self, rows, p=None):
        from torch.func import functional_call, grad, vmap
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        keys = self._KEYS

        def f(prm, *args):
            pk = dict(zip(keys, args))
            out = functional_call(self.model, prm, (pk["xin"], pk["t"]))
            return self._loss(out, pk)

        if p is None:
            p = self._noised(rows, reps=self.m)
        if self.sizes:
            args = [p[k] for k in keys]
        else:                         # numeric-only tables: inert placeholders
            z = torch.zeros((len(p["t"]), 1), device=self.device)
            args = [p["xin"], p["t"], p["eps"], z, z, z, z]
        args = [a.reshape(len(rows), self.m, *a.shape[1:]) for a in args]
        return vmap(grad(f), in_dims=(None,) + (0,) * len(keys))(params, *args)

    def _clipped_sum(self, rows, unit_of_row):
        """Sum over units of clip(sum of the unit's per-row gradients)."""
        total = {k: torch.zeros_like(v)
                 for k, v in self.model.named_parameters()}
        order = np.argsort(unit_of_row, kind="stable")
        rows, unit_of_row = rows[order], unit_of_row[order]
        bounds = np.flatnonzero(np.diff(unit_of_row)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(rows)]])
        i = 0
        while i < len(starts):        # chunk over UNITS: a patient never splits
            j, size = i, 0
            while j < len(starts) and (size == 0 or size + ends[j] - starts[j]
                                       <= self.chunk):
                size += ends[j] - starts[j]; j += 1
            sel = slice(starts[i], ends[j - 1])
            r = rows[sel]
            _, u_local = np.unique(unit_of_row[sel], return_inverse=True)
            P = self.chunk            # fixed shape; padding unit is dropped
            if len(r) < P:
                pad = P - len(r)
                r = np.concatenate([r, np.repeat(r[:1], pad)])
                u_local = np.concatenate([u_local, np.full(pad, P)])
            slots = max(P, int(u_local.max())) + 1
            u = torch.as_tensor(u_local, device=self.device)
            g = self._per_row_grads(r)
            sq = torch.zeros(slots, device=self.device)
            per_unit = {}
            for k, gk in g.items():
                acc = torch.zeros((slots,) + gk.shape[1:], device=self.device)
                acc.index_add_(0, u, gk)
                per_unit[k] = acc
                sq += acc.reshape(slots, -1).pow(2).sum(1)
            scale = (self.clip_norm / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            if slots > P:
                scale[P] = 0.0
            for k, acc in per_unit.items():
                total[k] += (acc * scale.view(-1, *([1] * (acc.dim() - 1)))
                             ).sum(0)
            i = j
        return total

    def fit(self, df: pd.DataFrame,
            discrete_columns: Optional[Sequence[str]] = None,
            groups: Optional[Iterable] = None) -> "DPTabDDPM":
        self._setup(df, discrete_columns)
        n = len(df)
        if self.unit == "patient":
            if groups is None:
                raise ValueError("privacy_unit='patient' needs groups= "
                                 "(one patient id per row)")
            _, unit_of_row = np.unique(np.asarray(groups), return_inverse=True)
        else:
            unit_of_row = np.arange(n)
        n_units = int(unit_of_row.max()) + 1
        order = np.argsort(unit_of_row, kind="stable")
        unit_rows = np.split(order,
                             np.flatnonzero(np.diff(unit_of_row[order])) + 1)

        q, steps, self.sigma_ = self.plan_budget(n_units)
        self.q_, self.steps_, self.n_units_ = q, steps, n_units
        noise_std = self.sigma_ * 2.0 * self.clip_norm
        if self.verbose:
            print(f"  DP-SGD ({self.unit}-level): units={n_units:,} q={q:.4f} "
                  f"steps={steps} sigma={self.sigma_:.3f}", flush=True)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        t0 = time.time()
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = self._lr_at(step, steps)
            picked = np.flatnonzero(self.rng.random(n_units) < q)
            if len(picked):
                rows = np.concatenate([unit_rows[u] for u in picked])
                total = self._clipped_sum(rows, unit_of_row[rows])
            else:
                total = {k: torch.zeros_like(v)
                         for k, v in self.model.named_parameters()}
            opt.zero_grad(set_to_none=True)
            for k, prm in self.model.named_parameters():
                noise = torch.randn(prm.shape, generator=self.gen,
                                    device=self.device) * noise_std
                prm.grad = (total[k] + noise) / self.expected_batch
            opt.step()
            self._ema_update(step)
            if self.verbose and ((step + 1) % 500 == 0 or step == steps - 1):
                # progress only: a loss on real rows would be a statistic of
                # the private data outside the guarantee
                print(f"  step {step+1}/{steps} ({time.time()-t0:.0f}s)",
                      flush=True)

        self.privacy_ = Run(q, self.sigma_, steps)
        self.epsilon_ = compose([self.privacy_], self.target_delta)
        self.epsilon_add_remove_ = compose(
            [Run(q, 2.0 * self.sigma_, steps)], self.target_delta)
        self._release_training_data()
        return self
