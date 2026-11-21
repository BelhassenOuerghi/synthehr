"""
tabddpm.py
----------
A compact, self-contained implementation of **TabDDPM** (Kotelnikov, Baranchuk,
Rubachev & Babenko, "TabDDPM: Modelling Tabular Data with Diffusion Models",
ICML 2023, arXiv:2209.15421), written from scratch (MIT, no third-party
diffusion code), plus a **differentially private trainer** for it.

The model:
  * numeric columns are mapped to a standard normal by a quantile transform and
    diffused with Gaussian noise (Ho, Jain & Abbeel, 2020), epsilon-prediction;
  * discrete columns are diffused with **multinomial diffusion** (Hoogeboom et
    al., 2021) towards the uniform distribution, with an x0-prediction
    parameterisation and the KL between true and model posteriors as the loss;
  * one MLP denoiser sees both parts, with a sinusoidal timestep embedding
    added to a linear projection of the input, as in the TabDDPM paper;
  * cosine noise schedule (Nichol & Dhariwal, 2021); an exponential moving
    average of the weights is used for sampling.

The private trainer (`DPTabDDPM`) is DP-SGD (Abadi et al., 2016) with:
  * **true Poisson subsampling** of privacy units at rate q, and a fixed
    denominator (the expected batch size), so every step is exactly the sampled
    Gaussian mechanism the accountant analyses (Stages 1 and 2 used fixed-size
    batches accounted as Poisson, a documented approximation);
  * **per-unit clipping**, where the unit is either one ICU stay or one
    **patient** (all of a patient's stays summed before clipping), which turns
    the guarantee into a patient-level one;
  * **noise multiplicity** (Dockhorn, Cao, Vahdat & Kreis, TMLR 2023): each
    unit's loss is averaged over several (timestep, noise) draws before its
    gradient is clipped. This reduces gradient variance at no privacy cost,
    because the averaging happens inside a single unit's contribution;
  * the **same noise convention as Stage 1** (noise std = sigma * 2C, sigma
    calibrated to the target epsilon by the same Renyi accountant), so that
    "epsilon = 10" means the same noise level on every rung of the ladder. Under
    add/remove-one adjacency, for which the sampled Gaussian accountant is
    exact, the same run satisfies the smaller `epsilon_add_remove_`.

Everything the generator does after training (sampling, the EMA, any
re-weighting of its output) is post-processing and costs no budget.

Known non-privatised residue, identical in kind to DP-CTGAN's: the quantile
transform and the category list of each discrete column are fitted on the
real training rows before private training starts.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

from privacy_accountant import calibrate_sigma, compose_epsilon


# ----------------------------------------------------------------------------
# 1. Encoding: quantile-normal numerics, one-hot discretes
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# 2. Denoiser
# ----------------------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self, d_in, d_out, hidden=(256, 256, 256), emb=128,
                 dropout=0.0):
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


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# 3. Schedules
# ----------------------------------------------------------------------------
def cosine_betas(T, s=0.008):
    steps = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return np.clip(1.0 - abar[1:] / abar[:-1], 1e-8, 0.999)


class _Schedule:
    """All per-timestep constants, indexed by t in 0..T (t = 0 is the data)."""

    def __init__(self, T, device):
        b = np.concatenate([[0.0], cosine_betas(T)])            # b[0] unused
        a = 1.0 - b
        abar = np.cumprod(a)                                     # abar[0] = 1
        abar_prev = np.concatenate([[1.0], abar[:-1]])
        post_var = np.zeros_like(b)
        post_var[1:] = b[1:] * (1 - abar_prev[1:]) / (1 - abar[1:])

        def flog(x):
            return np.log(np.maximum(x, 1e-30))

        f = dict(beta=b, alpha=a, abar=abar,
                 sqrt_abar=np.sqrt(abar), sqrt_1m_abar=np.sqrt(1 - abar),
                 post_var=post_var,
                 log_alpha=flog(a), log_1m_alpha=flog(b),
                 log_abar=flog(abar), log_1m_abar=flog(1 - abar))
        self.T = T
        for k, v in f.items():
            setattr(self, k, torch.tensor(v, dtype=torch.float32,
                                          device=device))


# ----------------------------------------------------------------------------
# 4. Multinomial-diffusion helpers (log space, per discrete column segment)
# ----------------------------------------------------------------------------
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
    """One-hot draw per segment from log-probabilities via the Gumbel trick."""
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


# ----------------------------------------------------------------------------
# 5. TabDDPM (non-private)
# ----------------------------------------------------------------------------
class TabDDPM:
    def __init__(self, hidden=(256, 256, 256), emb=128, timesteps=1000,
                 steps=20000, batch_size=4096, lr=1e-3, weight_decay=1e-5,
                 ema_decay=0.999, dropout=0.0, device="cpu", seed=0,
                 verbose=True):
        self.hidden, self.emb, self.T = tuple(hidden), emb, timesteps
        self.steps, self.batch = steps, batch_size
        self.lr, self.wd, self.ema_decay = lr, weight_decay, ema_decay
        self.dropout = dropout
        self.device = torch.device(device)
        self.seed, self.verbose = seed, verbose
        self.loss_hist = []

    # -- setup ---------------------------------------------------------------
    def _setup(self, df, discrete_columns):
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

    # -- forward diffusion + loss targets (independent of the parameters) ----
    def _noised(self, idx, reps=1):
        """For rows `idx` (each repeated `reps` times) draw t and noise, and
        return the network input plus everything the loss needs."""
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
        """Mean loss over the rows in `out`; usable inside torch.func.vmap."""
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

    def _ema_update(self, step):
        # warm-up: without it the average keeps (decay ** steps) of the random
        # initialisation, about 22% after 1,500 steps at decay 0.999, which
        # visibly distorts sampled category shares on short (e.g. DP) runs
        d = min(self.ema_decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for pe, pm in zip(self.ema.parameters(), self.model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)

    def _lr_at(self, step, total):
        return self.lr * (1.0 - step / total)

    def fit(self, df, discrete_columns, groups=None):
        self._setup(df, discrete_columns)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        n = len(df)
        t0 = time.time()
        run = 0.0
        for step in range(self.steps):
            for g in opt.param_groups:
                g["lr"] = self._lr_at(step, self.steps)
            idx = self.rng.integers(0, n, self.batch)
            p = self._noised(idx)
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
        return self

    # -- reverse process -----------------------------------------------------
    @torch.no_grad()
    def sample(self, n, batch=8192, seed_offset=7):
        s, model = self.sch, self.ema.eval()
        gen = torch.Generator(device=self.device).manual_seed(
            self.seed + seed_offset)
        nums, cats = [], []
        done = 0
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
                    if t > 1:
                        xn = mean + torch.sqrt(s.post_var[t]) * torch.randn(
                            xn.shape, generator=gen, device=self.device)
                    else:
                        xn = mean
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


# ----------------------------------------------------------------------------
# 6. DP-TabDDPM
# ----------------------------------------------------------------------------
class DPTabDDPM(TabDDPM):
    """TabDDPM trained by DP-SGD with Poisson-sampled privacy units.

    unit = "stay": each row is a privacy unit (the Stage-1 notion).
    unit = "patient": `groups` gives a patient id per row; all of a patient's
                      rows are sampled together and their gradients summed
                      before clipping, so the guarantee covers a patient.
    """

    _KEYS = ("xin", "t", "eps", "fact1", "labp", "l1mabp", "log_true")

    def __init__(self, *, target_epsilon=10.0, target_delta=1e-5,
                 clip_norm=1.0, epochs=100, expected_batch=1024,
                 noise_multiplicity=4, unit="stay", chunk=256,
                 noise_multiplier=None, **kw):
        kw.setdefault("batch_size", expected_batch)
        super().__init__(**kw)
        self.target_epsilon, self.target_delta = target_epsilon, target_delta
        self.clip_norm, self.epochs = clip_norm, epochs
        self.expected_batch = expected_batch
        self.m, self.unit, self.chunk = noise_multiplicity, unit, chunk
        self.noise_multiplier = noise_multiplier
        self.sigma_ = self.epsilon_spent_ = self.epsilon_add_remove_ = None

    def plan_budget(self, n_units):
        q = min(self.expected_batch / n_units, 1.0)
        steps = int(math.ceil(self.epochs / q))
        sigma = (float(self.noise_multiplier) if self.noise_multiplier
                 else calibrate_sigma(self.target_epsilon, q, steps,
                                      self.target_delta))
        return q, steps, sigma

    def _per_row_grads(self, rows, p=None):
        """Per-row gradients of the multiplicity-averaged loss (torch.func).
        `p` (a `_noised(rows, reps=m)` pack) may be injected for testing."""
        from torch.func import functional_call, grad, vmap
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        keys = self._KEYS

        def f(prm, *args):
            p = dict(zip(keys, args))
            out = functional_call(self.model, prm, (p["xin"], p["t"]))
            return self._loss(out, p)

        if p is None:
            p = self._noised(rows, reps=self.m)
        if self.sizes:
            args = [p[k] for k in keys]
        else:                    # numeric-only tables: inert placeholders
            z = torch.zeros((len(p["t"]), 1), device=self.device)
            args = [p["xin"], p["t"], p["eps"], z, z, z, z]
        args = [a.reshape(len(rows), self.m, *a.shape[1:]) for a in args]
        return vmap(grad(f), in_dims=(None,) + (0,) * len(keys))(params, *args)

    def _clipped_sum(self, rows, unit_of_row):
        """Sum over units of clip(sum of that unit's per-row gradients)."""
        total = {k: torch.zeros_like(v)
                 for k, v in self.model.named_parameters()}
        order = np.argsort(unit_of_row, kind="stable")
        rows, unit_of_row = rows[order], unit_of_row[order]
        bounds = np.flatnonzero(np.diff(unit_of_row)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(rows)]])
        i = 0
        while i < len(starts):      # chunk over UNITS: a patient never splits
            j, size = i, 0
            while j < len(starts) and (size == 0 or
                                       size + ends[j] - starts[j]
                                       <= self.chunk):
                size += ends[j] - starts[j]; j += 1
            sel = slice(starts[i], ends[j - 1])
            r = rows[sel]
            _, u_local = np.unique(unit_of_row[sel], return_inverse=True)
            # Pad every chunk to exactly `chunk` rows so tensor shapes never
            # change between steps (MPS recompiles its graph on every new
            # shape, which Poisson-sized batches would otherwise trigger).
            # Padding rows belong to a dummy unit whose contribution is
            # zeroed below, so the result is unchanged.
            P = self.chunk
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
                scale[P] = 0.0                       # the padding unit
            for k, acc in per_unit.items():
                total[k] += (acc * scale.view(-1, *([1] * (acc.dim() - 1)))
                             ).sum(0)
            i = j
        return total

    def fit(self, df, discrete_columns, groups=None):
        self._setup(df, discrete_columns)
        n = len(df)
        if self.unit == "patient":
            if groups is None:
                raise ValueError("patient-level DP needs a patient id per row")
            _, unit_of_row = np.unique(np.asarray(groups), return_inverse=True)
        elif self.unit == "stay":
            unit_of_row = np.arange(n)
        else:
            raise ValueError(f"unknown privacy unit {self.unit!r}")
        n_units = int(unit_of_row.max()) + 1
        order = np.argsort(unit_of_row, kind="stable")
        cuts = np.flatnonzero(np.diff(unit_of_row[order])) + 1
        unit_rows = np.split(order, cuts)

        q, steps, self.sigma_ = self.plan_budget(n_units)
        self.q_, self.steps_, self.n_units_ = q, steps, n_units
        noise_std = self.sigma_ * 2.0 * self.clip_norm
        if self.verbose:
            print(f"  DP-SGD ({self.unit}-level): units={n_units:,} q={q:.4f} "
                  f"steps={steps} m={self.m} sigma={self.sigma_:.3f} "
                  f"params={n_params(self.model):,}", flush=True)

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
                # progress only: no loss on real rows is computed or logged,
                # because a released training trace would itself be a
                # statistic of the private data outside the guarantee
                print(f"  step {step+1}/{steps} ({time.time()-t0:.0f}s)",
                      flush=True)

        self.epsilon_spent_ = compose_epsilon(q, self.sigma_, steps,
                                              self.target_delta)
        self.epsilon_add_remove_ = compose_epsilon(q, 2.0 * self.sigma_,
                                                   steps, self.target_delta)
        return self
