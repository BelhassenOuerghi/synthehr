"""
DP-CTGAN: CTGAN trained under (epsilon, delta)-differential privacy, with
optional fair conditional training and sampling.

Only the critic touches real data, so privatising the critic's updates
privatises the generator, and every sample it emits, by post-processing.

Where the privacy budget goes:

1. **DP-SGD on the critic** (Abadi et al., 2016). Real rows enter a critic
   step in PacGAN groups; each group's gradient (Wasserstein loss and gradient
   penalty) is clipped to norm ``clip_norm``, the clipped gradients are summed,
   and Gaussian noise with standard deviation ``sigma * 2 * clip_norm`` is
   added (replace-one adjacency). Fixed-size batches are accounted as Poisson
   sampling at the same rate, the standard approximation.
2. **One noisy release of the category counts.** Training-by-sampling needs
   category frequencies, a statistic of the data. They are released once
   through a Gaussian mechanism and charged to the same accountant. Both the
   conditional sampling weights and the subgroup shares used for fair sampling
   are derived from that single release.
3. **Real rows are drawn uniformly** and conditioned on their own category,
   instead of being matched to a sampled category, which would break the
   uniform-sampling assumption the accountant relies on.

Not covered by the guarantee: the per-column Gaussian mixtures and the
category lists are fitted on the training data before private training.

Fairness. ``fair_train_columns`` trains with uniform (data-independent)
sampling weights for those columns; ``sample_balanced`` releases data whose
subgroup composition is uniform over adequately supported groups. Both cost
no additional epsilon.

Correction relative to the programme's Stage 2 code: that version derived the
subgroup shares from a *second* noisy release of the counts, which the
accountant did not charge (actual epsilon 9.9922 against 9.9902 charged, still
within the budget of 10). Here both come from one release, so the charged
epsilon is the true epsilon. Without fairness options the numerics are
identical to Stage 1.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from synthehr.fairness.resample import fair_target
from synthehr.generators._base import infer_discrete_columns
from synthehr.generators.ctgan import (CTGAN, CondSampler, DataTransformer,
                                       Discriminator, Generator, _activate)
from synthehr.privacy.accountant import Run, calibrate_sigma, compose


class DPCTGAN(CTGAN):
    """CTGAN with a DP-SGD critic and end-to-end (epsilon, delta) accounting.

    Parameters
    ----------
    epsilon, delta:
        Target privacy budget; the noise multiplier is calibrated to meet it.
    clip_norm:
        Per-group gradient clipping bound C.
    noise_multiplier:
        Set to fix sigma instead of calibrating it (``epsilon`` is then only
        reported, not targeted).
    hist_noise_multiplier:
        Noise multiplier of the single category-count release.
    fair_train_columns:
        Discrete columns trained with uniform conditional weights.
    epochs:
        Defaults to 100: every critic step spends budget.
    """

    def __init__(self, *, epsilon: float = 10.0, delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 noise_multiplier: Optional[float] = None,
                 hist_noise_multiplier: float = 32.0,
                 fair_train_columns: Sequence[str] = (), epochs: int = 100,
                 **ctgan_kwargs):
        super().__init__(epochs=epochs, **ctgan_kwargs)
        self.target_epsilon, self.target_delta = epsilon, delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.hist_noise_multiplier = hist_noise_multiplier
        self.fair_train_columns = list(fair_train_columns)
        self.sigma_: Optional[float] = None
        self.epsilon_: Optional[float] = None
        self.privacy_: Optional[Run] = None
        self.noisy_shares_: dict[str, np.ndarray] = {}

    # -- the single noisy release of category counts -------------------------
    def _dp_cond_freqs(self, data, disc_spans, rng):
        names = [c.name for c in self.transformer.columns
                 if c.ctype == "discrete"]
        std = self.hist_noise_multiplier * math.sqrt(max(len(disc_spans), 1))
        logfreq = []
        for name, (st, w) in zip(names, disc_spans):
            counts = data[:, st:st + w].sum(axis=0)
            noisy = np.maximum(counts + rng.normal(0.0, std, size=w), 1.0)
            self.noisy_shares_[name] = noisy / noisy.sum()
            lf = np.log(noisy)
            if name in self.fair_train_columns:
                logfreq.append(np.ones(w) / w)          # data-independent
            else:
                logfreq.append(lf / lf.sum() if lf.sum() > 0
                               else np.ones(w) / w)
        return logfreq

    # -- condition each real row on its own category -------------------------
    @staticmethod
    def _self_cond(rows_encoded, disc_spans, offsets, rng):
        b = rows_encoded.shape[0]
        cond = np.zeros((b, sum(w for _, w in disc_spans)), dtype="float32")
        cols = rng.integers(0, len(disc_spans), size=b)
        for i, ci in enumerate(cols):
            st, w = disc_spans[ci]
            k = int(rows_encoded[i, st:st + w].argmax())
            cond[i, offsets[ci] + k] = 1.0
        return cond

    def fit(self, df: pd.DataFrame,
            discrete_columns: Optional[Sequence[str]] = None,
            groups: Optional[Iterable] = None) -> "DPCTGAN":
        self._check_input(df)
        if discrete_columns is None:
            discrete_columns = infer_discrete_columns(df)
        self.discrete_columns_ = list(discrete_columns)
        self.columns_ = list(df.columns)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.transformer = DataTransformer(self.vgm, seed=self.seed).fit(
            df, discrete_columns)
        data = self.transformer.transform(df)
        spans = self.transformer.output_spans()
        disc_spans = self.transformer.discrete_spans()
        n, data_dim = len(data), self.transformer.output_dim
        data_t = torch.from_numpy(data).to(self.device)

        self.cond = CondSampler(data, disc_spans, seed=self.seed)
        sigma_hist = None
        if disc_spans:
            self.cond.cat_logfreq = self._dp_cond_freqs(data, disc_spans, rng)
            sigma_hist = self.hist_noise_multiplier
        cond_dim = self.cond.cond_dim
        offsets = (np.cumsum([0] + [w for _, w in disc_spans])[:-1]
                   if disc_spans else None)

        steps_per_epoch = max(n // self.batch, 1)
        total_steps = self.epochs * steps_per_epoch * self.d_steps
        q = self.batch / n
        self.sigma_ = (float(self.noise_multiplier) if self.noise_multiplier
                       else calibrate_sigma(self.target_epsilon, q,
                                            total_steps, self.target_delta,
                                            sigma_hist=sigma_hist))
        if self.verbose:
            print(f"  DP-SGD: n={n:,} q={q:.4f} steps={total_steps} "
                  f"sigma={self.sigma_:.3f}", flush=True)

        self.G = Generator(self.z_dim, cond_dim, self.gdim, data_dim
                           ).to(self.device)
        self.D = Discriminator(data_dim + cond_dim, self.ddim, self.pac
                               ).to(self.device)
        optG = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.9),
                                weight_decay=1e-6)
        optD = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.9))

        noise_std = self.sigma_ * 2.0 * self.clip_norm
        n_groups = self.batch // self.pac
        steps_done = 0
        for epoch in range(self.epochs):
            g_running = d_running = 0.0
            for _ in range(steps_per_epoch):
                for _ in range(self.d_steps):     # critic: DP-SGD over groups
                    with torch.no_grad():
                        z = torch.randn(self.batch, self.z_dim,
                                        device=self.device)
                        c, _, _, _ = self.cond.sample(self.batch)
                        c_fake = (torch.from_numpy(c).to(self.device)
                                  if c is not None else None)
                        gin = (torch.cat([z, c_fake], 1)
                               if c_fake is not None else z)
                        fake = _activate(self.G(gin), spans)

                    real_idx = (rng.choice(n, size=self.batch, replace=False)
                                if n >= self.batch
                                else rng.integers(0, n, self.batch))
                    real = data_t[torch.from_numpy(real_idx).to(self.device)]
                    if disc_spans:
                        c_real = torch.from_numpy(self._self_cond(
                            data[real_idx], disc_spans, offsets, rng)
                        ).to(self.device)
                        real_in = torch.cat([real, c_real], 1)
                        fake_in = torch.cat([fake, c_fake], 1)
                    else:
                        real_in, fake_in = real, fake

                    agg = [torch.zeros_like(p) for p in self.D.parameters()]
                    d_loss_val = 0.0
                    for g in range(n_groups):
                        sl = slice(g * self.pac, (g + 1) * self.pac)
                        rg, fg = real_in[sl], fake_in[sl]
                        loss_g = (self.D(fg).mean() - self.D(rg).mean()
                                  + self._gp(rg, fg))
                        grads = torch.autograd.grad(loss_g,
                                                    list(self.D.parameters()))
                        gnorm = torch.sqrt(sum((gr ** 2).sum()
                                               for gr in grads))
                        scale = min(1.0, self.clip_norm / (float(gnorm) + 1e-12))
                        for a, gr in zip(agg, grads):
                            a.add_(gr, alpha=scale)
                        d_loss_val += float(loss_g.detach())
                    optD.zero_grad(set_to_none=True)
                    for p, a in zip(self.D.parameters(), agg):
                        p.grad = (a + noise_std * torch.randn_like(a)) / n_groups
                    optD.step()
                    d_running += d_loss_val / n_groups
                    steps_done += 1

                # generator update: post-processing, no real data
                z = torch.randn(self.batch, self.z_dim, device=self.device)
                c, m, col_ids, cat = self.cond.sample(self.batch)
                if c is not None:
                    c_t = torch.from_numpy(c).to(self.device)
                    gen = self.G(torch.cat([z, c_t], 1))
                    fake = _activate(gen, spans)
                    dfake = self.D(torch.cat([fake, c_t], 1))
                    ce = self._cond_ce(gen, m, col_ids, cat, disc_spans,
                                       np.cumsum([0] + [w for _, w
                                                        in disc_spans]))
                else:
                    gen = self.G(z); fake = _activate(gen, spans)
                    dfake = self.D(fake); ce = 0.0
                lossG = -dfake.mean() + ce
                optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()
                g_running += float(lossG)

            self.loss_hist.append((g_running / steps_per_epoch,
                                   d_running / steps_per_epoch))
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:3d}/{self.epochs}", flush=True)

        self.privacy_ = Run(q, self.sigma_, steps_done, sigma_hist)
        self.epsilon_ = compose([self.privacy_], self.target_delta)
        self._release_training_data()
        return self

    # -- fair sampling: zero additional epsilon -------------------------------
    def fair_plan(self, sensitive_columns: Sequence[str],
                  min_support: float = 0.02) -> list:
        """(column, target distribution) pairs from the released noisy shares:
        uniform over categories with at least ``min_support`` share, rarer
        categories keep their share."""
        return [(c, fair_target(self.noisy_shares_[c], min_support))
                for c in sensitive_columns]

    @torch.no_grad()
    def sample_fair(self, n: int, plan: list) -> pd.DataFrame:
        """Sample n rows, conditioning round-robin on ``plan`` =
        [(column, target probability vector), ...]."""
        names = [c.name for c in self.transformer.columns
                 if c.ctype == "discrete"]
        spans = self.transformer.output_spans()
        disc_spans = self.transformer.discrete_spans()
        offsets = np.cumsum([0] + [w for _, w in disc_spans])[:-1]
        col_idx = {c: i for i, c in enumerate(names)}
        rng = np.random.default_rng(self.seed + 1)
        self.G.eval()
        out, done, turn = [], 0, 0
        while done < n:
            bb = self.batch
            cond = np.zeros((bb, self.cond.cond_dim), dtype="float32")
            for i in range(bb):
                col, probs = plan[turn % len(plan)]; turn += 1
                ci = col_idx[col]
                cond[i, offsets[ci] + rng.choice(len(probs), p=probs)] = 1.0
            z = torch.randn(bb, self.z_dim, device=self.device)
            gen = self.G(torch.cat([z, torch.from_numpy(cond).to(self.device)],
                                   1))
            take = min(bb, n - done)
            out.append(_activate(gen, spans, hard=True).cpu().numpy()[:take])
            done += take
        self.G.train()
        return self.transformer.inverse_transform(
            np.concatenate(out, axis=0)[:n])

    def sample_balanced(self, n: int, sensitive_columns: Sequence[str],
                        min_support: float = 0.02) -> pd.DataFrame:
        """Release n rows whose sensitive-column composition is uniform over
        adequately supported groups. Post-processing: no extra epsilon."""
        return self.sample_fair(n, self.fair_plan(sensitive_columns,
                                                  min_support))
