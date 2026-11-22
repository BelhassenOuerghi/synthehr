"""
dp_ctgan.py
-----------
**DP-CTGAN**: the Stage-0 CTGAN trained under (ε, δ)-differential privacy.

Only the critic ever touches real data, the generator learns exclusively
through the critic's gradients, so privatising the critic's training suffices:
by the post-processing property of DP, the generator, and every sample it ever
produces, inherits the same (ε, δ) guarantee.

Mechanisms, and where every unit of privacy budget goes:

1. **DP-SGD on the critic** (Abadi et al., 2016). Real rows enter a critic step
   in PacGAN groups of `pac`; each group's gradient (WGAN loss + its gradient
   penalty) is clipped to L2 norm `clip_norm`, the clipped group gradients are
   summed, and Gaussian noise with std `σ · 2 · clip_norm` is added
   (replace-one adjacency: one record lives in exactly one group per batch, so
   swapping it moves that group's clipped gradient by at most 2·clip_norm).
   Steps compose under the RDP accountant for the sampled Gaussian mechanism
   with rate q = batch / n. Fixed-size batches drawn without replacement are
   accounted as Poisson subsampling at the same rate, the standard
   approximation, stated in the paper.

2. **Noisy conditional frequencies.** CTGAN's training-by-sampling draws the
   conditioned category ∝ log-frequency, a data-derived statistic. We release
   the per-category counts once through a Gaussian mechanism (one record shifts
   one bin per discrete column, so the concatenated count vector has L2
   sensitivity √n_disc) and use the noisy log-frequencies for both training and
   generation-time conditioning. Costed in the same accountant.

3. **Category-matched real batches are replaced by self-conditioning.** The
   non-private CTGAN picks real rows *by the sampled category*, which breaks
   the uniform-subsampling assumption the accountant needs. Here real rows are
   drawn uniformly, and each row's conditional vector is derived from the row
   itself (a uniformly random discrete column, set to the row's own category),
   the critic still sees consistent (row, condition) pairs, and the
   subsampling amplification remains valid.

Known non-privatised residue (documented in the paper, refinement planned): the
per-column Gaussian-mixture normalisation (VGM) and the discrete category
schema are fitted on real data before training.

The generator update, the conditional cross-entropy term, and sampling touch no
real data and are free post-processing.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ctgan import CTGAN, CondSampler, DataTransformer, Discriminator, \
    Generator, _activate
from privacy_accountant import DEFAULT_ORDERS, calibrate_sigma, \
    compose_epsilon, rdp_sampled_gaussian


def _epsilon_total(q, sigma, steps, sigma_hist, delta, orders=DEFAULT_ORDERS):
    """(ε, δ) of DP-SGD steps + one noisy-histogram release, composed in RDP."""
    best = math.inf
    for a in orders:
        rdp = steps * rdp_sampled_gaussian(q, sigma, a)
        if sigma_hist is not None:
            rdp += rdp_sampled_gaussian(1.0, sigma_hist, a)
        best = min(best, rdp + math.log(1.0 / delta) / (a - 1))
    return best


def _calibrate_sigma_total(target_eps, q, steps, sigma_hist, delta,
                           lo=0.4, hi=200.0, tol=1e-3):
    if _epsilon_total(q, hi, steps, sigma_hist, delta) > target_eps:
        raise ValueError(f"ε={target_eps} unreachable at σ≤{hi}; "
                         "reduce epochs or batch size")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _epsilon_total(q, mid, steps, sigma_hist, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


class DPCTGAN(CTGAN):
    """CTGAN with a DP-SGD critic and end-to-end (ε, δ) accounting."""

    def __init__(self, *, target_epsilon=10.0, target_delta=1e-5,
                 clip_norm=1.0, noise_multiplier=None,
                 hist_noise_multiplier=32.0, **ctgan_kwargs):
        super().__init__(**ctgan_kwargs)
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier      # None -> calibrated
        self.hist_noise_multiplier = hist_noise_multiplier
        self.sigma_ = None
        self.epsilon_spent_ = None

    # -- noisy conditional frequencies (Gaussian mechanism, one release) -----
    def _dp_cond_freqs(self, data, disc_spans, rng):
        noisy_logfreq = []
        n_disc = max(len(disc_spans), 1)
        std = self.hist_noise_multiplier * math.sqrt(n_disc)
        for (st, w) in disc_spans:
            counts = data[:, st:st + w].sum(axis=0)
            counts = np.maximum(counts + rng.normal(0.0, std, size=w), 1.0)
            lf = np.log(counts)
            noisy_logfreq.append(lf / lf.sum() if lf.sum() > 0
                                 else np.ones(w) / w)
        return noisy_logfreq

    # -- conditional vector derived from the real row itself -----------------
    def _self_cond(self, rows_encoded, disc_spans, offsets, rng):
        b = rows_encoded.shape[0]
        cond = np.zeros((b, sum(w for _, w in disc_spans)), dtype="float32")
        cols = rng.integers(0, len(disc_spans), size=b)
        for i, ci in enumerate(cols):
            st, w = disc_spans[ci]
            k = int(rows_encoded[i, st:st + w].argmax())
            cond[i, offsets[ci] + k] = 1.0
        return cond

    def fit(self, df, discrete_columns):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.transformer = DataTransformer(self.vgm, seed=self.seed).fit(
            df, discrete_columns)
        data = self.transformer.transform(df)
        spans = self.transformer.output_spans()
        disc_spans = self.transformer.discrete_spans()
        n, data_dim = len(data), self.transformer.output_dim
        data_t = torch.from_numpy(data).to(self.device)

        # conditional sampler for FAKE rows, with DP-noised frequencies
        self.cond = CondSampler(data, disc_spans, seed=self.seed)
        sigma_hist = None
        if disc_spans:
            self.cond.cat_logfreq = self._dp_cond_freqs(data, disc_spans, rng)
            sigma_hist = self.hist_noise_multiplier
        cond_dim = self.cond.cond_dim
        offsets = np.cumsum([0] + [w for _, w in disc_spans])[:-1] \
            if disc_spans else None

        # ---- privacy budget --------------------------------------------------
        steps_per_epoch = max(n // self.batch, 1)
        total_steps = self.epochs * steps_per_epoch * self.d_steps
        q = self.batch / n
        if self.noise_multiplier is None:
            self.sigma_ = _calibrate_sigma_total(
                self.target_epsilon, q, total_steps, sigma_hist,
                self.target_delta)
        else:
            self.sigma_ = float(self.noise_multiplier)
        if self.verbose:
            print(f"  DP-SGD: n={n:,} q={q:.4f} steps={total_steps} "
                  f"clip={self.clip_norm} sigma={self.sigma_:.3f} "
                  f"(target eps={self.target_epsilon}, "
                  f"delta={self.target_delta})", flush=True)

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
                # ---- critic: DP-SGD over PacGAN-group microbatches ----------
                for _ in range(self.d_steps):
                    with torch.no_grad():
                        z = torch.randn(self.batch, self.z_dim,
                                        device=self.device)
                        c, _, _, _ = self.cond.sample(self.batch)
                        c_fake = torch.from_numpy(c).to(self.device) \
                            if c is not None else None
                        gin = torch.cat([z, c_fake], 1) if c_fake is not None \
                            else z
                        fake = _activate(self.G(gin), spans)

                    real_idx = rng.choice(n, size=self.batch, replace=False) \
                        if n >= self.batch else rng.integers(0, n, self.batch)
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

                # ---- generator: post-processing, exactly as Stage 0 ---------
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

            self.epsilon_spent_ = _epsilon_total(
                q, self.sigma_, steps_done, sigma_hist, self.target_delta)
            self.loss_hist.append((g_running / steps_per_epoch,
                                   d_running / steps_per_epoch))
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:3d}/{self.epochs}  "
                      f"lossG {g_running/steps_per_epoch:+.3f}  "
                      f"lossD {d_running/steps_per_epoch:+.3f}  "
                      f"eps {self.epsilon_spent_:.2f}", flush=True)
        return self
