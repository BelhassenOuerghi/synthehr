"""
fair_dp_ctgan.py
----------------
**Fair-DP-CTGAN**: bias mitigation layered on the Stage-1 DP generator, using
the hook Stage 0 chose CTGAN for, its conditional mechanism, and the hook
Stage 1 prepared, the DP-accounted conditional frequencies.

Two mitigation arms, deliberately different in where they intervene:

1. **Fair sampling (generation-time, zero additional privacy cost).** The
   trained generator honours its conditional vector (the conditional
   cross-entropy term forces it to), so the *released* dataset's subgroup
   composition is a free choice at sampling time. `sample_fair()` conditions
   every draw on a sensitive column, with categories drawn from a *target*
   distribution (default: uniform over adequately-supported groups). Because
   the DP-trained generator is post-processing-safe, re-weighting its
   conditioning consumes **no additional ε**, the central design argument of
   this stage.

2. **Fair training (training-time).** Training-by-sampling normally draws the
   conditioned category ∝ (DP-noised) log-frequency, so minority categories
   are seen less often. `fair_train_cols` replaces those weights with a
   uniform distribution for the sensitive columns, a data-independent choice,
   so again no additional privacy cost, making the critic/generator pair see
   minority subgroups as often as majority ones during fitting.

Eligibility floor: a category is balanced only if its **DP-noised** share is
at least `min_support` (tiny groups, e.g. 'No charge' insurance at ≈0.03%,
would otherwise be inflated by orders of magnitude into synthetic populations
the generator has never meaningfully seen). Ineligible categories keep their
noisy-share probability. All shares used here come from the same noisy-count
release Stage 1 already accounts for, no new real-data statistic is touched.

Scope note (documented in the paper): with several sensitive columns the plan
round-robins single-column conditioning, balancing each *marginal*;
intersectional (joint) balancing is future work.
"""
from __future__ import annotations

import numpy as np
import torch

from ctgan import _activate
from dp_ctgan import DPCTGAN


def fair_target(noisy_shares, min_support=0.02):
    """Uniform over adequately-supported categories; rare ones keep their
    (noisy) share. Input and output are probability vectors."""
    p = np.asarray(noisy_shares, dtype=float)
    p = p / p.sum()
    eligible = p >= min_support
    if eligible.sum() <= 1:
        return p
    rare_mass = p[~eligible].sum()
    out = np.where(eligible, (1.0 - rare_mass) / eligible.sum(), p)
    return out / out.sum()


class FairDPCTGAN(DPCTGAN):
    """DP-CTGAN with fair conditional training and/or fair sampling."""

    def __init__(self, *, fair_train_cols=None, **dp_kwargs):
        super().__init__(**dp_kwargs)
        self.fair_train_cols = list(fair_train_cols or [])
        self.noisy_shares_ = {}          # col name -> noisy share vector

    # discrete column names, in the order of discrete_spans()
    def _disc_names(self):
        return [c.name for c in self.transformer.columns
                if c.ctype == "discrete"]

    def _dp_cond_freqs(self, data, disc_spans, rng):
        logfreq = super()._dp_cond_freqs(data, disc_spans, rng)
        names = self._disc_names()
        # recover noisy shares from the same release (post-processing) and
        # flatten the training weights for the fair-trained columns
        n_disc = max(len(disc_spans), 1)
        std = self.hist_noise_multiplier * np.sqrt(n_disc)
        for i, (st, w) in enumerate(disc_spans):
            counts = data[:, st:st + w].sum(axis=0)
            noisy = np.maximum(counts + rng.normal(0.0, std, size=w), 1.0)
            self.noisy_shares_[names[i]] = noisy / noisy.sum()
            if names[i] in self.fair_train_cols:
                logfreq[i] = np.ones(w) / w          # data-independent
        return logfreq

    @torch.no_grad()
    def sample_fair(self, n, plan):
        """Sample n rows, conditioning round-robin on `plan` =
        [(col_name, target_prob_vector), ...]."""
        names = self._disc_names()
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
                k = rng.choice(len(probs), p=probs)
                cond[i, offsets[ci] + k] = 1.0
            z = torch.randn(bb, self.z_dim, device=self.device)
            c_t = torch.from_numpy(cond).to(self.device)
            gen = self.G(torch.cat([z, c_t], 1))
            fake = _activate(gen, spans, hard=True).cpu().numpy()
            take = min(bb, n - done)
            out.append(fake[:take]); done += take
        self.G.train()
        return self.transformer.inverse_transform(
            np.concatenate(out, axis=0)[:n])

    def build_plan(self, sensitive_cols, min_support=0.02):
        """(col, target) pairs from the DP-noised shares, no new data touched."""
        return [(c, fair_target(self.noisy_shares_[c], min_support))
                for c in sensitive_cols]
