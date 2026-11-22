"""
CTGAN: the Conditional Tabular GAN of Xu, Skoularidou, Cuesta-Infante &
Veeramachaneni (NeurIPS 2019), implemented from scratch.

Faithful ingredients:

* mode-specific normalisation of numeric columns with a per-column variational
  Gaussian mixture;
* a conditional generator trained by sampling a discrete column and category
  (training-by-sampling, categories drawn in proportion to log-frequency);
* a WGAN-GP critic with PacGAN packing, and the generator's conditional
  cross-entropy term.

This is the programme's Stage 0 generator. Its numerics are identical to the
frozen research implementation (``research/common/ctgan.py``); the test suite
checks bit-for-bit equality of samples for a fixed seed.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import BayesianGaussianMixture

from synthehr.generators._base import (TabularGenerator,
                                       infer_discrete_columns, resolve_device)


# ---------------------------------------------------------------------------
# Encoding: mode-specific normalisation and one-hot
# ---------------------------------------------------------------------------
class _ColumnInfo:
    def __init__(self, name, ctype, dim, gm=None, active=None, categories=None):
        self.name = name
        self.ctype = ctype            # "continuous" | "discrete"
        self.dim = dim                # width in the encoded matrix
        self.gm = gm                  # fitted mixture (continuous)
        self.active = active          # retained mixture modes (continuous)
        self.categories = categories  # category order (discrete)


class DataTransformer:
    """Encode a mixed-type DataFrame into CTGAN's training representation:
    each numeric value becomes (scalar within its mode, one-hot of the mode);
    each category becomes a one-hot block."""

    def __init__(self, vgm_components: int = 10,
                 weight_threshold: float = 5e-3, seed: int = 0):
        self.vgm_components = vgm_components
        self.weight_threshold = weight_threshold
        self.seed = seed
        self.columns: list[_ColumnInfo] = []
        self.output_dim = 0

    def fit(self, df: pd.DataFrame, discrete_columns: Sequence[str]
            ) -> "DataTransformer":
        self.columns = []
        for name in df.columns:
            col = df[name].to_numpy()
            if name in discrete_columns:
                cats = sorted(df[name].astype(str).unique().tolist())
                self.columns.append(_ColumnInfo(name, "discrete", len(cats),
                                                categories=cats))
            else:
                x = col.astype(float).reshape(-1, 1)
                gm = BayesianGaussianMixture(
                    n_components=self.vgm_components,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=1e-3,
                    max_iter=100, n_init=1, random_state=self.seed)
                gm.fit(x)
                active_idx = np.where(gm.weights_ > self.weight_threshold)[0]
                if len(active_idx) == 0:             # degenerate column
                    active_idx = np.array([int(np.argmax(gm.weights_))])
                self.columns.append(_ColumnInfo(
                    name, "continuous", 1 + len(active_idx), gm=gm,
                    active=active_idx))
        self.output_dim = sum(c.dim for c in self.columns)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = []
        for c in self.columns:
            col = df[c.name].to_numpy()
            if c.ctype == "discrete":
                idx = {v: i for i, v in enumerate(c.categories)}
                oh = np.zeros((len(df), c.dim), dtype="float32")
                for i, v in enumerate(col.astype(str)):
                    oh[i, idx.get(v, 0)] = 1.0
                out.append(oh)
            else:
                x = col.astype(float).reshape(-1, 1)
                means = c.gm.means_.reshape(-1)[c.active]
                stds = np.sqrt(c.gm.covariances_.reshape(-1))[c.active]
                probs = np.nan_to_num(c.gm.predict_proba(x)[:, c.active],
                                      nan=0.0)
                row_sum = probs.sum(axis=1)
                # values far from every mode get ~0 mass: assign the nearest
                bad = row_sum <= 1e-12
                if bad.any():
                    nearest = np.abs(x[bad] - means.reshape(1, -1)).argmin(axis=1)
                    probs[bad] = 0.0
                    probs[np.where(bad)[0], nearest] = 1.0
                    row_sum = probs.sum(axis=1)
                probs = probs / row_sum[:, None]
                rng = np.random.default_rng(self.seed)
                modes = np.array([rng.choice(len(c.active), p=probs[i])
                                  for i in range(len(x))])
                alpha = ((x.reshape(-1) - means[modes]) /
                         (4.0 * stds[modes])).clip(-0.99, 0.99)
                beta = np.zeros((len(df), len(c.active)), dtype="float32")
                beta[np.arange(len(df)), modes] = 1.0
                out.append(np.concatenate(
                    [alpha.reshape(-1, 1).astype("float32"), beta], axis=1))
        return np.concatenate(out, axis=1).astype("float32")

    def inverse_transform(self, data: np.ndarray) -> pd.DataFrame:
        cols, pos = {}, 0
        for c in self.columns:
            chunk = data[:, pos:pos + c.dim]
            pos += c.dim
            if c.ctype == "discrete":
                cols[c.name] = np.array(c.categories, dtype=object)[
                    chunk.argmax(axis=1)]
            else:
                means = c.gm.means_.reshape(-1)[c.active]
                stds = np.sqrt(c.gm.covariances_.reshape(-1))[c.active]
                alpha = chunk[:, 0].clip(-1, 1)
                modes = chunk[:, 1:].argmax(axis=1)
                cols[c.name] = alpha * 4.0 * stds[modes] + means[modes]
        return pd.DataFrame(cols)

    def output_spans(self) -> list[tuple[int, int, str]]:
        """(start, width, activation) of every output block."""
        spans, pos = [], 0
        for c in self.columns:
            if c.ctype == "discrete":
                spans.append((pos, c.dim, "softmax")); pos += c.dim
            else:
                spans.append((pos, 1, "tanh"))
                spans.append((pos + 1, c.dim - 1, "softmax")); pos += c.dim
        return spans

    def discrete_spans(self) -> list[tuple[int, int]]:
        """(start, width) of each discrete one-hot block."""
        spans, pos = [], 0
        for c in self.columns:
            if c.ctype == "discrete":
                spans.append((pos, c.dim))
            pos += c.dim
        return spans


# ---------------------------------------------------------------------------
# Conditional sampler (training-by-sampling)
# ---------------------------------------------------------------------------
class CondSampler:
    """Draw conditional vectors (a discrete column and one of its categories,
    the category in proportion to its log-frequency) and matching real rows."""

    def __init__(self, data: np.ndarray, discrete_spans, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.spans = discrete_spans
        self.n_disc = len(discrete_spans)
        self.cond_dim = sum(w for _, w in discrete_spans)
        self.cat_rows, self.cat_logfreq = [], []
        for (st, w) in discrete_spans:
            rows_by_cat, freqs = [], []
            block = data[:, st:st + w]
            for k in range(w):
                idx = np.where(block[:, k] > 0.5)[0]
                rows_by_cat.append(idx)
                freqs.append(max(len(idx), 1))
            self.cat_rows.append(rows_by_cat)
            lf = np.log(np.array(freqs, dtype=float))
            self.cat_logfreq.append(lf / lf.sum() if lf.sum() > 0
                                    else np.ones(w) / w)

    def sample(self, batch: int):
        if self.n_disc == 0:
            return None, None, None, None
        cond = np.zeros((batch, self.cond_dim), dtype="float32")
        mask = np.zeros((batch, self.n_disc), dtype="float32")
        col_ids = self.rng.integers(0, self.n_disc, size=batch)
        rows = np.empty(batch, dtype=int)
        offsets = np.cumsum([0] + [w for _, w in self.spans])
        cat_choice = np.empty(batch, dtype=int)
        for i, ci in enumerate(col_ids):
            p = self.cat_logfreq[ci]
            k = self.rng.choice(len(p), p=p)
            cat_choice[i] = k
            cond[i, offsets[ci] + k] = 1.0
            mask[i, ci] = 1.0
            pool = self.cat_rows[ci][k]
            rows[i] = pool[self.rng.integers(0, len(pool))] if len(pool) else \
                self.rng.integers(0, 10)
        return cond, mask, col_ids, cat_choice

    def sample_zero(self, batch: int):
        """Conditional vectors for generation (no matching rows needed)."""
        if self.n_disc == 0:
            return None
        cond = np.zeros((batch, self.cond_dim), dtype="float32")
        offsets = np.cumsum([0] + [w for _, w in self.spans])
        for i in range(batch):
            ci = self.rng.integers(0, self.n_disc)
            p = self.cat_logfreq[ci]
            k = self.rng.choice(len(p), p=p)
            cond[i, offsets[ci] + k] = 1.0
        return cond.astype("float32")


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class _Residual(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)

    def forward(self, x):
        return torch.cat([F.relu(self.bn(self.fc(x))), x], dim=1)


class Generator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, hidden, data_dim: int):
        super().__init__()
        seq, d = [], z_dim + cond_dim
        for h in hidden:
            seq.append(_Residual(d, h)); d += h
        seq.append(nn.Linear(d, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden, pac: int = 10):
        super().__init__()
        self.pac = pac
        d, seq = in_dim * pac, []
        for h in hidden:
            seq += [nn.Linear(d, h), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            d = h
        seq.append(nn.Linear(d, 1))
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x.view(-1, x.size(1) * self.pac))


def _activate(data, spans, tau: float = 0.2, hard: bool = False):
    """tanh on scalars, Gumbel-softmax on one-hot blocks."""
    out = []
    for (st, w, kind) in spans:
        chunk = data[:, st:st + w]
        out.append(torch.tanh(chunk) if kind == "tanh"
                   else F.gumbel_softmax(chunk, tau=tau, hard=hard))
    return torch.cat(out, dim=1)


# ---------------------------------------------------------------------------
# CTGAN
# ---------------------------------------------------------------------------
class CTGAN(TabularGenerator):
    """Conditional tabular GAN.

    Parameters mirror the original paper; the defaults are the ones the
    programme used (Stage 0). ``epochs`` counts passes over the fitting data.
    """

    def __init__(self, embedding_dim: int = 128,
                 generator_dim: Sequence[int] = (256, 256),
                 discriminator_dim: Sequence[int] = (256, 256),
                 batch_size: int = 500, epochs: int = 200, pac: int = 10,
                 gp_lambda: float = 10.0, vgm_components: int = 10,
                 d_steps: int = 1, device: str = "cpu", seed: int = 0,
                 verbose: bool = False):
        self.z_dim = embedding_dim
        self.gdim, self.ddim = tuple(generator_dim), tuple(discriminator_dim)
        self.batch = (batch_size // pac) * pac
        self.epochs, self.pac = epochs, pac
        self.gp_lambda, self.vgm = gp_lambda, vgm_components
        self.d_steps = d_steps
        self.device = resolve_device(device)
        self.seed, self.verbose = seed, verbose
        self.transformer: Optional[DataTransformer] = None
        self.cond: Optional[CondSampler] = None
        self.G = None
        self.loss_hist: list[tuple[float, float]] = []

    # gradient penalty on PacGAN-packed interpolates
    def _gp(self, real, fake):
        a = torch.rand(real.size(0) // self.pac, 1, 1, device=self.device)
        a = a.repeat(1, self.pac, real.size(1)).view(-1, real.size(1))
        interp = (a * real + (1 - a) * fake).requires_grad_(True)
        d = self.D(interp)
        g = torch.autograd.grad(d, interp, torch.ones_like(d),
                                create_graph=True, retain_graph=True)[0]
        g = g.view(-1, self.pac * real.size(1))
        return ((g.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda

    def fit(self, df: pd.DataFrame,
            discrete_columns: Optional[Sequence[str]] = None,
            groups: Optional[Iterable] = None) -> "CTGAN":
        self._check_input(df)
        if discrete_columns is None:
            discrete_columns = infer_discrete_columns(df)
        self.discrete_columns_ = list(discrete_columns)
        self.columns_ = list(df.columns)
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.transformer = DataTransformer(self.vgm, seed=self.seed).fit(
            df, discrete_columns)
        data = self.transformer.transform(df)
        spans = self.transformer.output_spans()
        disc_spans = self.transformer.discrete_spans()
        self.cond = CondSampler(data, disc_spans, seed=self.seed)
        cond_dim = self.cond.cond_dim
        data_dim = self.transformer.output_dim
        data_t = torch.from_numpy(data).to(self.device)

        self.G = Generator(self.z_dim, cond_dim, self.gdim, data_dim
                           ).to(self.device)
        self.D = Discriminator(data_dim + cond_dim, self.ddim, self.pac
                               ).to(self.device)
        optG = torch.optim.Adam(self.G.parameters(), lr=2e-4, betas=(0.5, 0.9),
                                weight_decay=1e-6)
        optD = torch.optim.Adam(self.D.parameters(), lr=2e-4, betas=(0.5, 0.9),
                                weight_decay=1e-6)

        n = len(data)
        steps = max(n // self.batch, 1)
        rng = np.random.default_rng(self.seed)
        offsets = (np.cumsum([0] + [w for _, w in disc_spans])
                   if disc_spans else None)

        for epoch in range(self.epochs):
            g_running = d_running = 0.0
            for _ in range(steps):
                for _ in range(self.d_steps):                    # critic
                    z = torch.randn(self.batch, self.z_dim, device=self.device)
                    c, m, col_ids, cat = self.cond.sample(self.batch)
                    if c is not None:
                        c_t = torch.from_numpy(c).to(self.device)
                        fake = _activate(self.G(torch.cat([z, c_t], 1)), spans)
                        fake_cat = torch.cat([fake, c_t], 1)
                        real_rows = data_t[torch.from_numpy(
                            self._match_rows(m, col_ids, cat)).to(self.device)]
                        real_cat = torch.cat([real_rows, c_t], 1)
                    else:
                        z = torch.randn(self.batch, self.z_dim,
                                        device=self.device)
                        fake_cat = _activate(self.G(z), spans)
                        real_cat = data_t[torch.from_numpy(
                            rng.integers(0, n, self.batch)).to(self.device)]
                    dfake = self.D(fake_cat); dreal = self.D(real_cat)
                    lossD = (dfake.mean() - dreal.mean()
                             + self._gp(real_cat, fake_cat))
                    optD.zero_grad(set_to_none=True); lossD.backward()
                    optD.step()
                    d_running += float(lossD)

                z = torch.randn(self.batch, self.z_dim, device=self.device)
                c, m, col_ids, cat = self.cond.sample(self.batch)  # generator
                if c is not None:
                    c_t = torch.from_numpy(c).to(self.device)
                    gen = self.G(torch.cat([z, c_t], 1))
                    fake = _activate(gen, spans)
                    dfake = self.D(torch.cat([fake, c_t], 1))
                    ce = self._cond_ce(gen, m, col_ids, cat, disc_spans,
                                       offsets)
                else:
                    gen = self.G(z); fake = _activate(gen, spans)
                    dfake = self.D(fake); ce = 0.0
                lossG = -dfake.mean() + ce
                optG.zero_grad(set_to_none=True); lossG.backward(); optG.step()
                g_running += float(lossG)

            self.loss_hist.append((g_running / steps, d_running / steps))
            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:3d}/{self.epochs}  "
                      f"lossG {g_running/steps:+.3f}  "
                      f"lossD {d_running/steps:+.3f}", flush=True)
        self._release_training_data()
        return self

    def _release_training_data(self):
        """Drop what sampling does not need, so that a saved model carries
        neither the critic nor the per-category row indices (which hold the
        exact category counts of the training data)."""
        self.D = None
        if self.cond is not None:
            self.cond.cat_rows = None

    def _match_rows(self, mask, col_ids, cat):
        """Real rows whose conditioned column equals the sampled category."""
        return np.array([self.cond.cat_rows[ci][k][
            np.random.randint(len(self.cond.cat_rows[ci][k]))]
            if len(self.cond.cat_rows[ci][k]) else 0
            for ci, k in zip(col_ids, cat)])

    def _cond_ce(self, gen, mask, col_ids, cat, disc_spans, offsets):
        """Cross-entropy that the generated conditioned column matches."""
        if not len(disc_spans):
            return 0.0
        full_offsets, pos = [], 0
        for c in self.transformer.columns:
            if c.ctype == "discrete":
                full_offsets.append((pos, c.dim))
            pos += c.dim
        loss = 0.0
        for b in range(len(col_ids)):
            st, w = full_offsets[col_ids[b]]
            logits = gen[b, st:st + w].unsqueeze(0)
            target = torch.tensor([cat[b]], device=self.device)
            loss = loss + F.cross_entropy(logits, target)
        return loss / len(col_ids)

    @torch.no_grad()
    def sample(self, n: int) -> pd.DataFrame:
        if self.G is None:
            raise RuntimeError("call fit() before sample()")
        self.G.eval()
        spans = self.transformer.output_spans()
        out, done = [], 0
        while done < n:
            b = min(self.batch, n - done)
            bb = self.batch                 # full batch for BatchNorm, trimmed
            z = torch.randn(bb, self.z_dim, device=self.device)
            c = self.cond.sample_zero(bb)
            gen = (self.G(torch.cat([z, torch.from_numpy(c).to(self.device)],
                                    1)) if c is not None else self.G(z))
            out.append(_activate(gen, spans, hard=True).cpu().numpy()[:b])
            done += b
        self.G.train()
        return self.transformer.inverse_transform(
            np.concatenate(out, axis=0)[:n])
