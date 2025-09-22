"""
fuka5.substrate.decoders
------------------------
Local, lightweight decoders for reconstructing source signals using only
edge-local features (no backprop). Implements simple LMS heads:

  y_hat = X @ w
  w <- w + eta_w * X^T (y - y_hat) / T

Where:
  - X : (T, F) features built from edge time series (per epoch window)
  - y : (T,) target signal (sum of source tones for the selected subset)

We provide:
  * feature_builder(v_te): produce edge-local features from edge time traces
  * DecoderHead(name, F): an LMS head with update/predict
  * DecoderBank: manages multiple heads (s1, s1p, fused)

Notes
-----
- This is deliberately simple: features are linear in v_e(t). You can extend
  to include nonlinear combos if needed.
- Targets y are synthesized in the runner (sim_cli) using the configured tones,
  restricted to the subset of sources per head.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np


# ---------------------------
# Feature builder
# ---------------------------

def feature_builder(v_te_edge: np.ndarray) -> np.ndarray:
    """
    Build per-edge features from an edge time series (T,).
    Current set (F=3):
      f0 = v(t)
      f1 = diff(v)(t) ~ discrete derivative
      f2 = ema(v)(t)  ~ slow low-pass
    Returns X with shape (T, F).
    """
    v = v_te_edge.astype(np.float32, copy=False)
    T = v.shape[0]
    if T < 3:
        return np.stack([v, v*0, v], axis=1)

    # Discrete derivative
    dv = np.zeros_like(v)
    dv[1:] = v[1:] - v[:-1]

    # EMA
    ema = np.zeros_like(v)
    alpha = 0.02
    acc = 0.0
    for t in range(T):
        acc = (1 - alpha) * acc + alpha * v[t]
        ema[t] = acc

    X = np.stack([v, dv, ema], axis=1)
    return X


# ---------------------------
# LMS head
# ---------------------------

@dataclass
class DecoderHead:
    name: str
    F: int
    eta_w: float
    w: np.ndarray

    @staticmethod
    def create(name: str, F: int, eta_w: float, rng: Optional[np.random.Generator] = None) -> "DecoderHead":
        rng = rng or np.random.default_rng(0)
        w0 = (rng.standard_normal(F) * 1e-3).astype(np.float32)
        return DecoderHead(name=name, F=F, eta_w=float(eta_w), w=w0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (T, F) -> y_hat: (T,)"""
        return X @ self.w

    def update(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        One LMS step; returns MSE.
        """
        assert X.shape[1] == self.F
        yhat = X @ self.w
        e = y - yhat
        mse = float(np.mean(e * e))
        # normalized by T to keep eta scale stable
        self.w = (self.w + self.eta_w * (X.T @ e) / max(1, X.shape[0])).astype(np.float32, copy=False)
        return mse


# ---------------------------
# Decoder bank
# ---------------------------

class DecoderBank:
    """
    Manages per-edge heads for: 's1', 's1p', 'mix'.
    In this prototype we keep one head per run (not per edge) that sees the
    concatenation of all edge features. This keeps the system light while still
    being "local" in the sense of using only edge-observable signals.

    If you want strictly per-edge heads, you can instantiate DecoderHead per edge
    and aggregate their y_hats; left as an optional extension.
    """

    def __init__(self, n_edges: int, eta_w: float, rng: Optional[np.random.Generator] = None):
        self.n_edges = int(n_edges)
        self.F_edge = 3  # from feature_builder
        self.F_total = self.F_edge * self.n_edges
        self.rng = rng or np.random.default_rng(0)

        self.heads: Dict[str, DecoderHead] = {
            "s1":  DecoderHead.create("s1",  self.F_total, eta_w, self.rng),
            "s1p": DecoderHead.create("s1p", self.F_total, eta_w, self.rng),
            "mix": DecoderHead.create("mix", self.F_total, eta_w, self.rng),
        }

    def _stack_features_all_edges(self, v_te: np.ndarray) -> np.ndarray:
        """
        v_te: (T, E) → X: (T, F_total), concatenating features for each edge.
        """
        T, E = v_te.shape
        assert E == self.n_edges
        feats = []
        for e in range(E):
            X_e = feature_builder(v_te[:, e])  # (T, F_edge)
            feats.append(X_e)
        X = np.concatenate(feats, axis=1)  # (T, F_total)
        return X

    def predict(self, v_te: np.ndarray, head: str) -> np.ndarray:
        X = self._stack_features_all_edges(v_te)
        return self.heads[head].predict(X)

    def update(self, v_te: np.ndarray, target: np.ndarray, head: str) -> float:
        X = self._stack_features_all_edges(v_te)
        return self.heads[head].update(X, target)