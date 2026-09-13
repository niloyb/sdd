"""Status Quo and distilled (SDD) losses, and the CRPS metric.

Status Quo scores a prediction against the realized future; SDD scores it against
the conditional law of the future, whose Gaussian marginals `(mu, sd)` come from
the generator. The distilled forms are Table 1 of the paper.
"""

from __future__ import annotations

import math

import torch

Tensor = torch.Tensor
_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: Tensor) -> Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def _Phi(z: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(z / _SQRT2))


# ------------------------------------------------------------------ point head

def mse_realized(pred: Tensor, y: Tensor) -> Tensor:
    return (pred - y).pow(2).mean()


def mse_distilled(pred: Tensor, mu: Tensor, sd: Tensor) -> Tensor:
    """(y_hat - mu)^2 + s^2. The s^2 term is constant in theta but keeps the two
    arms' logged losses on the same scale."""
    return ((pred - mu).pow(2) + sd.pow(2)).mean()


# --------------------------------------------------------------- quantile head

def pinball_realized(q: Tensor, y: Tensor, taus: Tensor) -> Tensor:
    """q: (..., K) predicted quantiles; y: (...) realized."""
    d = y.unsqueeze(-1) - q
    return torch.maximum(taus * d, (taus - 1) * d).mean()


def pinball_distilled(q: Tensor, mu: Tensor, sd: Tensor, taus: Tensor) -> Tensor:
    """s [phi(z) + z (Phi(z) - tau)] with z = (q - mu)/s, integrated exactly."""
    sd = sd.clamp_min(1e-6).unsqueeze(-1)
    z = (q - mu.unsqueeze(-1)) / sd
    return (sd * (_phi(z) + z * (_Phi(z) - taus))).mean()


def crps(q: Tensor, y: Tensor, taus: Tensor) -> Tensor:
    """CRPS as 2/K times the summed pinball loss over K quantile levels."""
    assert q.dim() == y.dim() + 1, (
        f"crps needs a quantile head, q: (..., K) vs y: (...); got q.shape={tuple(q.shape)} "
        f"y.shape={tuple(y.shape)} -- a point forecast has no quantiles to score, use mse instead"
    )
    d = y.unsqueeze(-1) - q
    return 2.0 * torch.maximum(taus * d, (taus - 1) * d).sum(-1).mean() / len(taus)


def make_taus(device, k: int = 9) -> Tensor:
    return torch.tensor([i / (k + 1) for i in range(1, k + 1)], device=device)


# ------------------------------------------------------------------- selection

def objective(
    arm: str,
    pred: Tensor,
    y: Tensor,
    mu: Tensor,
    sd: Tensor,
    taus: Tensor | None,
) -> Tensor:
    """`arm` is "status_quo" or "sdd"; a quantile head is signalled by `taus`."""
    if taus is None:
        return mse_realized(pred, y) if arm == "status_quo" else mse_distilled(pred, mu, sd)
    return (
        pinball_realized(pred, y, taus)
        if arm == "status_quo"
        else pinball_distilled(pred, mu, sd, taus)
    )
