"""Abstract TSFM interface, a dependency-free placeholder model, and the dispatcher.

`forward` returns `(pred, loc, scale)`. A model that normalizes internally -- Toto-2
does -- reports the location and scale it used so the trainer can move targets into
that same space, which is where losses and CRPS are scored. One that does not
normalize returns `loc = 0`, `scale = 1` and nothing changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

Tensor = torch.Tensor


class TSFM(nn.Module, ABC):
    """Next-patch predictor: output at patch k predicts patch k+1."""

    patch_length: int
    n_quantiles: int  # 0 for a point head

    @abstractmethod
    def forward(
        self, y: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor | float, Tensor | float]:
        """`y` is (B, T); `mask` marks positions hidden from the model.

        Returns `(pred, loc, scale)` with `pred` of shape (B, T) for a point head or
        (B, T, K) for a quantile head, aligned so index t predicts `y[:, t]`.
        """

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PatchedTransformer(TSFM):
    """PatchTST-style encoder: a placeholder so the pipeline runs with no optional
    dependencies. The paper's figures use Toto-2 (`toto2.py`), not this."""

    def __init__(
        self,
        patch_length: int = 32,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_quantiles: int = 9,
        dropout: float = 0.0,
        max_patches: int = 256,
    ):
        super().__init__()
        self.patch_length = patch_length
        self.n_quantiles = n_quantiles
        out = patch_length * max(n_quantiles, 1)
        # Two input channels per position, value and mask flag, so the model can tell
        # an imputed zero from a real zero.
        self.embed = nn.Linear(2 * patch_length, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, out)

    def forward(self, y: Tensor, mask: Tensor | None = None):
        B, T = y.shape
        P = self.patch_length
        n = T // P
        m = torch.zeros_like(y) if mask is None else mask.float()
        x = torch.stack([y * (1 - m), m], dim=-1).reshape(B, n, 2 * P)
        o = self.head(self.body(self.embed(x) + self.pos[:, :n]))
        if self.n_quantiles:
            o = o.reshape(B, n * P, self.n_quantiles)
        else:
            o = o.reshape(B, n * P)
        # Shift by one patch: output at patch k predicts patch k+1.
        pred = torch.cat([torch.zeros_like(o[:, :P]), o[:, :-P]], dim=1)
        return pred, 0.0, 1.0


PATCHED_SIZES: dict[str, dict] = {
    "tiny": dict(d_model=128, n_layers=2, n_heads=2),
    "small": dict(d_model=256, n_layers=4, n_heads=4),
    "base": dict(d_model=512, n_layers=6, n_heads=8),
    "large": dict(d_model=768, n_layers=8, n_heads=12),
    "xl": dict(d_model=1024, n_layers=12, n_heads=16),
}


def build(
    arch: str,
    size: str,
    context_length: int,
    patch_length: int = 32,
    n_quantiles: int = 9,
) -> TSFM:
    """`arch` is "toto2" (the paper's models) or "patched" (the placeholder)."""
    if arch == "toto2":
        from .toto2 import build as build_toto2

        return build_toto2(size, context_length, n_quantiles)
    if arch == "patched":
        return PatchedTransformer(
            patch_length=patch_length, n_quantiles=n_quantiles, **PATCHED_SIZES[size]
        )
    raise ValueError(f"unknown arch {arch!r}")


def sizes(arch: str) -> list[str]:
    if arch == "toto2":
        from .toto2 import TOTO2_SIZES

        return list(TOTO2_SIZES)
    return list(PATCHED_SIZES)
