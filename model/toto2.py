"""The five Toto-2 architectures behind the `TSFM` interface.

These are the models the paper's Figures 2 and 3 use. No pre-trained weights are
loaded: the configs are the published ones, instantiated fresh. Seed the global torch
generator before calling `build` -- two arms of a comparison only start from the same
untrained model if their processes drew the same initialization.

Needs the Apache-2.0 `toto-ts` package (https://github.com/DataDog/toto), which is
an optional dependency -- everything else in `sdd` runs without it.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import TSFM

# Fields identical across all five published config.json files.
_COMMON: dict[str, Any] = {
    "patch_size": 32,
    "num_variate_layers_per_group": 1,
    "variate_layer_first": False,
    "num_output_patches": 1,
    "pre_norm": True,
    "attn_bias": True,
    "mlp_bias": False,
    "qk_dim": 64,
    "v_dim": 64,
    "qk_norm": False,
    "qk_norm_include_weight": False,
    "norm_include_weight": False,
    "per_dim_scale": True,
    "use_xpos": True,
    "residual_mult": 0.75,
    "dropout_p": 0.0,
}

# Verbatim from huggingface.co/Datadog/Toto-2.0-<size>/config.json.
TOTO2_SIZES: dict[str, dict[str, Any]] = {
    "4m": dict(d_model=256, d_ff=688, num_heads=4, num_groups=4, num_layers=4,
               layer_group_size=4, norm_eps=1e-4),
    "22m": dict(d_model=512, d_ff=1368, num_heads=8, num_groups=8, num_layers=6,
                layer_group_size=6, norm_eps=1e-4),
    "313m": dict(d_model=1024, d_ff=2736, num_heads=16, num_groups=16, num_layers=24,
                 layer_group_size=24, norm_eps=1e-4),
    "1B": dict(d_model=1536, d_ff=4096, num_heads=24, num_groups=24, num_layers=36,
               layer_group_size=36, norm_eps=5e-4),
    "2.5B": dict(d_model=2048, d_ff=5464, num_heads=32, num_groups=32, num_layers=48,
                 layer_group_size=48, norm_eps=5e-4),
}


def _require_toto2():
    try:
        from toto2.toto2 import Toto2Model, Toto2ModelConfig
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Toto-2 needs the `toto-ts` package: pip install "
            "'toto-ts @ git+https://github.com/DataDog/toto'"
        ) from e
    return Toto2Model, Toto2ModelConfig


def toto2_config(size: str, context_length: int):
    """Published config for `size`, re-balanced for `context_length`.

    `residual_attn_ratio` is the one published field that depends on the number of
    patches rather than the architecture, and every checkpoint ships the value for a
    4096-step context. Reusing it at a shorter context mis-scales the residual stream,
    so it is recomputed.
    """
    if size not in TOTO2_SIZES:
        raise ValueError(f"unknown Toto-2 size {size!r}; pick from {list(TOTO2_SIZES)}")
    _, Toto2ModelConfig = _require_toto2()
    spec = {**_COMMON, **TOTO2_SIZES[size]}
    return Toto2ModelConfig(
        residual_attn_ratio=Toto2ModelConfig.compute_residual_attn_ratio(
            context_length, int(spec["patch_size"])
        ),
        **spec,
    )


class Toto2TSFM(TSFM):
    """Randomly-initialized Toto-2 exposing the `TSFM` forward contract.

    Toto-2 always carries its nine-knot head, so `n_quantiles` selects only what is
    read out: 9 for the quantile arm, 0 to take the median knot as a point forecast.
    """

    def __init__(self, config, n_quantiles: int = 9):
        super().__init__()
        Toto2Model, _ = _require_toto2()
        self.config = config
        self.patch_length = config.patch_size
        self.n_quantiles = n_quantiles
        self.model = Toto2Model(config)

    def forward(self, y: torch.Tensor, mask: torch.Tensor | None = None):
        x = y.unsqueeze(1)  # (B, 1, T): this study is univariate
        observed = torch.ones_like(x, dtype=torch.bool)
        # Toto-2 marks usable positions; `mask` marks positions to predict.
        usable = observed if mask is None else ~mask.bool().unsqueeze(1)
        sid = torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)
        quantiles, loc, scale = self.model(x, observed, usable, sid)
        # (K, B, C, patches, patch) -> (K, B, T); undo the model's asinh so the knots
        # are linear in the scaler's space.
        z = quantiles.flatten(-2).sinh()[:, :, 0]
        loc, scale = loc[:, 0], scale[:, 0]
        pred = z.permute(1, 2, 0) if self.n_quantiles else z[len(z) // 2]
        return pred, loc, scale


def build(size: str, context_length: int, n_quantiles: int = 9) -> Toto2TSFM:
    return Toto2TSFM(toto2_config(size, context_length), n_quantiles)
