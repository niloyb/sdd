"""Training loop: contiguous patch masking or teacher forcing, either arm.

Writes one CSV row per evaluation (`step, tokens, flops, crps`), which is what the
figure scripts read, and checkpoints so a run can resume after preemption. CRPS is
scored on the masked span, matching the objective -- see `evaluate`.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..dataloader import CachedCorpus, collate
from .losses import crps, make_taus, objective
from ..model import TSFM

Tensor = torch.Tensor


@dataclass
class Config:
    cache: str
    out: str
    arch: str = "toto2"  # "toto2" (paper) | "patched" (dependency-free placeholder)
    size: str = "313m"
    arm: str = "sdd"  # "sdd" | "status_quo"
    masking: str = "cpm"  # "cpm" | "tf"
    batch_size: int = 16
    max_steps: int = 10_000
    warmup: int = 0  # defaults to 1% of max_steps
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    eval_every: int = 0  # defaults to max_steps/200
    ckpt_every: int = 2000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_quantiles: int = 9
    patch_length: int = 32
    n_val: int = 512

    def __post_init__(self):
        self.warmup = self.warmup or max(1, round(0.01 * self.max_steps))
        self.eval_every = self.eval_every or max(1, self.max_steps // 200)


def sample_span(n_patches: int, g: torch.Generator, device) -> tuple[int, int]:
    """One contiguous span of patches, never starting at patch 0."""
    hi = max(1, min(16, int(0.4 * n_patches), n_patches - 1))
    length = int(torch.randint(1, hi + 1, (1,), generator=g, device=device))
    start = int(torch.randint(1, n_patches - length + 1, (1,), generator=g, device=device))
    return start, length


def _tail(v: Tensor | float, ctx: int) -> Tensor | float:
    """Align a per-time-point normalizer with targets that start at `ctx`. Scalars and
    per-series statistics broadcast as they are."""
    return v[:, ctx:] if torch.is_tensor(v) and v.dim() > 1 and v.shape[1] > 1 else v


def _lr_scale(step: int, cfg: Config) -> float:
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.max_steps - cfg.warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model: TSFM, loader, taus: Tensor | None, device: str, n_patches: int) -> float:
    """Held-out CRPS on the masked span, in the model's own normalized space.

    Scored the same way the model is trained: a contiguous span is hidden and predicted
    from its prefix. An unmasked next-token pass would be a task the model never trains
    on -- once it has specialized to masked inputs, that number measures the gap between
    the two tasks rather than the quality of the model, and drifts upward late in
    training even though nothing is overfitting. The span generator is reseeded per call
    so every evaluation scores the same spans.

    A point head (`taus is None`) has no quantiles for CRPS to score, so it's scored
    with MSE instead -- logged in the same `crps` column, since it's still "the metric
    this run's objective is evaluated on".
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(0)
    P = model.patch_length
    tot, n = 0.0, 0
    for y, _, _ in loader:
        y = y.to(device)
        s, L = sample_span(n_patches, g, device)
        tgt = slice(s * P, (s + L) * P)
        mask = torch.zeros_like(y, dtype=torch.bool)
        mask[:, tgt] = True
        pred, loc, scale = model(y, mask)
        yn = (y - loc) / scale if torch.is_tensor(scale) else y
        if taus is None:  # point head: no quantiles for crps to score
            score = float((pred[:, tgt] - yn[:, tgt]).pow(2).mean())
        else:
            score = float(crps(pred[:, tgt], yn[:, tgt], taus))
        tot += score * y.shape[0]
        n += y.shape[0]
    model.train()
    return tot / max(1, n)


def train(model: TSFM, cfg: Config) -> Path:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(str(asdict(cfg)))

    torch.manual_seed(cfg.seed)
    dev = cfg.device
    model = model.to(dev)
    taus = make_taus(dev, cfg.n_quantiles) if cfg.n_quantiles else None

    corpus = CachedCorpus(cfg.cache)
    T = corpus.manifest["seq_len"]
    P = cfg.patch_length
    n_patches = T // P
    gens = corpus.generators

    # A single-pass loader: the corpus is large enough that no series repeats.
    train_ds = torch.utils.data.Subset(corpus, range(cfg.n_val, len(corpus)))
    val_ds = torch.utils.data.Subset(corpus, range(cfg.n_val))
    val = DataLoader(
        val_ds, batch_size=128, collate_fn=lambda b: collate(b, T - P, gens)
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    g = torch.Generator(device=dev).manual_seed(cfg.seed)

    start = _resume(model, opt, out)
    log = open(out / "log.csv", "a", newline="")
    w = csv.writer(log)
    if start == 0:
        w.writerow(["step", "tokens", "flops", "crps"])

    n_par = model.num_parameters()
    tokens_per_step = cfg.batch_size * n_patches
    step = start
    while step < cfg.max_steps:
        for batch in DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=lambda b: b,  # posterior needs the span, chosen below
        ):
            if step >= cfg.max_steps:
                break
            for gp in opt.param_groups:
                gp["lr"] = cfg.lr * _lr_scale(step, cfg)

            if cfg.masking == "cpm":
                s, L = sample_span(n_patches, g, dev)
                ctx = s * P
                tgt = slice(ctx, ctx + L * P)
            else:  # teacher forcing scores every next-patch position
                ctx = P
                tgt = slice(P, T)

            y, mu, sd = collate(batch, ctx, gens)
            y, mu, sd = y.to(dev), mu.to(dev), sd.to(dev)
            mask = torch.zeros_like(y, dtype=torch.bool)
            if cfg.masking == "cpm":
                mask[:, tgt] = True

            pred, loc, scale = model(y, mask)
            if torch.is_tensor(scale):  # score in the space the model predicts in
                # loc/scale span the whole series; mu/sd start at ctx.
                lo, sc = _tail(loc, ctx), _tail(scale, ctx)
                y, mu, sd = (y - loc) / scale, (mu - lo) / sc, sd / sc
            off = tgt.start - ctx
            span = slice(off, off + (tgt.stop - tgt.start))
            loss = objective(
                cfg.arm, pred[:, tgt], y[:, tgt], mu[:, span], sd[:, span], taus
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            step += 1

            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                c = evaluate(model, val, taus, dev, n_patches)
                tok = step * tokens_per_step
                w.writerow([step, tok, 6 * n_par * tok, f"{c:.6f}"])
                log.flush()
            if step % cfg.ckpt_every == 0:
                _save(model, opt, step, out)
    _save(model, opt, step, out)
    log.close()
    return out / "log.csv"


def _save(model, opt, step: int, out: Path) -> None:
    torch.save(
        {"step": step, "model": model.state_dict(), "opt": opt.state_dict()},
        out / "ckpt.pt",
    )


def _resume(model, opt, out: Path) -> int:
    p = out / "ckpt.pt"
    if not p.exists():
        return 0
    s = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(s["model"])
    opt.load_state_dict(s["opt"])
    return int(s["step"])
