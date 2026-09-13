#!/usr/bin/env python3
"""End-to-end check: every generator's posterior is calibrated, each distilled loss
matches a Monte-Carlo estimate of its realized counterpart, initialization is
seed-paired, and a short run trains, checkpoints and resumes. Run it directly."""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sdd.dataloader import CachedCorpus, Component, REGISTRY, build_cache, collate
from sdd.train.losses import (
    make_taus, mse_distilled, mse_realized, pinball_distilled, pinball_realized,
)
from sdd.model import build
from sdd.train import Config, train


def check_posteriors() -> None:
    """z-scores of the realized future under the reported conditional law must be
    standard normal; a wrong posterior shows up immediately as std != 1."""
    for name, cls in REGISTRY.items():
        g = torch.Generator().manual_seed(0)
        gen = cls()
        y, p = gen.sample(256, 64, g)
        mu, sd = gen.posterior(y, p, 48)
        z = ((y[:, 48:] - mu) / sd).flatten()
        assert abs(float(z.mean())) < 0.15, f"{name}: z mean {float(z.mean()):.3f}"
        assert abs(float(z.std()) - 1) < 0.15, f"{name}: z std {float(z.std()):.3f}"
        print(f"  {name:5s} posterior ok  (z mean {float(z.mean()):+.3f}, std {float(z.std()):.3f})")


def check_distilled_equals_expected_realized() -> None:
    """The distilled loss is the conditional expectation of the realized loss, so
    averaging the realized loss over many draws of the future must converge to it."""
    torch.manual_seed(0)
    mu, sd = torch.randn(64), torch.rand(64) + 0.5
    pred = torch.randn(64)
    draws = mu + sd * torch.randn(20000, 64)

    mc = torch.stack([mse_realized(pred, d) for d in draws]).mean()
    assert torch.allclose(mc, mse_distilled(pred, mu, sd), rtol=0.02), "MSE mismatch"
    print(f"  mse      distilled {float(mse_distilled(pred, mu, sd)):.4f}  MC {float(mc):.4f}")

    taus = make_taus("cpu")
    q = mu.unsqueeze(-1) + sd.unsqueeze(-1) * torch.randn(64, 9).sort(-1).values
    mc = torch.stack([pinball_realized(q, d, taus) for d in draws]).mean()
    dist = pinball_distilled(q, mu, sd, taus)
    assert torch.allclose(mc, dist, rtol=0.02), f"pinball {float(mc)} vs {float(dist)}"
    print(f"  pinball  distilled {float(dist):.4f}  MC {float(mc):.4f}")


def check_train_and_resume(tmp: Path) -> None:
    """Build a mixture cache, train a few steps, then restart and continue from it."""
    cache = tmp / "cache"
    build_cache(cache, [Component(REGISTRY["gp"](), 0.7), Component(REGISTRY["inid"](), 0.3)],
                n_series=192, seq_len=128, shard_size=96, seed=0)
    corpus = CachedCorpus(cache)
    assert len(corpus) == 192
    y, mu, sd = collate([corpus[i] for i in range(8)], 96, corpus.generators)
    assert y.shape == (8, 128) and mu.shape == (8, 32) and sd.shape == (8, 32)
    print(f"  cache ok  ({len(corpus)} series, mixture of {len(corpus.generators)})")

    out = tmp / "run"
    cfg = Config(cache=str(cache), out=str(out), arch="patched", size="tiny", arm="sdd",
                 batch_size=8, max_steps=6, ckpt_every=3, n_val=64, device="cpu",
                 patch_length=32)
    model = build("patched", "tiny", 128, 32, 9)
    train(model, cfg)
    rows = (out / "log.csv").read_text().strip().split("\n")
    assert len(rows) >= 2, "no eval rows logged"
    assert (out / "ckpt.pt").exists()
    print(f"  train ok  ({len(rows)-1} eval rows, checkpoint written)")

    cfg2 = Config(cache=str(cache), out=str(out), arch="patched", size="tiny", arm="sdd",
                  batch_size=8, max_steps=9, ckpt_every=3, n_val=64, device="cpu",
                  patch_length=32)
    train(build("patched", "tiny", 128, 32, 9), cfg2)
    step = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)["step"]
    assert step == 9, f"resume ended at {step}"
    print(f"  resume ok (continued to step {step})")


def check_toto2(tmp: Path) -> None:
    """The paper's architecture, if `toto-ts` is installed. Skipped otherwise so the
    rest of the suite stays dependency-free."""
    try:
        from sdd.model.toto2 import TOTO2_SIZES, build as build_toto2
    except ImportError:
        print("  toto-ts not installed, skipped")
        return
    try:
        m = build_toto2("4m", 128, 9)
    except ImportError as e:
        print(f"  toto-ts not installed, skipped ({e.args[0][:40]}...)")
        return
    y = torch.randn(2, 128)
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, 64:96] = True
    pred, loc, scale = m(y, mask)
    assert pred.shape == (2, 128, 9), pred.shape
    assert torch.is_tensor(scale), "Toto-2 must report its normalization"
    print(f"  toto2/4m ok  ({m.num_parameters()/1e6:.1f}M params, sizes {list(TOTO2_SIZES)})")

    # Toto-2 normalizes internally and reports loc/scale, so training it exercises the
    # branch that moves targets into the model's space -- untouched by the placeholder.
    cache = tmp / "cache"
    out = tmp / "run_toto2"
    cfg = Config(cache=str(cache), out=str(out), arch="toto2", size="4m", arm="sdd",
                 batch_size=4, max_steps=2, eval_every=2, ckpt_every=2, n_val=32,
                 device="cpu", patch_length=32)
    train(build_toto2("4m", 128, 9), cfg)
    rows = (out / "log.csv").read_text().strip().split("\n")
    assert len(rows) >= 2, "toto2 run logged no evals"
    crps_val = float(rows[1].split(",")[-1])
    assert crps_val == crps_val and crps_val > 0, f"bad CRPS {crps_val}"
    print(f"  toto2 train ok (normalized-space CRPS {crps_val:.4f})")


def check_paired_init() -> None:
    """Same seed before `build` => same weights, so a seed's two arms are paired.

    Unseeded, initialization falls through to torch's per-process default generator:
    the arms then differ by init as well as objective, and a bad draw in one reads as
    an effect of the other.
    """
    from sdd.model import build

    def weights(seed: int):
        torch.manual_seed(seed)
        m = build("patched", "tiny", 128, 32, 9)
        # every parameter, not just the first: that one is a zero-filled positional
        # embedding, identical under any seed
        return torch.cat([p.detach().flatten() for p in m.parameters()])

    assert torch.equal(weights(0), weights(0)), "same seed, different init"
    assert not torch.equal(weights(0), weights(1)), "seed misses init"
    print("  init is seed-paired ok")


def check_figure1(tmp: Path) -> None:
    """The one figure that needs no completed runs."""
    from sdd import figures
    figures.figure1(tmp / "fig")
    assert (tmp / "fig" / "status_quo_sdd.pdf").exists()
    print("  figure 1 ok")


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        print("posteriors:");  check_posteriors()
        print("losses:");      check_distilled_equals_expected_realized()
        print("pipeline:");    check_train_and_resume(tmp)
        print("seeding:");     check_paired_init()
        print("toto2:");       check_toto2(tmp)
        print("figures:");     check_figure1(tmp)
        print("\nALL CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
