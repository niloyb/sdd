#!/usr/bin/env python3
"""Train one arm of one configuration.

  python scripts/train.py --cache data/gp --out runs/size/small/sdd/seed0 \
      --size small --arm sdd --masking cpm --max-steps 20000
"""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sdd.model import build, sizes
from sdd.train import Config, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="toto2", choices=["toto2", "patched"])
    p.add_argument("--size", default="313m",
                   help="toto2: 4m|22m|313m|1B|2.5B; patched: tiny|small|base|large|xl")
    p.add_argument("--arm", default="sdd", choices=["sdd", "status_quo"])
    p.add_argument("--masking", default="cpm", choices=["cpm", "tf"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-quantiles", type=int, default=9)
    p.add_argument("--patch-length", type=int, default=32)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    if a.size not in sizes(a.arch):
        p.error(f"--size {a.size} not available for --arch {a.arch}: {sizes(a.arch)}")
    cfg = Config(
        cache=a.cache, out=a.out, arch=a.arch, size=a.size, arm=a.arm, masking=a.masking,
        batch_size=a.batch_size, max_steps=a.max_steps, lr=a.lr, seed=a.seed,
        n_quantiles=a.n_quantiles, patch_length=a.patch_length,
        **({"device": a.device} if a.device else {}),
    )
    from sdd.dataloader import CachedCorpus

    ctx = CachedCorpus(a.cache).manifest["seq_len"]
    # Before build(): `train` reseeds too, but by then the weights already exist and
    # came from torch's per-process default generator, leaving a seed's arms unpaired.
    torch.manual_seed(a.seed)
    model = build(a.arch, a.size, ctx, a.patch_length, a.n_quantiles)
    print(f"{a.arch}/{a.size}: {model.num_parameters()/1e6:.1f}M params")
    print("wrote", train(model, cfg))


if __name__ == "__main__":
    main()
