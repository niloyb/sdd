#!/usr/bin/env python3
"""Generate and cache a synthetic corpus, optionally a mixture of generators.

  python scripts/make_cache.py --out data/gp --gen gp:1.0 --n 20000 --seq-len 512
  python scripts/make_cache.py --out data/mix --gen gp:0.5 --gen inid:0.3 --gen ou:0.2
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sdd.dataloader import Component, REGISTRY, build_cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--gen", action="append", required=True,
                   help="name:weight, e.g. gp:0.5 (names: " + ", ".join(REGISTRY) + ")")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--shard-size", type=int, default=2000)
    p.add_argument("--noise-std", type=float, default=0.25, help="GP observation noise")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    comps = []
    for spec in a.gen:
        name, _, w = spec.partition(":")
        cls = REGISTRY[name]
        gen = cls(noise_std=a.noise_std) if name == "gp" else cls()
        comps.append(Component(gen, float(w or 1.0)))
    build_cache(a.out, comps, a.n, a.seq_len, a.shard_size, a.seed)
    print(f"wrote {a.n} series to {a.out}")


if __name__ == "__main__":
    main()
