"""Caching and loading synthetic corpora, including mixtures of generators.

A shard stores the series and the per-series generator parameters, not the
conditional moments: the moments are O(T^2) to store for arbitrary context
lengths but cheap to recompute from the parameters at batch time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .generators import Generator, REGISTRY

Tensor = torch.Tensor

# Generators have different parameter counts, so a mixture's shards are padded to a
# common width; each generator reads only the leading columns it wrote.
PARAM_WIDTH = 16


@dataclass
class Component:
    generator: Generator
    weight: float = 1.0


def build_cache(
    path: str | Path,
    components: list[Component],
    n_series: int,
    seq_len: int,
    shard_size: int = 2000,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    """Draw `n_series` across the mixture and write shards plus a manifest."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    total_w = sum(c.weight for c in components)
    quotas = [round(n_series * c.weight / total_w) for c in components]
    quotas[-1] += n_series - sum(quotas)

    g = torch.Generator(device=device).manual_seed(seed)
    shard, written, sid = [], 0, 0
    for comp, quota in zip(components, quotas):
        left = quota
        while left > 0:
            take = min(shard_size - sum(y.shape[0] for y, _, _ in shard), left)
            y, p = comp.generator.sample(take, seq_len, g)
            shard.append((y.cpu(), p.cpu(), comp.generator.name))
            left -= take
            if sum(y.shape[0] for y, _, _ in shard) >= shard_size:
                _flush(path, shard, sid)
                written += sum(y.shape[0] for y, _, _ in shard)
                shard, sid = [], sid + 1
    if shard:
        _flush(path, shard, sid)
        written += sum(y.shape[0] for y, _, _ in shard)
        sid += 1

    (path / "manifest.json").write_text(
        json.dumps(
            {
                "num_series": written,
                "num_shards": sid,
                "seq_len": seq_len,
                "seed": seed,
                "components": [
                    {"generator": c.generator.name, "weight": c.weight, "quota": q}
                    for c, q in zip(components, quotas)
                ],
            },
            indent=2,
        )
    )


def _pad(p: Tensor) -> Tensor:
    if p.shape[1] > PARAM_WIDTH:
        raise ValueError(f"generator needs {p.shape[1]} params > PARAM_WIDTH")
    return torch.nn.functional.pad(p, (0, PARAM_WIDTH - p.shape[1]))


def _flush(path: Path, shard: list, sid: int) -> None:
    torch.save(
        {
            "y": torch.cat([y for y, _, _ in shard]),
            "params": torch.cat([_pad(p) for _, p, _ in shard]),
            "gen": sum(([n] * y.shape[0] for y, _, n in shard), []),
        },
        path / f"shard_{sid:05d}.pt",
    )


class CachedCorpus(torch.utils.data.Dataset):
    """Reads shards lazily. Items carry the generator name so the collate function
    can ask the right generator for the conditional moments."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        self.shards = sorted(self.path.glob("shard_*.pt"))
        self._cache: tuple[int, dict] | None = None
        head = torch.load(self.shards[0], map_location="cpu", weights_only=True)
        self.per_shard = head["y"].shape[0]
        self.generators = {
            name: REGISTRY[name]() for name in {c["generator"] for c in self.manifest["components"]}
        }

    def __len__(self) -> int:
        return self.manifest["num_series"]

    def _shard(self, i: int) -> dict:
        if self._cache is None or self._cache[0] != i:
            self._cache = (i, torch.load(self.shards[i], map_location="cpu", weights_only=True))
        return self._cache[1]

    def __getitem__(self, idx: int):
        s = self._shard(min(idx // self.per_shard, len(self.shards) - 1))
        j = idx % self.per_shard
        j = min(j, s["y"].shape[0] - 1)
        return s["y"][j], s["params"][j], s["gen"][j]


def collate(batch, ctx: int, generators: dict[str, Generator]):
    """Stack a batch and attach the conditional moments of `y[:, ctx:]`.

    Series from different generators are grouped so each posterior is computed once
    per generator rather than once per series.
    """
    ys = torch.stack([b[0] for b in batch])
    names = [b[2] for b in batch]
    mu = torch.empty(ys.shape[0], ys.shape[1] - ctx)
    sd = torch.empty_like(mu)
    for name in set(names):
        sel = [i for i, n in enumerate(names) if n == name]
        params = torch.stack([batch[i][1] for i in sel])
        m, s = generators[name].posterior(ys[sel], params, ctx)
        mu[sel], sd[sel] = m, s
    return ys, mu, sd
