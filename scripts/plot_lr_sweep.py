#!/usr/bin/env python3
"""Ad hoc plot for a learning-rate sweep: CRPS vs. training iteration, one line per LR.

  python scripts/plot_lr_sweep.py --root runs/lr_sweep_tiny --out assets --stem fig_cpu_run_patched_tiny_lr_sweep
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
}
SMOOTH = 25


def _read(path: Path):
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    step = np.array([int(r[0]) for r in rows])
    crps = np.array([float(r[3]) for r in rows])
    return step, crps


def _smooth(y: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    if len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dir holding lr_<lr>/log.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--stem", default="fig_lr_sweep_crps")
    p.add_argument("--title", default="")
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)

    runs = sorted(root.glob("lr_*/log.csv"), key=lambda p: float(p.parent.name.removeprefix("lr_")))
    lrs = [p.parent.name.removeprefix("lr_") for p in runs]
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(runs)))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.2, 2.6))
        for path, lr, col in zip(runs, lrs, shades):
            step, crps = _read(path)
            ax.plot(step, _smooth(crps), color=col, label=f"lr={lr}")
        ax.set_yscale("log")
        ax.set_xlabel("training iterations")
        ax.set_ylabel("held-out CRPS, masked span")
        if a.title:
            ax.set_title(a.title, fontsize=8)
        ax.legend(frameon=False)
        out.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{a.stem}.{ext}", bbox_inches="tight")
        plt.close(fig)
    print("wrote", out / f"{a.stem}.png")


if __name__ == "__main__":
    main()
