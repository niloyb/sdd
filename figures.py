"""The paper's Figures 1-3 and Table 4.

Figure 1 is drawn from a fresh GP draw. The rest read `log.csv` files written by
`train.py`, one directory per run, named `<group>/<level>/<arm>/seed<k>/log.csv`.
Their `crps` column is scored on the masked span, matching the training objective.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataloader import GaussianProcess

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
C_SQ, C_SDD = "#b0632a", "#2f6fae"
SMOOTH = 25


def _save(fig, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- Figure 1

def figure1(out: Path, seed: int = 3, ctx: int = 96, horizon: int = 32) -> None:
    """Status Quo vs SDD targets on one trajectory."""
    gp = GaussianProcess(noise_std=0.3, kernels=("rbf",))
    g = torch.Generator().manual_seed(seed)
    y, p = gp.sample(1, ctx + horizon, g)
    mu, sd = gp.posterior(y, p, ctx)
    y, mu, sd = y[0].numpy(), mu[0].numpy(), sd[0].numpy()
    t = np.arange(len(y))
    f = t[ctx:]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.9), sharey=True)
        for a, title in zip(ax, ("Status Quo target", "SDD target")):
            a.plot(t[:ctx], y[:ctx], color="0.25", label="history $y_{0:t}$")
            a.axvline(ctx - 0.5, color="0.75", lw=0.6)
            a.set_title(title, fontsize=8)
            a.set_xlabel("time")
        ax[0].plot(f, y[ctx:], color=C_SQ, label="realized future $y_{t+1:t+h}$")
        ax[1].fill_between(
            f, mu - 2 * sd, mu + 2 * sd, color=C_SDD, alpha=0.20, linewidth=0
        )
        ax[1].plot(f, mu, color=C_SDD, label=r"conditional mean $\mathbb{E}[Y|y_{0:t}]$")
        ax[1].plot(f, y[ctx:], color=C_SQ, lw=0.7, alpha=0.55)
        ax[0].set_ylabel("value")
        for a in ax:
            a.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        _save(fig, out, "status_quo_sdd")


# ------------------------------------------------------------- run bookkeeping

def _read(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step, flops, crps = [], [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            step.append(int(r["step"]))
            flops.append(float(r["flops"]))
            crps.append(float(r["crps"]))
    return np.array(step), np.array(flops), np.array(crps)


def _smooth(y: np.ndarray, w: int = SMOOTH) -> np.ndarray:
    """Centred moving average whose window shrinks symmetrically at the two ends.

    Centred, not trailing: a trailing mean lags by half its window, and while that
    window is still filling the lag grows and then abruptly stops, drawing a kink at
    index `w`. Shrinking symmetrically rather than truncating keeps the ends unbiased,
    so the first point is the first measurement rather than an average of what follows.
    """
    if len(y) < 2:
        return y
    h = max(1, min(w, len(y) // 8)) // 2
    n = len(y)
    return np.array(
        [y[i - r : i + r + 1].mean() for i in range(n) for r in (min(h, i, n - 1 - i),)]
    )


def _arm(root: Path, level: str, arm: str):
    """Seed-averaged curve for one (level, arm), over seeds that ran to the same step."""
    runs = [_read(p) for p in sorted((root / level / arm).glob("seed*/log.csv"))]
    runs = [r for r in runs if len(r[0])]
    if not runs:
        return None
    n = min(len(r[0]) for r in runs)
    step, flops = runs[0][0][:n], runs[0][1][:n]
    return step, flops, _smooth(np.mean([r[2][:n] for r in runs], axis=0))


def speedup(root: Path, level: str) -> float | None:
    """Status Quo's full budget over the compute SDD needs to match it.

    Both curves are reduced to a running best first, and the target is Status Quo's
    best rather than its last value. For a monotone run these are the same number;
    for one that degrades late they are not, and comparing raw values would report a
    large speed-up simply because SDD crossed a level Status Quo had already passed
    and given back. `None` means SDD never got there.
    """
    a, b = _arm(root, level, "status_quo"), _arm(root, level, "sdd")
    if a is None or b is None:
        return None
    target = float(np.minimum.accumulate(a[2])[-1])
    hit = np.nonzero(np.minimum.accumulate(b[2]) <= target)[0]
    return float(a[0][-1] / b[0][hit[0]]) if len(hit) else None


# ------------------------------------------------------------------- Figure 2

def figure2(root: Path, out: Path, levels: list[str], stem: str) -> None:
    """CRPS against training compute, one colour per model size, dash per arm."""
    shades = plt.cm.Blues(np.linspace(0.35, 0.95, len(levels)))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        ends: dict[str, list] = {"status_quo": [], "sdd": []}
        for lv, col in zip(levels, shades):
            for arm, style in (("status_quo", "-"), ("sdd", "--")):
                r = _arm(root, lv, arm)
                if r is None:
                    continue
                ax.plot(r[1], r[2], style, color=col, linewidth=1.0)
                ends[arm].append((r[1][-1], r[2][-1]))
        for arm, col, style, lab in (
            ("status_quo", C_SQ, "-", "Status Quo"),
            ("sdd", C_SDD, "--", "SDD"),
        ):
            pts = sorted(ends[arm])
            if pts:
                ax.plot(
                    [x for x, _ in pts], [y for _, y in pts], style, color=col,
                    lw=1.8, marker="o", markersize=3.5, zorder=5, label=lab,
                )
        ax.set_xscale("log")
        ax.set_xlabel("training compute (FLOPs, $6ND$)")
        ax.set_ylabel("held-out CRPS, masked span")
        ax.legend(frameon=False, loc="upper right")
        size_keys = [
            plt.Line2D([], [], color=c, label=lv) for lv, c in zip(levels, shades)
        ]
        ax.add_artist(
            ax.legend(handles=size_keys, frameon=False, fontsize=6.5, loc="lower left",
                      title="Model size", title_fontsize=6.5)
        )
        _save(fig, out, stem)


# --------------------------------------------------- single-model training run

def figure_run(
    root: Path, out: Path, level: str, stem: str, title: str = "", logx: bool = False,
    ylabel: str = "held-out CRPS, masked span",
) -> None:
    """Both arms of one model against training iterations, with the speed-up read off.

    Iterations, not compute: at a single model size the two are proportional, and steps
    are what a reader reproducing the run actually sets. The dotted line is Status Quo's
    best CRPS and the marker is where SDD first reaches it. Linear x by default -- a log
    axis would compress the region where the arms separate into the right-hand edge.

    The lower panel plots the two arms' relative difference. At a few percent the gap
    is far thinner than the range the curves fall over, so it is close to unreadable
    from the main panel alone; this also shows how it varies over training.
    """
    a, b = _arm(root, level, "status_quo"), _arm(root, level, "sdd")
    if a is None or b is None:
        raise SystemExit(f"need both arms under {root / level}")
    with plt.rc_context(RC):
        fig, (ax, axg) = plt.subplots(
            2, 1, figsize=(4.2, 3.1), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
        )
        axg.plot(a[0], 100 * (b[2] / a[2] - 1), color=C_SDD)
        axg.axhline(0, color=C_SQ, lw=0.8)
        axg.set_ylabel("gap (%)", fontsize=7)
        axg.tick_params(labelsize=6)
        ax.plot(a[0], a[2], "-", color=C_SQ, label="Status Quo")
        ax.plot(b[0], b[2], "--", color=C_SDD, label="SDD")
        target = float(np.minimum.accumulate(a[2])[-1])
        ax.axhline(target, color="0.6", lw=0.6, ls=":")
        hit = np.nonzero(np.minimum.accumulate(b[2]) <= target)[0]
        if len(hit):
            x = b[0][hit[0]]
            ax.plot([x], [b[2][hit[0]]], "o", color=C_SDD, ms=4, zorder=5)
            ax.annotate(
                f"{a[0][-1] / b[0][hit[0]]:.2f}$\\times$ speed-up",
                xy=(x, target), xytext=(6, 8), textcoords="offset points",
                fontsize=7, color=C_SDD,
            )
        if logx:
            ax.set_xscale("log")
        axg.set_xlabel("training iterations")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=8)
        ax.legend(frameon=False)
        _save(fig, out, stem)


# ------------------------------------------------------------------- Figure 3

def figure3(root: Path, out: Path, levels: list[str], stem: str) -> None:
    """One panel per size, each on its own scale: for effects too small to share an axis."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(levels), figsize=(5.5, 1.7))
        axes = np.atleast_1d(axes)
        for ax, lv in zip(axes, levels):
            for arm, col, style, lab in (
                ("status_quo", C_SQ, "-", "Status Quo"),
                ("sdd", C_SDD, "--", "SDD"),
            ):
                r = _arm(root, lv, arm)
                if r is None:
                    continue
                half = len(r[1]) // 2  # crop to the converged band
                ax.plot(r[1][half:], r[2][half:], style, color=col, label=lab)
            ax.set_title(lv, fontsize=8)
            ax.tick_params(labelsize=6)
        axes[0].set_ylabel("held-out CRPS, masked span", fontsize=7)
        fig.supxlabel("training compute (FLOPs, $6ND$)", fontsize=8)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=2, loc="upper center", fontsize=7)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        _save(fig, out, stem)


# -------------------------------------------------------------------- Table 4

def table4(groups: list[tuple[str, Path, list[str]]], out: Path, stem: str) -> None:
    """One speed-up table per swept factor; `groups` is (label, root, levels)."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(groups), figsize=(5.5, 1.9))
        axes = np.atleast_1d(axes)
        for ax, (label, root, levels) in zip(axes, groups):
            rows = []
            for lv in levels:
                s = speedup(Path(root), lv)
                if s is not None:
                    rows.append([lv, f"{s:.2f}$\\times$"])
            ax.axis("off")
            if not rows:
                continue
            head = f"${label.split('$')[1]}$" if "$" in label else label.split()[-1]
            ax.set_title(label, fontsize=7.5, pad=3)
            t = ax.table(
                cellText=rows, colLabels=[head, "speed-up"], colWidths=[0.42, 0.58],
                cellLoc="center", colLoc="center", loc="upper center",
                edges="horizontal",
            )
            t.auto_set_font_size(False)
            t.set_fontsize(7)
            t.scale(1.0, 1.15)
            for (r, _), cell in t.get_celld().items():
                cell.set_linewidth(0.6)
                if r == 0:
                    cell.set_text_props(fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.04, wspace=0.45)
        _save(fig, out, stem)
