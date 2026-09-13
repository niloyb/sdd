#!/usr/bin/env python3
"""Render the paper's Figures 1-3 and Table 4 from completed runs.

  python scripts/make_figures.py --runs runs --out figures
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sdd import figures


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="figures")
    p.add_argument("--sizes", default="tiny,small,base,large,xl")
    p.add_argument("--only", default="", help="fig1 | fig2 | fig3 | table4 | run")
    p.add_argument("--run-root", default="", help="for --only run: dir holding <level>/<arm>/seed*/")
    p.add_argument("--run-level", default="level")
    p.add_argument("--run-title", default="")
    a = p.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    sizes = a.sizes.split(",")
    want = lambda k: not a.only or a.only == k

    if want("run") and a.run_root:
        figures.figure_run(Path(a.run_root), out, a.run_level, "fig_run_cpm_crps",
                           a.run_title)
    if want("fig1"):
        figures.figure1(out)
    if want("fig2") and (runs / "size_cpm").exists():
        figures.figure2(runs / "size_cpm", out, sizes, "fig2_flops_cpm_crps")
    if want("fig3") and (runs / "size_tf").exists():
        figures.figure3(runs / "size_tf", out, sizes, "fig3_flops_panels_tf_crps")
    if want("table4"):
        groups = [
            (r"observation noise $\sigma$", runs / "noise", ["0.1", "0.25", "0.5", "1", "5", "10"]),
            (r"batch size $B$", runs / "batch", ["64", "128", "256", "512"]),
            (r"sequence length $T$", runs / "seqlen", ["128", "256", "512", "1024"]),
        ]
        groups = [g for g in groups if Path(g[1]).exists()]
        if groups:
            figures.table4(groups, out, "table4_speedups")
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
