"""Render two pre-defined quantiles of held-out performance over actual Scroll 5 CT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import map_coordinates

from analyze import CONTROL, PRIMARY, read_pairs, summary
from fiber_trace import read_volume

INK = "#172B36"
MUTED = "#486474"
GREEN = "#00877C"
ORANGE = "#BC5D18"
TRACE = "#39F3CA"
FLOW = "#FFAA59"


def choose_examples(pairs):
    # Pick the 95th and 5th percentiles of the paired gain; no visual selection.
    gain = np.array([float(p[CONTROL]["mean_vox"]) - float(p[PRIMARY]["mean_vox"]) for p in pairs])
    return [pairs[int(np.argmin(abs(gain - np.quantile(gain, q))))] for q in (.95, .05)]


def aggregate_figure(report, target):
    cubes = list(report["by_cube"].items())
    fig, ax = plt.subplots(figsize=(8.3, 3.9), constrained_layout=True, facecolor="#FAF9F5")
    ax.set_facecolor("#FAF9F5")
    for index, (name, case) in enumerate(cubes):
        y = len(cubes) - index
        old, new = case["two_way_flow_mean"], case["proposed_mean"]
        ax.plot([new, old], [y, y], lw=2, color="#AABABA", zorder=1)
        ax.scatter(old, y, s=50, color=ORANGE, label="Bidirectional flow" if index == 0 else None, zorder=2)
        ax.scatter(new, y, s=50, color=GREEN, label="Two-click bridge" if index == 0 else None, zorder=2)
    ax.set_yticks(range(len(cubes), 0, -1), [f"z {int(name.split('_')[1]):,}    ·    {v['fibers']} fibers" for name, v in cubes])
    ax.set_xlim(2.18, 3.52)
    ax.set_ylim(.38, len(cubes) + .68)
    ax.set_xlabel("Distance to hidden manual points in voxels  ·  lower is better", color=MUTED)
    ax.set_title("Scroll 5   /   Mean 3D error by region", loc="left", weight="bold", color=INK, pad=15)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="x", lw=.7, color="#D9E0DC")
    ax.tick_params(length=0, colors=MUTED)
    ax.spines[:].set_visible(False)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def case_panel(ax, vol, record):
    true = np.array(record["annotation"])
    flow = np.array(record["paths"][CONTROL])
    new = np.array(record["paths"][PRIMARY])
    start, end = true[[0, -1]]
    length = np.linalg.norm(end - start)
    forward = (end - start) / length
    # Project in a reproducible direction showing the separation of the two estimates.
    diff = new - flow
    lateral = diff - (diff @ forward)[:, None] * forward
    side = lateral[np.argmax(np.linalg.norm(lateral, axis=1))]
    if np.linalg.norm(side) < 1e-8:
        axis = np.eye(3)[np.argmin(abs(forward))]
        side = np.cross(forward, axis)
    side = side / np.linalg.norm(side)
    normal = np.cross(forward, side)

    def flatten(path):
        return (path - start) @ forward, (path - start) @ side

    height = max(13, np.ceil(max(abs(flatten(p)[1]).max() for p in (flow, new, true))) + 4)
    x = np.linspace(-4, length + 4, int((length + 8) * 3))
    y = np.linspace(-height, height, int(6 * height))
    center = start + x[None, :, None] * forward + y[:, None, None] * side
    slabs = []
    for delta in range(-3, 4):
        coords = np.moveaxis((center + delta * normal)[..., ::-1], -1, 0)
        slabs.append(map_coordinates(vol, coords, order=1, mode="nearest", output=np.float32))
    ct = np.max(slabs, axis=0)
    lo, hi = np.quantile(ct, (.025, .985))
    ax.imshow(ct, extent=(x.min(), x.max(), -height, height), origin="lower", cmap="gray",
              vmin=lo, vmax=hi, aspect="equal", interpolation="bilinear")
    ax.plot(*flatten(flow), "--", color=FLOW, lw=2.3, zorder=3)
    ax.plot(*flatten(new), color=TRACE, lw=2.3, zorder=4)
    ax.scatter(*flatten(true[1:-1]), c="white", edgecolors="#061015", lw=1.0, s=45, zorder=5)
    ax.scatter(*flatten(true[[0, -1]]), marker="D", c="#AD8BFF", edgecolors="white", lw=.9, s=44, zorder=6)
    ybar = -height + 2.4
    ax.plot([2, 12], [ybar, ybar], color="white", lw=2.2)
    ax.text(7, ybar + 1.1, "10 vox", ha="center", color="white", fontsize=7,
            bbox=dict(facecolor="#17242A", alpha=.65, pad=.3, edgecolor="none"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/heldout"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("evidence/scroll5_results.png"))
    parser.add_argument("--aggregate-only", type=Path, metavar="AGGREGATE_JSON",
                        help="Draw a chart solely from published aggregate data.")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate_figure(json.loads(args.aggregate_only.read_text(encoding="utf-8")), args.output)
        return
    report = summary(args.input)
    overall = report["overall"]
    examples = choose_examples(read_pairs(args.input))
    chosen = {case[PRIMARY]["key"] for case in examples}
    paths = {rec["key"]: rec for rec in json.loads((args.input / "paths.json").read_text())
             if rec["key"] in chosen}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig = plt.figure(figsize=(12.4, 8.45), facecolor="#FAF9F5")
    fig.text(.055, .946, "Can two clicks recover a papyrus fiber?", size=29, weight="bold")
    fig.text(.056, .9, "Held-out CT from Scroll 5  •  six regions  •  one segment per annotated tree",
             size=13, color=MUTED)
    fig.text(.058, .79, f"{overall['two_way_flow_mean']:.2f}  →  {overall['proposed_mean']:.2f}",
             size=35, weight="bold", color=GREEN)
    fig.text(.058, .75, "mean error in voxels", size=15, weight="bold")
    fig.text(.058, .683, f"{overall['improved']:,} of {overall['fibers']:,} segments improved", size=14)
    fig.text(.058, .639, f"{overall['worsened']:,} got worse", size=14)
    fig.text(.058, .572, f"Scored on {overall['hidden_manual_nodes']:,} hidden human clicks", size=11, color=MUTED)
    chart = fig.add_axes([.60, .57, .34, .251], facecolor="none")
    cubes = list(report["by_cube"].items())
    xs, ys = [], []
    labels = []
    for j, (name, stats) in enumerate(cubes):
        y = len(cubes) - j
        a, b = stats["two_way_flow_mean"], stats["proposed_mean"]
        chart.plot([a, b], [y, y], color="#AEBCBF", lw=2.2, zorder=1)
        chart.scatter([a, b], [y, y], c=[ORANGE, GREEN], s=42, zorder=2)
        xs.extend([a, b])
        ys.append(y)
        labels.append(f"z {int(name.split('_')[1]):,}   ·   n {stats['fibers']}")
    chart.set_xlim(min(xs) - .12, max(xs) + .13)
    chart.set_ylim(.55, len(cubes) + .55)
    chart.set_yticks(ys, labels)
    chart.tick_params(axis="both", length=0, labelsize=9)
    chart.grid(axis="x", color="#DEE3E2", lw=.7)
    chart.spines[:].set_visible(False)
    chart.set_xlabel("Mean 3D error in voxels   ·   lower is better", fontsize=9, labelpad=6)
    fig.text(.592, .843, "Every region improved on average", size=14, weight="bold")
    fig.add_artist(Line2D([.055, .945], [.508, .508], transform=fig.transFigure, color="#CFDAD7", lw=1))
    for col, case in enumerate(examples):
        x0 = .055 if col == 0 else .522
        rec = paths[case[PRIMARY]["key"]]
        title = ("A recovered bend", "A missed bend")[col]
        fig.text(x0, .463, title, size=17, weight="bold")
        a, b = float(case[CONTROL]["mean_vox"]), float(case[PRIMARY]["mean_vox"])
        fig.text(x0, .425, f"Mean 3D error  {a:.2f}  →  {b:.2f} voxels", color=MUTED, size=11)
        vol = read_volume(args.data / (rec["cube"] + "_0000.tif"))
        ax = fig.add_axes([x0, .229, .425, .172], facecolor="none")
        case_panel(ax, vol, rec)
        fig.text(x0, .208, f"Tree {rec['tree']}   ·   cube {rec['cube'][3:]}", size=8.5, color=MUTED)
        del vol
    handles = [Line2D([], [], ls="--", color=ORANGE, lw=2.5, label="Bidirectional flow"),
               Line2D([], [], color=GREEN, lw=2.5, label="Two-click bridge"),
               Line2D([], [], marker="o", lw=0, color="white", mec=INK, label="Hidden human click"),
               Line2D([], [], marker="D", lw=0, color="#AD8BFF", mec=INK, label="Given endpoint")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.5, .155), ncol=4, frameon=False, fontsize=10)
    fig.text(.055, .084, "Examples are the 95th and 5th percentiles of paired improvement. Paths are projected onto a plane along the chord.",
             size=9, color=MUTED)
    fig.text(.055, .054, "Backgrounds show the actual CT maximum across ±3 voxels of depth. Scores use original 3D positions. Source: Vesuvius Challenge fiber skeletons.",
             size=9, color=MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
