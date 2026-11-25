"""Summarize held-out traces without pooling all manual clicks into a single average."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY = "refit_blend"
CONTROL = "two_way_flow"


def read_pairs(folder):
    pairs = defaultdict(dict)
    with (folder / "per_trace.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            pairs[row["key"]][row["method"]] = row
    return list(pairs.values())


def measure(pairs):
    def column(method, field="mean_vox"):
        return np.array([float(case[method][field]) for case in pairs])
    scores = column(PRIMARY)
    baseline = column(CONTROL)
    improvement = baseline - scores
    tails = column(PRIMARY, "p90_vox")
    low = column(PRIMARY, "refit_change_vox") <= 0.67
    return {
        "fibers": len(pairs),
        "hidden_manual_nodes": int(column(PRIMARY, "nodes").sum()),
        "line_mean": float(column("line").mean()),
        "two_way_flow_mean": float(baseline.mean()),
        "multiscale_texture_mean": float(column("multiscale_texture").mean()),
        "proposed_mean": float(scores.mean()),
        "two_way_flow_reverse_mean": float(column(CONTROL, "reverse_mean_vox").mean()),
        "proposed_reverse_mean": float(column(PRIMARY, "reverse_mean_vox").mean()),
        "mean_reduction_vox": float(improvement.mean()),
        "relative_reduction_percent": float(100 * improvement.mean() / baseline.mean()),
        "improved": int(np.sum(improvement > 0)),
        "equal": int(np.sum(improvement == 0)),
        "worsened": int(np.sum(improvement < 0)),
        "two_way_flow_p90_over_5": int(np.sum(column(CONTROL, "p90_vox") > 5)),
        "proposed_p90_over_5": int(np.sum(tails > 5)),
        "low_priority": int(low.sum()),
        "low_priority_p90_over_5": int(np.sum(low & (tails > 5))),
        "higher_priority_p90_over_5": int(np.sum(~low & (tails > 5))),
        "second_fit_only_exploratory_mean": float(column("refit").mean()),
    }


def summary(folder):
    pairs = read_pairs(folder)
    groups = defaultdict(list)
    for case in pairs:
        groups[case[PRIMARY]["cube"]].append(case)
    return {"overall": measure(pairs), "by_cube": {k: measure(v) for k, v in sorted(groups.items())}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/heldout"))
    parser.add_argument("--output", type=Path, default=Path("results/scroll5.json"))
    args = parser.parse_args()
    result = summary(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["overall"], indent=2))
