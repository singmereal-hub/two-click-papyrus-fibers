"""Trace between the two endpoints of the first eligible public Scroll 5 tree."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import tasks_for_cube
from fiber_model import Predictor
from fiber_trace import read_volume
from trace_scroll import DEFAULT_MODEL, cube_metadata, trace, write_vc3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("example_fiber.json"))
    args = parser.parse_args()
    stack = args.data / "s5_03997_01497_03997_256_0000.tif"
    example = tasks_for_cube(stack, 1)[0]
    a, b = example.path[[0, -1]]
    torch.set_num_threads(4)
    path, change = trace(read_volume(stack), a, b, Predictor(DEFAULT_MODEL))
    origin, _, _ = cube_metadata(stack)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_vc3d(args.out, path + origin, None)
    print("Local XYZ clicks:", a.tolist(), b.tolist())
    print(f"{args.out}: {len(path)} proposed centerline points. Mean refit movement: {change:.3f} voxels.")


if __name__ == "__main__":
    main()
