"""Trace one papyrus fiber between two clicks in a grayscale TIFF stack."""
from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from fiber_model import Predictor
from fiber_trace import estimate, read_volume, write_nml

DEFAULT_MODEL = Path(__file__).resolve().parent / "weights" / "fiber_long_s1.pt"
PUBLIC_CUBE = re.compile(r"(s1|s5)_(\d+)_(\d+)_(\d+)_\d+_0000")


def trace(volume, start_xyz, end_xyz, predictor):
    """Return dense local XYZ points and the mean movement between the two fits."""
    routes, diagnostics = estimate(volume, start_xyz, end_xyz, strengths=(1.0,),
                                  predictor=predictor, neural_strengths=(0.35,), refit=True)
    return routes["refit_blend"], diagnostics["refit_change_vox"]


def cube_metadata(path):
    match = PUBLIC_CUBE.fullmatch(path.stem)
    if not match:
        return np.zeros(3), None, None
    scroll, z, y, x = match.groups()
    experiment = {"s1": "scroll1a", "s5": "scroll5-full"}[scroll]
    return np.array([int(x), int(y), int(z)]), experiment, 7.91


def write_vc3d(path, points, volume_id):
    # Version 4 supports externally estimated geometry without native optimizer provenance.
    xyz = np.asarray(points, dtype=float).tolist()
    obj = {
        "type": "vc3d_fiber",
        "version": 4,
        "uuid": str(uuid.uuid4()),
        "generation": 1,
        "line_points": xyz,
        "control_points": [{"position": xyz[0]}, {"position": xyz[-1]}],
        "tags": [],
    }
    if volume_id:
        obj["coordinates"] = {"volume_uuid": volume_id, "scale_factor": 1}
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path, help="Single-channel multipage TIFF, one page per Z slice.")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--end", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    parser.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Global position of the first voxel. Inferred for the published fiber TIFFs, otherwise zero.")
    parser.add_argument("--global-clicks", action="store_true", help="Interpret the two clicks in global volume coordinates.")
    parser.add_argument("--volume-id", help="Source volume ID for VC3D coordinates, for input cropped at level zero.")
    parser.add_argument("--experiment", help="WEBKNOSSOS dataset name for NML. Inferred for the public fiber cubes.")
    parser.add_argument("--voxel-size-um", type=float, help="Isotropic voxel size for NML. Inferred for the public fiber cubes.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Default: CUDA if available, otherwise CPU.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, required=True, help="Write a VC3D .json or WEBKNOSSOS .nml.")
    args = parser.parse_args()
    origin, known_experiment, known_size = cube_metadata(args.volume)
    if args.origin is not None:
        origin = np.asarray(args.origin)
    if not np.isfinite(origin).all():
        parser.error("The volume origin must be finite.")
    suffix = args.out.suffix.lower()
    if suffix not in (".json", ".nml"):
        parser.error("Use a .json or .nml output filename.")
    experiment = args.experiment or known_experiment
    size = args.voxel_size_um if args.voxel_size_um is not None else known_size
    if suffix == ".nml" and (not experiment or size is None or not np.isfinite(size) or size <= 0):
        parser.error("Set --experiment and a positive --voxel-size-um for this stack.")
    if args.out.resolve() == args.volume.resolve():
        parser.error("Choose an output path different from the input volume.")
    shift = origin if args.global_clicks else np.zeros(3)
    start, end = np.asarray(args.start) - shift, np.asarray(args.end) - shift
    torch.set_num_threads(4)
    predictor = Predictor(args.model, args.device)
    volume = read_volume(args.volume)
    before = time.perf_counter()
    path, change = trace(volume, start, end, predictor)
    seconds = time.perf_counter() - before
    points = path.astype(float) + origin
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".nml":
        write_nml(args.out, points, experiment, size)
    else:
        write_vc3d(args.out, points, args.volume_id)
    print(json.dumps({"output": str(args.out), "points": len(points), "origin_xyz": origin.tolist(),
                      "fit_change_vox": round(change, 4), "review_priority": "higher" if change > 0.67 else "lower",
                      "seconds_for_tracing": round(seconds, 3), "device": str(predictor.device)}))


if __name__ == "__main__":
    main()
