"""Read public WEBKNOSSOS centerlines and choose one two-click task per annotated tree."""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Task:
    cube: str
    tree: str
    path: np.ndarray  # local x,y,z (only the endpoints may be used by the tracer)
    key: str


def cube_annotations(tiff: Path) -> tuple[Path, np.ndarray]:
    scroll, z, y, x, size, _ = tiff.stem.split("_")
    prefix = "s1a" if scroll == "s1" else scroll
    pattern = f"fibers_{prefix}_{z}z_{y}y_{x}x_{size}_*.nml"
    matches = sorted(tiff.parent.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one official annotation for {tiff.name}: {matches}")
    return matches[0], np.array([int(x), int(y), int(z)])


def chains(tree: ET.Element, origin: np.ndarray):
    coords = {int(n.attrib["id"]): np.array([float(n.attrib[k]) for k in "xyz"]) - origin
              for n in tree.findall("./nodes/node")}
    adjacent = {n: [] for n in coords}
    for e in tree.findall("./edges/edge"):
        a, b = int(e.attrib["source"]), int(e.attrib["target"])
        adjacent[a].append(b)
        adjacent[b].append(a)
    seen = set()
    for node in sorted(adjacent):
        if len(adjacent[node]) == 2:
            continue
        for neighbor in sorted(adjacent[node]):
            edge = tuple(sorted((node, neighbor)))
            if edge in seen:
                continue
            ordered = [node]
            a, b = node, neighbor
            while True:
                seen.add(tuple(sorted((a, b))))
                ordered.append(b)
                if len(adjacent[b]) != 2:
                    break
                nxt = adjacent[b][0] if adjacent[b][1] == a else adjacent[b][1]
                if tuple(sorted((b, nxt))) in seen:
                    break
                a, b = b, nxt
            yield np.stack([coords[idx] for idx in ordered])


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tasks_for_cube(tiff: Path, limit: int, per_tree: int = 1) -> list[Task]:
    ann, origin = cube_annotations(tiff)
    cube = re.sub(r"_0000$", "", tiff.stem)
    side = int(cube.split("_")[-1])
    tasks = []
    for tree in ET.parse(ann).getroot().findall("thing"):
        ident = tree.attrib["id"]
        options = []
        for chain_no, line in enumerate(chains(tree, origin)):
            inside = ((line >= 1) & (line <= side - 2)).all(axis=1)
            # Pick a real pair of clicks around 96 voxels apart. Do not filter by straightness,
            # intensity, difficulty, or any method's performance on the hidden interior.
            for i in range(0, len(line) - 4, 3):
                distances = np.linalg.norm(line[i + 4:] - line[i], axis=1)
                if not len(distances):
                    continue
                j = i + 4 + int(np.argmin(np.abs(distances - 96.0)))
                if not (64 <= distances[j - i - 4] <= 128 and inside[i:j+1].all()):
                    continue
                key = stable_key(f"fiber-bridge-v1/{cube}/{ident}/{chain_no}/{i}/{j}")
                options.append((key, line[i:j+1].copy()))
        if options:
            for key, path in sorted(options, key=lambda pair: pair[0])[:per_tree]:
                tasks.append(Task(cube, ident, path, key))
    tasks.sort(key=lambda t: stable_key(f"sample-v1/{t.cube}/{t.tree}"))
    return tasks[:limit] if limit > 0 else tasks
