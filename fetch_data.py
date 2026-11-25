"""Fetch the published EduceLab-Scrolls / Vesuvius fiber volumes and manual NML traces."""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import shutil
import urllib.request
from pathlib import Path

ROOT = "https://dl.ash2txt.org/datasets/fiber-skeletons/Dataset001_sk-fibers-20250124/"
SOURCE_TERMS = "https://dl.ash2txt.org/LICENSE.txt"
HEADERS = {"User-Agent": "Vesuvius fiber tracing research / public data"}
DATA = Path(__file__).resolve().parent / "data"


def listing(directory: str) -> list[str]:
    with urllib.request.urlopen(urllib.request.Request(ROOT + directory + "/", headers=HEADERS), timeout=40) as r:
        return re.findall(r'href="([^"/]+)"', r.read().decode("utf-8"))


def download(task: tuple[str, Path]) -> str:
    url, path = task
    if path.is_file() and path.stat().st_size:
        return path.name
    tmp = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=90) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
                size = response.headers.get("Content-Length")
            if size and tmp.stat().st_size != int(size):
                raise OSError("Incomplete download")
            tmp.replace(path)
            print(f"Downloaded {path.name} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
            return path.name
        except OSError:
            if attempt == 2:
                raise
    raise RuntimeError("Download failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["s1", "s5", "all"], default="all")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    jobs = []
    for folder, suffix in (("imagesTr", ".tif"), ("nml", ".nml")):
        for name in listing(folder):
            if not name.endswith(suffix):
                continue
            if args.split != "all" and not re.search(r"(?:^|_)" + args.split + r"a?_", name):
                continue
            jobs.append((ROOT + folder + "/" + name, DATA / name))
    if not jobs:
        raise RuntimeError("No official files matched")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, jobs))


if __name__ == "__main__":
    main()
