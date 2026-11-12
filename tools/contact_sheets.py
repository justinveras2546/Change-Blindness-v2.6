# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Close-ups of every change, original | variant, for reviewing the variants.

Each tile is labelled "<n> <scene>/<variant> <object> [M] <change>" (M = main
object). Bad variants go into tools/rejected.txt as "<scene>/<variant>".

Writes build/sheets/sheet_NN.jpg (36 tiles each).

Usage (from the example folder):
  build/venv/Scripts/python tools/contact_sheets.py [--skip SCENE_LIST]

--skip leaves out the scenes listed in a file (one ID per line), e.g. those
already reviewed in an earlier run.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VARIANTS = HERE / "build/variants"
OUT = HERE / "build/sheets"

TILE, COLS, PER_SHEET = 150, 6, 36


def close_up(img, r):
    x0, y0, x1, y1 = (int(r[k]) for k in ("x0", "y0", "x1", "y1"))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = max(max(x1 - x0, y1 - y0) // 2 + 40, 50)
    h, w = img.shape[:2]
    crop = img[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]
    return cv2.resize(crop, (TILE, TILE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=Path, help="file of scene IDs to leave out")
    args = ap.parse_args()
    skip = set(args.skip.read_text().split()) if args.skip else set()
    rows = [r for r in csv.DictReader(open(VARIANTS / "variants.csv")) if r["scene"] not in skip]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("sheet_*.jpg"):
        old.unlink()
    for start in range(0, len(rows), PER_SHEET):
        tiles = []
        for i, r in enumerate(rows[start:start + PER_SHEET], start=start):
            orig = cv2.imread(str(VARIANTS / r["scene"] / "original.jpg"))
            var = cv2.imread(str(VARIANTS / r["scene"] / f"{r['variant']}.jpg"))
            t = np.hstack([close_up(orig, r), np.full((TILE, 4, 3), 255, np.uint8), close_up(var, r)])
            t = np.vstack([np.zeros((22, t.shape[1], 3), np.uint8), t])
            label = (f"{i} {r['scene']}/{r['variant']} {r['object'][:10]} "
                     f"{'M ' if r['main'] == '1' else ''}{r['change'][:4]}")
            cv2.putText(t, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(np.pad(t, ((3, 3), (3, 3), (0, 0)), constant_values=60))
        while len(tiles) % COLS:
            tiles.append(np.full_like(tiles[0], 60))
        grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
        cv2.imwrite(str(OUT / f"sheet_{start // PER_SHEET:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(rows)} close-ups -> {OUT}")


if __name__ == "__main__":
    main()
