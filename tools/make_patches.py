# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Replace each full variant image by the patch where it differs.

A variant differs from the original only inside its region (vK_region.png),
so storing the whole 1024x768 photograph again for every variant is wasted:
48 MB of the 61 MB of stimuli. This keeps, for each variant, the bounding box
of its region cut from the variant (vK_patch.jpg, JPEG quality 95) and adds
its position (patch_x, patch_y) to stimuli/manifest.csv; main.go pastes the
patch, within the region, onto the original. The full vK.jpg files are then
deleted. This is what makes the browser build small enough to download.

Run once, after export_stimuli.py (run it again after a new export).

Usage (from the repository root):
  python tools/make_patches.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
STIM = HERE / "stimuli"


def main():
    rows = list(csv.DictReader(open(STIM / "manifest.csv", newline="")))
    fields = list(rows[0])
    for extra in ("patch_x", "patch_y"):
        if extra not in fields:
            fields.append(extra)
    before = after = 0
    for r in rows:
        d = STIM / r["scene"]
        full = d / f"{r['variant']}.jpg"
        patch = d / f"{r['variant']}_patch.jpg"
        if not full.exists():
            if not patch.exists():
                raise SystemExit(f"{r['scene']}/{r['variant']}: neither {full.name} nor {patch.name}")
            continue  # already done
        region = cv2.imread(str(d / f"{r['variant']}_region.png"), cv2.IMREAD_GRAYSCALE) < 128
        ys, xs = np.nonzero(region)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        img = cv2.imread(str(full))
        cv2.imwrite(str(patch), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        r["patch_x"], r["patch_y"] = int(x0), int(y0)
        before += full.stat().st_size
        after += patch.stat().st_size
        full.unlink()
    with open(STIM / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if before:
        print(f"{len(rows)} variants: {before / 1e6:.1f} MB of full images -> {after / 1e6:.1f} MB of patches")


if __name__ == "__main__":
    main()
