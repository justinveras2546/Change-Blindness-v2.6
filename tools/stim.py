# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Rebuild a full variant image from the stimuli (see make_patches.py).

stimuli/ keeps, for each variant, only the patch where it differs from the
original; variant_bgr() pastes it back within the variant's region, which is
what main.go shows. While the full vK.jpg is still there (right after
export_stimuli.py), it is used instead.
"""

import csv
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent.parent
STIM = HERE / "stimuli"
_manifest = None


def _row(scene, variant):
    global _manifest
    if _manifest is None:
        _manifest = {(r["scene"], r["variant"]): r for r in csv.DictReader(open(STIM / "manifest.csv"))}
    return _manifest[scene, variant]


def variant_bgr(scene, variant):
    """The variant as a 1024x768 BGR image (OpenCV order)."""
    full = STIM / scene / f"{variant}.jpg"
    if full.exists():
        return cv2.imread(str(full))
    r = _row(scene, variant)
    img = cv2.imread(str(STIM / scene / "original.jpg"))
    patch = cv2.imread(str(STIM / scene / f"{variant}_patch.jpg"))
    region = cv2.imread(str(STIM / scene / f"{variant}_region.png"), cv2.IMREAD_GRAYSCALE) < 128
    x0, y0 = int(r["patch_x"]), int(r["patch_y"])
    h, w = patch.shape[:2]
    sub = region[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w][sub] = patch[sub]
    return img


def variant_rgb(scene, variant):
    return cv2.cvtColor(variant_bgr(scene, variant), cv2.COLOR_BGR2RGB)
