# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Pick COCO val2017 scenes built around one main object.

A scene is kept when, after a centred 4:3 crop:
  - it has one dominant, non-person object (the "main object"), large and
    near the centre, at least 2.5x larger than any other object;
  - it is not cluttered (few annotated objects, no crowd annotation);
  - it has at least 4 other objects that can be changed: small, fully inside
    the frame, and with a gap to the main object and to each other (checked
    on the outlines, not the bounding boxes).

Both val2017 and train2017 are searched (train2017 only through its
annotations: make_variants.py downloads the few photographs it needs).

Writes build/scenes.json, read by make_variants.py.

--freeview keeps only the photographs of COCO-FreeView (free-viewing eye
tracking, 10 observers per photograph; see tools/compare_eyetracking.py), so
that every scene can be compared with real fixations.

Usage (from the example folder):
  build/venv/Scripts/python tools/select_scenes.py [--freeview]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO

HERE = Path(__file__).resolve().parent.parent
ANN_DIR = HERE / "build/coco/annotations"
SPLITS = ["val2017", "train2017"]
OUT = HERE / "build/scenes.json"

ASPECT = 4 / 3
MAX_OBJECTS = 20           # annotated instances in the crop
MAIN_MIN, MAIN_MAX = 0.08, 0.50   # main object area / crop area
MAIN_DOMINANCE = 1.5       # main area >= this x any other object
MAIN_CENTRE = 0.25         # centroid within this fraction of the centre
SEC_MIN, SEC_MAX = 0.001, 0.06    # changeable object area / crop area
# LaMa only erases small objects cleanly; people leave ghosts sooner. Larger
# objects stay candidates only if they can change colour instead (see
# make_variants.py, which applies the same limits).
ERASE_MAX = {"person": 0.0064}
ERASE_MAX_DEFAULT = 0.015
COLOUR_NEVER = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "potted plant",
    "stop sign", "traffic light",
}
MARGIN = 0.012             # gap to the frame / other changes, fraction of width
MIN_SECONDARY = 4
PER_CATEGORY = 8           # at most this many scenes per main category
MAX_SCENES = 260           # enough to throw the bad ones away after review
# The variants are modified photographs: "NoDerivs" licences (COCO licence
# IDs 3 and 6) forbid sharing them.
NO_DERIVATIVES = {3, 6}
# "person" is not an ImageNet class; the others make a room, not an object.
EXCLUDED_MAIN = {"person", "dining table", "bed", "couch", "oven", "refrigerator", "sink"}


def crop_box(w, h):
    """Centred 4:3 crop (x0, y0, w, h)."""
    if w / h > ASPECT:
        cw = round(h * ASPECT)
        return (w - cw) // 2, 0, cw, h
    ch = round(w / ASPECT)
    return 0, (h - ch) // 2, w, ch


def grown(mask, gap):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap + 1, 2 * gap + 1))
    return cv2.dilate(mask, k)


def plausible(anns, cats, w, h):
    """Cheap test on the annotation areas, before decoding any mask."""
    if not anns or len(anns) > MAX_OBJECTS or any(a["iscrowd"] for a in anns):
        return False
    areas = sorted((a["area"] / (w * h) for a in anns), reverse=True)
    biggest = max(anns, key=lambda a: a["area"])
    if cats[biggest["category_id"]] in EXCLUDED_MAIN:
        return False
    if not 0.6 * MAIN_MIN <= areas[0] <= 1.2 * MAIN_MAX:
        return False
    return sum(0.6 * SEC_MIN <= a <= 1.2 * SEC_MAX for a in areas[1:]) >= MIN_SECONDARY


FREEVIEW = HERE / "build/eyetracking/COCOFreeView_fixations_trainval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeview", action="store_true", help="only COCO-FreeView photographs")
    args = ap.parse_args()
    only = None
    if args.freeview:
        only = {int(r["name"].split(".")[0]) for r in json.load(open(FREEVIEW))}
        print(f"COCO-FreeView: {len(only)} photographs")
        # COCO-FreeView holds the photographs of a visual-search study, mostly
        # kitchens and desks: with the criteria above only 23 pass. Relaxed:
        global MAX_OBJECTS, MAIN_MIN, MAIN_MAX, MAIN_DOMINANCE, MAIN_CENTRE, PER_CATEGORY, EXCLUDED_MAIN
        MAX_OBJECTS, MAIN_DOMINANCE, MAIN_CENTRE = 25, 1.3, 0.30
        MAIN_MIN, MAIN_MAX = 0.06, 0.60
        PER_CATEGORY = 12
        EXCLUDED_MAIN = {"person", "dining table", "bed", "couch"}  # sinks, ovens, fridges allowed
    scenes, per_cat = [], Counter()
    for split in SPLITS:
        coco = COCO(str(ANN_DIR / f"instances_{split}.json"))
        select(coco, split, scenes, per_cat, only)
        if len(scenes) >= MAX_SCENES:
            break

    OUT.write_text(json.dumps(scenes, indent=1))
    print(f"{len(scenes)} scenes -> {OUT}")
    print("main categories:", dict(per_cat.most_common()))


def select(coco, split, scenes, per_cat, only=None):
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    for img_id in sorted(coco.getImgIds()):
        if only is not None and img_id not in only:
            continue
        if len(scenes) >= MAX_SCENES:
            return
        info = coco.loadImgs(img_id)[0]
        w, h = info["width"], info["height"]
        if w < h:  # portrait: the crop would throw most of it away
            continue
        if info["license"] in NO_DERIVATIVES:
            continue
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=None))
        if not plausible(anns, cats, w, h):
            continue
        cx0, cy0, cw, ch = crop_box(w, h)

        objs = []
        for a in anns:
            m = coco.annToMask(a)[cy0:cy0 + ch, cx0:cx0 + cw]
            area = int(m.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(m)
            full = int(coco.annToMask(a).sum())
            objs.append({
                "mask": m,
                "ann_id": a["id"], "category": cats[a["category_id"]],
                "area": area / (cw * ch), "inside": area == full,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "cx": float(xs.mean()) / cw, "cy": float(ys.mean()) / ch,
            })
        if not objs or len(objs) > MAX_OBJECTS:
            continue

        objs.sort(key=lambda o: -o["area"])
        main_obj = objs[0]
        if main_obj["category"] in EXCLUDED_MAIN:
            continue
        if not MAIN_MIN <= main_obj["area"] <= MAIN_MAX:
            continue
        if len(objs) > 1 and main_obj["area"] < MAIN_DOMINANCE * objs[1]["area"]:
            continue
        if abs(main_obj["cx"] - 0.5) > MAIN_CENTRE or abs(main_obj["cy"] - 0.5) > MAIN_CENTRE:
            continue
        if per_cat[main_obj["category"]] >= PER_CATEGORY:
            continue

        gap = max(2, round(MARGIN * cw))
        taken = grown(main_obj["mask"], gap)  # pixels a new change may not touch
        chosen = []
        for o in objs[1:]:
            x0, y0, x1, y1 = o["bbox"]
            if not (SEC_MIN <= o["area"] <= SEC_MAX and o["inside"]):
                continue
            erasable = o["area"] <= ERASE_MAX.get(o["category"], ERASE_MAX_DEFAULT)
            if not erasable and o["category"] in COLOUR_NEVER:
                continue
            if x0 < gap or y0 < gap or x1 > cw - gap or y1 > ch - gap:
                continue
            g = grown(o["mask"], gap)
            if (g & taken).any():
                continue
            chosen.append(o)
            taken |= g
        if len(chosen) < MIN_SECONDARY:
            continue

        per_cat[main_obj["category"]] += 1
        for o in [main_obj] + chosen:
            del o["mask"]
        scenes.append({
            "coco_id": img_id, "split": split, "file": info["file_name"],
            "crop": [cx0, cy0, cw, ch], "main": main_obj, "secondary": chosen,
        })


if __name__ == "__main__":
    main()
