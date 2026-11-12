# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Copy the reviewed variants into stimuli/, which main.go embeds.

Variants listed in tools/rejected.txt are dropped: one entry per line, either
"scene/vK" (one variant) or "scene" (the whole scene); '#' starts a comment.
With --freeview (the current stimuli), only the photographs of COCO-FreeView
are kept (they come with eye tracking), and a scene needs at least MIN_CHANGES
usable changes, main object included: main.go shows 3 at once. Without it, a
scene needs MIN_SECONDARY changes outside its main object (the first version,
one change per trial).

Each variant also gets vK_region.png: its object's mask widened by REGION_PX,
which covers every pixel the change touched (erasures are dilated by 15 px
and feathered by 7 px). main.go pastes these regions of 3 variants onto the
original to show 3 changes at once.

Writes stimuli/<scene>/..., stimuli/manifest.csv (one row per variant) and
stimuli/ATTRIBUTION.csv (licence and source of every COCO photograph).

Usage (from the example folder):
  build/venv/Scripts/python tools/export_stimuli.py [--freeview]
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent.parent
VARIANTS = HERE / "build/variants"
REJECTED = HERE / "tools/rejected.txt"
ANN_DIR = HERE / "build/coco/annotations"
OUT = HERE / "stimuli"
MIN_SECONDARY = 4
MIN_CHANGES = 3
REGION_PX = 25
FREEVIEW = HERE / "build/eyetracking/COCOFreeView_fixations_trainval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeview", action="store_true", help="COCO-FreeView photographs only")
    args = ap.parse_args()
    only = None
    if args.freeview:
        only = {int(r["name"].split(".")[0]) for r in json.load(open(FREEVIEW))}
    rejected = set()
    if REJECTED.exists():
        for line in REJECTED.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                rejected.add(line)

    by_scene = defaultdict(list)
    with open(VARIANTS / "variants.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["scene"] in rejected or f"{r['scene']}/{r['variant']}" in rejected:
                continue
            if only is not None and int(r["coco_id"]) not in only:
                continue
            by_scene[r["scene"]].append(r)
    if only is not None:
        kept = {s: rs for s, rs in by_scene.items() if len(rs) >= MIN_CHANGES}
    else:
        kept = {s: rs for s, rs in by_scene.items()
                if sum(r["main"] == "0" for r in rs) >= MIN_SECONDARY}

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    rows = []
    for sid, rs in sorted(kept.items()):
        (OUT / sid).mkdir()
        shutil.copy(VARIANTS / sid / "original.jpg", OUT / sid)
        # distance of every pixel to each changed object: where two widened
        # regions overlap, a pixel goes to the nearer object
        dist = [cv2.distanceTransform(
            (cv2.imread(str(VARIANTS / sid / f"{r['variant']}_mask.png"), cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8),
            cv2.DIST_L2, 5) for r in rs]
        nearest = np.argmin(np.stack(dist), axis=0)
        for k, r in enumerate(rs):
            for suffix in (".jpg", "_mask.png"):
                shutil.copy(VARIANTS / sid / f"{r['variant']}{suffix}", OUT / sid)
            region = (dist[k] <= REGION_PX) & (nearest == k)
            cv2.imwrite(str(OUT / sid / f"{r['variant']}_region.png"), np.where(region, 0, 255).astype(np.uint8))
            rows.append(r)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    licences, images = {}, {}
    for split in sorted({rs[0]["coco_split"] for rs in kept.values()}):
        coco = json.loads((ANN_DIR / f"instances_{split}.json").read_text())
        licences.update({l["id"]: l for l in coco["licenses"]})
        images.update({i["id"]: i for i in coco["images"]})
    with open(OUT / "ATTRIBUTION.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scene", "coco_file", "flickr_url", "licence", "licence_url"])
        for sid, rs in sorted(kept.items()):
            img = images[int(rs[0]["coco_id"])]
            lic = licences[img["license"]]
            wr.writerow([sid, img["file_name"], img.get("flickr_url", ""), lic["name"], lic["url"]])

    print(f"{len(kept)} scenes, {len(rows)} variants -> {OUT}"
          f" ({len(by_scene) - len(kept)} scenes dropped: too few changes left)")


if __name__ == "__main__":
    main()
