# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Compare real eye fixations (COCO-FreeView) with the models' maps and with
the participants' change-blindness data, on the photographs both share.

COCO-FreeView (Chen et al., 2022; Yang et al., 2023): 10 observers freely
view each photograph for 5 s. Download the fixations (about 50 MB) to
build/eyetracking/:

  http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json

The photographs were shown on a 1680x1050 screen, scaled with their aspect
ratio kept and centred (a 4:3 photograph has grey bands on the sides: almost
no fixation falls within 140 px of the screen edges). Fixations are mapped
back to the original photograph, then into our 4:3 crop at 1024x768. The
first fixation of each scanpath (on the central fixation point) is dropped.

Per photograph, the fixations make a density map (Gaussian, sigma 25 px). It
is compared with each model map (Grad-CAM for the predicted class, from
saliency/maps, and for the main object, from
../Saliency-Guided-Occlusion/saliency/maps) and with a central Gaussian
(centre bias), by:
  CC   correlation between the two maps
  NSS  mean of the z-scored model map at the fixated points (0 = chance)

Per changed object (variants of the photograph), the share of fixations that
land on it (mask dilated by 15 px) is compared, across objects, with the
models' saliency of the object (ratio) and with feat_dist.

Writes build/comparison/eyetracking.png, eyetracking_maps.jpg and
eyetracking.txt.

Usage (from Saliency-Change-Blindness):
  build/venv/Scripts/python tools/compare_eyetracking.py [--data DIR]
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OCC = HERE.parent / "Saliency-Guided-Occlusion"
FIX = HERE / "build/eyetracking/COCOFreeView_fixations_trainval.json"
OUT = HERE / "build/comparison"
W, H = 1024, 768
SCREEN_W, SCREEN_H = 1680, 1050
SIGMA = 25
DILATE = 15
MODELS = ["resnet50", "vit_b16"]


def norm(m):
    m = m.astype(np.float64)
    return (m - m.mean()) / (m.std() + 1e-12)


def cc(a, b):
    return float(np.mean(norm(a) * norm(b)))


def nss(sal, pts):
    z = norm(sal)
    return float(np.mean([z[int(y), int(x)] for x, y in pts]))


def overlay(img, sal):
    sal = sal / (sal.max() + 1e-12)
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.45, heat, 0.55, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "goxpy_data")
    args = ap.parse_args()
    if not FIX.exists():
        raise SystemExit(f"missing {FIX}: download it first (see the docstring)")
    OUT.mkdir(parents=True, exist_ok=True)

    fix = json.load(open(FIX))
    by_image = {}
    for r in fix:
        by_image.setdefault(int(r["name"].split(".")[0]), []).append(r)
    scenes = {s["coco_id"]: s for s in json.load(open(HERE / "build/scenes.json"))}
    ours = sorted(int(p.name[1:]) for p in (HERE / "stimuli").iterdir() if p.is_dir())
    shared = [i for i in ours if i in by_image]
    lines = [f"COCO-FreeView: {len(by_image)} photographs; {len(shared)} of our {len(ours)} are among them: "
             + ", ".join(f"s{i}" for i in shared), ""]

    model_rows = {(r["scene"], r["variant"], r["model"]): r
                  for r in csv.DictReader(open(HERE / "saliency/model_saliency.csv"))}
    manifest = list(csv.DictReader(open(HERE / "stimuli/manifest.csv")))
    # participants' data: current design only (3 changes per trial, "rank" column)
    mine = defaultdict(list)      # (scene, variant) -> rows, all participants
    by_trial = defaultdict(list)  # (subject, trial) -> its 3 rows
    for f in sorted(args.data.glob("*Change-Blindness*_sub-*.csv")):
        if "rank" not in open(f, encoding="utf-8").readline():
            continue
        for t in csv.DictReader(open(f, encoding="utf-8")):
            if t["phase"] == "main":
                mine[t["scene"], t["variant"]].append(t)
                by_trial[f.name, t["trial"]].append(t)

    yy, xx = np.mgrid[0:H, 0:W]
    centre = np.exp(-(((xx - W / 2) / (W / 4)) ** 2 + ((yy - H / 2) / (H / 4)) ** 2) / 2)

    cocos, stats, objects, tiles = {}, [], [], []
    for i in shared:
        sc = scenes[i]
        sid = f"s{i}"
        c = cocos.setdefault(sc["split"], COCO(str(HERE / f"build/coco/annotations/instances_{sc['split']}.json")))
        info = c.loadImgs(i)[0]
        w, h = info["width"], info["height"]
        s = min(SCREEN_W / w, SCREEN_H / h)
        ox, oy = (SCREEN_W - w * s) / 2, (SCREEN_H - h * s) / 2
        x0, y0, cw, ch = sc["crop"]
        pts = []
        for r in by_image[i]:
            for x, y in list(zip(r["X"], r["Y"]))[1:]:
                px = ((x - ox) / s - x0) * W / cw
                py = ((y - oy) / s - y0) * H / ch
                if 0 <= px < W and 0 <= py < H:
                    pts.append((px, py))
        dens = np.zeros((H, W))
        for x, y in pts:
            dens[int(y), int(x)] += 1
        dens = cv2.GaussianBlur(dens, (0, 0), SIGMA)

        img = cv2.imread(str(HERE / "stimuli" / sid / "original.jpg"))
        maps = {}
        for m in MODELS:
            maps[f"{m} (classe prédite)"] = cv2.imread(str(HERE / "saliency/maps" / m / f"{sid}.png"), 0) / 255.0
            p = OCC / "saliency/maps" / m / f"{sid}.png"
            if p.exists():
                maps[f"{m} (objet principal)"] = cv2.imread(str(p), 0) / 255.0
        maps["biais central"] = centre
        row = {"scene": sid, "n_fix": len(pts), "observers": len(by_image[i])}
        for k, mp in maps.items():
            row[k] = (cc(dens, mp), nss(mp, pts))
        stats.append(row)

        # objects that change in the change-blindness experiment
        for v in (r for r in manifest if r["scene"] == sid):
            mask = cv2.imread(str(HERE / "stimuli" / sid / f"{v['variant']}_mask.png"), 0) < 128
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * DILATE + 1,) * 2, np.uint8)) > 0
            share = np.mean([mask[int(y), int(x)] for x, y in pts])
            o = {"scene": sid, "variant": v["variant"], "object": v["object"], "main": v["main"],
                 "fix_share": share, "area": float(v["area_px"])}
            for m in MODELS:
                mr = model_rows[sid, v["variant"], m]
                o[f"{m}_ratio"], o[f"{m}_feat_dist"] = float(mr["ratio"]), float(mr["feat_dist"])
            o["times"] = [float(t["rt_ms"]) for t in mine.get((sid, v["variant"]), [])]
            objects.append(o)

        # figure row: fixations | ResNet | ViT | (your click)
        p0 = overlay(img, dens)
        for x, y in pts:
            cv2.circle(p0, (int(x), int(y)), 5, (255, 255, 255), -1)
        panels = [p0] + [overlay(img, maps[f"{m} (classe prédite)"]) for m in MODELS]
        for (sc_id, var), ts in mine.items():
            for t in ts:
                if sc_id == sid and t["found"] == "true":
                    for p in panels:
                        c = (int(float(t["click_x"])), int(float(t["click_y"])))
                        cv2.circle(p, c, 14, (0, 200, 0), -1)
                        cv2.circle(p, c, 14, (255, 255, 255), 3)
        for p, lab in zip(panels, [f"{sid} - regard humain (10 pers.)", "ResNet-50", "ViT-B/16"]):
            cv2.putText(p, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8)
            cv2.putText(p, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        tiles.append(cv2.resize(np.hstack(panels), (1200, 300)))

    cv2.imwrite(str(OUT / "eyetracking_maps.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])

    keys = [k for k in stats[0] if k not in ("scene", "n_fix", "observers")]
    lines.append("Maps vs fixation density, per photograph (CC; NSS):")
    for r in stats:
        lines.append(f"  {r['scene']} ({r['n_fix']} fixations, {r['observers']} observers)")
        for k in keys:
            if k in r:
                lines.append(f"    {k:28s} CC={r[k][0]:+.2f}  NSS={r[k][1]:+.2f}")
    lines.append("  mean over photographs:")
    for k in keys:
        vals = [r[k] for r in stats if k in r]
        lines.append(f"    {k:28s} CC={np.mean([v[0] for v in vals]):+.2f}  "
                     f"NSS={np.mean([v[1] for v in vals]):+.2f}  (n={len(vals)})")

    lines.append("")
    lines.append(f"Per object ({len(objects)} changed objects in these photographs):")
    fs = np.array([o["fix_share"] for o in objects])
    area = np.array([o["area"] for o in objects])

    def partial(x, y, z):
        """Spearman correlation of x and y with z partialled out."""
        rx, ry, rz = (rankdata(v) for v in (x, y, z))
        res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
        return float(np.corrcoef(res(rx), res(ry))[0, 1])

    for m in MODELS:
        for k in ("ratio", "feat_dist"):
            x = np.array([o[f"{m}_{k}"] for o in objects])
            rho, p = spearmanr(x, fs)
            lines.append(f"  share of fixations vs {m} {k:9s}: Spearman rho={rho:+.2f} p={p:.3f}"
                         f"   object area partialled out: {partial(x, fs, area):+.2f}")
    rho, p = spearmanr([o["area"] for o in objects], fs)
    lines.append(f"  share of fixations vs object area      : Spearman rho={rho:+.2f} p={p:.3f}")
    seen = [o for o in objects if o["times"]]
    lines.append("")
    lines.append(f"Participants (change blindness, 3 changes per trial): {len(seen)} changed objects shown")
    if len(seen) >= 3:
        mt = [np.median(o["times"]) for o in seen]
        rho, p = spearmanr([o["fix_share"] for o in seen], mt)
        lines.append(f"  share of fixations vs median time to find it: Spearman rho={rho:+.2f} p={p:.3f}"
                     f"   area partialled out: {partial([o['fix_share'] for o in seen], mt, [o['area'] for o in seen]):+.2f}")
    share = {(o["scene"], o["variant"]): o["fix_share"] for o in objects}
    hit = n = 0
    for ts in by_trial.values():
        first = [t for t in ts if t["rank"] == "1"]
        if len(ts) != 3 or not first or any((t["scene"], t["variant"]) not in share for t in ts):
            continue
        n += 1
        hit += max(ts, key=lambda t: share[t["scene"], t["variant"]]) is first[0]
    if n:
        lines.append(f"  the change found first is the most fixated of the 3 in {hit}/{n} trials "
                     f"({100 * hit / n:.0f} %, chance 33 %)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, m in zip(axes, MODELS):
        x = np.array([max(o[f"{m}_ratio"], 1e-3) for o in objects])
        ax.scatter(x, 100 * fs, c=["tab:red" if o["main"] == "1" else "tab:blue" for o in objects])
        for o, xi in zip(objects, x):
            ax.annotate(o["object"], (xi, 100 * o["fix_share"]), fontsize=7, alpha=0.7)
        rho, p = spearmanr(x, fs)
        ax.set_xscale("log")
        ax.set_xlabel(f"saillance Grad-CAM de l'objet, {m} (ratio)")
        ax.set_ylabel("% des fixations sur l'objet")
        ax.set_title(f"{m}: rho = {rho:+.2f} (p = {p:.2f}), {len(objects)} objets")
    fig.suptitle("Regard humain (COCO-FreeView) contre saillance des réseaux, objet par objet\n"
                 "(rouge = objet principal)")
    fig.tight_layout()
    fig.savefig(OUT / "eyetracking.png", dpi=110)
    plt.close(fig)

    (OUT / "eyetracking.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
