# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Make the changed versions of every scene picked by select_scenes.py.

Each scene is cropped to 4:3 and resized to 1024x768 ("original.jpg"). Then,
for every changeable object, one variant differs from the original by that
object only:

  - colour:    the object's hue is rotated (needs a saturated object);
  - disappear: the object is erased and the background rebuilt by LaMa
               (big-lama TorchScript, build/models/big-lama.pt).

Colour changes go only to man-made, saturated objects: a blue horse, blue
skin or a blue banana is a change nobody could miss for its oddity alone.
The main object only ever changes colour (erasing a large central object
leaves an obvious hole), so a natural main object gets no variant. Secondary
objects alternate between the two types where colour is possible.

Outside the changed region a variant is the original, pixel for pixel, so the
flicker shows nothing but the change.

Writes build/variants/<scene>/{original.jpg, vK.jpg, vK_mask.png},
build/variants/variants.csv and build/review.html.

Usage (from the example folder):
  build/venv/Scripts/python tools/make_variants.py
"""

import csv
import html
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO

HERE = Path(__file__).resolve().parent.parent
COCO_DIR = HERE / "build/coco"
SCENES = HERE / "build/scenes.json"
OUT = HERE / "build/variants"
REVIEW = HERE / "build/review.html"
LAMA = HERE / "build/models/big-lama.pt"

W, H = 1024, 768
MIN_SATURATION = 70   # mean HSV saturation (0-255) for a visible colour change
ERASE_DILATE = 15     # px: COCO outlines are tight; also take the edge/shadow
FEATHER = 7           # px: soft edge of the pasted region
JPEG_QUALITY = 95

# COCO categories whose colour is fixed by nature (or, for the last two, is
# their meaning): these can only disappear. Same lists as select_scenes.py.
from select_scenes import COLOUR_NEVER as NATURAL, ERASE_MAX, ERASE_MAX_DEFAULT


def load_lama():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(LAMA), map_location=dev)
    model.eval()
    return model, dev


def inpaint(model, dev, img_rgb, mask):
    """LaMa on the whole image; mask is uint8 0/1. Returns uint8 RGB."""
    t = torch.from_numpy(img_rgb).permute(2, 0, 1)[None].float().div(255).to(dev)
    m = torch.from_numpy(mask)[None, None].float().to(dev)
    with torch.no_grad():
        out = model(t, m)[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def feathered(mask, px):
    """Float alpha in [0, 1], soft over px pixels, zero outside the mask."""
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), px / 2)
    return np.clip(a, 0, 1) * (mask > 0)


def recolour(img_rgb, mask):
    """Rotate the hue of the masked object by the angle that moves it most."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # H in 0..179
    shifted = hsv.copy()
    shifted[..., 0] = (hsv[..., 0].astype(int) + 90) % 180  # opposite hue
    out = cv2.cvtColor(shifted, cv2.COLOR_HSV2RGB)
    a = feathered(mask, 3)[..., None]
    return (img_rgb * (1 - a) + out * a).round().astype(np.uint8)


def erase(model, dev, img_rgb, mask):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERASE_DILATE + 1,) * 2)
    region = cv2.dilate(mask, k)
    filled = inpaint(model, dev, img_rgb, region)
    a = feathered(cv2.dilate(region, k), FEATHER)[..., None]
    return (img_rgb * (1 - a) + filled * a).round().astype(np.uint8)


def save_mask(path, mask):
    """Black object on white, like the CB Database masks."""
    cv2.imwrite(str(path), np.where(mask > 0, 0, 255).astype(np.uint8))


def thumb_with_outline(img_rgb, mask, path):
    t = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(t, contours, -1, (0, 0, 255), 3)
    cv2.imwrite(str(path), cv2.resize(t, (W // 2, H // 2), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def photo(split, name):
    """Path of a COCO photograph, downloaded first if it is not here yet."""
    p = COCO_DIR / split / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"http://images.cocodataset.org/{split}/{name}", p)
    return p


def main():
    scenes = json.loads(SCENES.read_text())
    cocos = {}  # split -> COCO, loaded on first use
    model, dev = load_lama()
    print(f"LaMa on {dev}, {len(scenes)} scenes")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, review = [], []
    for si, sc in enumerate(scenes):
        sid = f"s{sc['coco_id']}"  # stable scene ID: the COCO image ID
        d = OUT / sid
        d.mkdir(exist_ok=True)
        if sc["split"] not in cocos:
            cocos[sc["split"]] = COCO(str(COCO_DIR / f"annotations/instances_{sc['split']}.json"))
        coco = cocos[sc["split"]]
        x0, y0, cw, ch = sc["crop"]
        src = cv2.imread(str(photo(sc["split"], sc["file"])))[y0:y0 + ch, x0:x0 + cw]
        img = cv2.cvtColor(cv2.resize(src, (W, H), interpolation=cv2.INTER_LANCZOS4), cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(d / "original.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        def obj_mask(o):
            m = coco.annToMask(coco.loadAnns(o["ann_id"])[0])[y0:y0 + ch, x0:x0 + cw]
            return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        def recolourable(o):
            # a grey object cannot change hue
            return o["category"] not in NATURAL and hsv[..., 1][obj_mask(o) > 0].mean() >= MIN_SATURATION

        def erasable(o):
            return obj_mask(o).sum() <= W * H * ERASE_MAX.get(o["category"], ERASE_MAX_DEFAULT)

        plan = []
        if recolourable(sc["main"]):
            plan.append((sc["main"], "colour", True))
        want_colour = True  # alternate, starting with colour
        for o in sc["secondary"]:
            can_colour, can_erase = recolourable(o), erasable(o)
            if can_colour and (want_colour or not can_erase):
                change = "colour"
            elif can_erase:
                change = "disappear"
            else:
                continue  # too big to erase cleanly, and cannot change colour
            plan.append((o, change, False))
            want_colour = change == "disappear"

        cells = [f'<img src="variants/{sid}/original.jpg" width="256" title="original">']
        for k, (o, change, is_main) in enumerate(plan, start=1):
            m = obj_mask(o)
            out = recolour(img, m) if change == "colour" else erase(model, dev, img, m)
            cv2.imwrite(str(d / f"v{k}.jpg"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            save_mask(d / f"v{k}_mask.png", m)
            thumb_with_outline(out, m, d / f"v{k}_review.jpg")
            ys, xs = np.nonzero(m)
            rows.append({
                "scene": sid, "coco_split": sc["split"], "coco_id": sc["coco_id"], "variant": f"v{k}",
                "object": o["category"], "main": int(is_main), "change": change,
                "area_px": int(m.sum()), "cx": round(float(xs.mean()), 1), "cy": round(float(ys.mean()), 1),
                "x0": int(xs.min()), "y0": int(ys.min()), "x1": int(xs.max()), "y1": int(ys.max()),
            })
            label = html.escape(f"{sid}/v{k} {o['category']} ({'MAIN, ' if is_main else ''}{change})")
            cells.append(f'<figure><img src="variants/{sid}/v{k}_review.jpg" width="256">'
                         f'<figcaption>{label}</figcaption></figure>')
        review.append(f"<h3>{sid} — main: {html.escape(sc['main']['category'])}</h3>"
                      f"<div class=row>{''.join(cells)}</div>")
        print(f"[{si + 1}/{len(scenes)}] {sid} {sc['main']['category']}: {len(plan)} variants")

    with open(OUT / "variants.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    REVIEW.write_text(
        "<!doctype html><meta charset=utf-8><title>Variant review</title>"
        "<style>body{font:13px sans-serif;background:#222;color:#ddd}"
        ".row{display:flex;flex-wrap:wrap;gap:6px}figure{margin:0}"
        "figcaption{max-width:256px}</style>"
        "<p>Red outline = changed object. List bad variants in tools/rejected.txt "
        "(one <code>scene/vK</code> per line).</p>" + "".join(review), encoding="utf-8")
    print(f"{len(rows)} variants -> {OUT}; review sheet {REVIEW}")


if __name__ == "__main__":
    main()
