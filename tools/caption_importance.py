# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Human importance of each object from the COCO captions.

Every COCO photograph has 5 captions written by different people asked to
describe it in one sentence. An object that people mention is one they found
important; one nobody mentions was ignored (Spain & Perona, 2011; Yun et al.,
2013; Berg et al., 2012). For every object category annotated in a scene,
this script counts how many of the 5 captions mention it (with synonyms and
plurals: "monitor" and "screen" count for tv) and where the first mention
comes in the sentence.

The importance of each changed object (the variants of the change-blindness
experiment) is then compared, across objects, with:
  - the models' Grad-CAM saliency of the object (ratio) and feat_dist
    (with the object's area partialled out too);
  - the share of COCO-FreeView fixations on it, where available;
  - the participants' time to find its change.

Needs captions_{train,val}2017.json (in annotations_trainval2017.zip) in
build/coco/annotations/. Writes saliency/caption_importance.csv (one row per
scene and annotated category) and build/comparison/captions.txt and
captions.png.

Usage (from Saliency-Change-Blindness):
  build/venv/Scripts/python tools/caption_importance.py [--data DIR]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, rankdata, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ANN = HERE / "build/coco/annotations"
OUT = HERE / "build/comparison"
FIX = HERE / "build/eyetracking/COCOFreeView_fixations_trainval.json"
MODELS = ["resnet50", "vit_b16"]

# Words that name each COCO category in a caption (singular; plurals added).
WORDS = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "girl", "child", "children",
               "kid", "guy", "lady", "player", "someone", "rider", "skier", "surfer"],
    "bicycle": ["bicycle", "bike"], "car": ["car", "vehicle", "suv", "taxi"], "motorcycle": ["motorcycle", "motorbike", "scooter"],
    "airplane": ["airplane", "plane", "jet", "aircraft", "airliner"], "bus": ["bus", "buses"],
    "train": ["train", "locomotive"], "truck": ["truck", "lorry", "van"], "boat": ["boat", "ship", "yacht", "canoe"],
    "traffic light": ["traffic light", "stop light", "signal"], "fire hydrant": ["hydrant"],
    "stop sign": ["stop sign"], "parking meter": ["parking meter", "meter"], "bench": ["bench"],
    "bird": ["bird", "pigeon", "duck", "gull", "seagull"], "cat": ["cat", "kitten", "kitty"],
    "dog": ["dog", "puppy"], "horse": ["horse", "pony"], "sheep": ["sheep", "lamb", "ram"],
    "cow": ["cow", "cattle", "bull"], "elephant": ["elephant"], "bear": ["bear"], "zebra": ["zebra"],
    "giraffe": ["giraffe"], "backpack": ["backpack", "back pack"], "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "bag"], "tie": ["tie"], "suitcase": ["suitcase", "luggage"],
    "frisbee": ["frisbee"], "skis": ["ski"], "snowboard": ["snowboard"], "sports ball": ["ball"],
    "kite": ["kite"], "baseball bat": ["bat"], "baseball glove": ["glove", "mitt"],
    "skateboard": ["skateboard"], "surfboard": ["surfboard", "surf board", "board"],
    "tennis racket": ["racket", "racquet"], "bottle": ["bottle"], "wine glass": ["wine glass", "glass", "wine"],
    "cup": ["cup", "mug", "coffee"], "fork": ["fork", "utensil", "silverware"],
    "knife": ["knife", "knives", "utensil", "silverware"], "spoon": ["spoon", "utensil", "silverware"],
    "bowl": ["bowl"], "banana": ["banana"], "apple": ["apple"], "sandwich": ["sandwich", "burrito", "wrap", "burger"],
    "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"], "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"], "donut": ["donut", "doughnut"], "cake": ["cake", "dessert"],
    "chair": ["chair", "seat", "stool"], "couch": ["couch", "sofa"], "potted plant": ["plant", "flower", "houseplant"],
    "bed": ["bed"], "dining table": ["table", "desk", "counter"], "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor", "screen", "computer", "display", "imac"],
    "laptop": ["laptop", "computer", "notebook", "macbook"], "mouse": ["mouse", "mice"],
    "remote": ["remote", "controller", "control"], "keyboard": ["keyboard"],
    "cell phone": ["phone", "cellphone", "cell phone", "iphone", "smartphone", "mobile"],
    "microwave": ["microwave"], "oven": ["oven", "stove", "range"], "toaster": ["toaster"],
    "sink": ["sink", "basin"], "refrigerator": ["refrigerator", "fridge", "freezer"],
    "book": ["book", "magazine", "novel"], "clock": ["clock"], "vase": ["vase"], "scissors": ["scissors"],
    "teddy bear": ["teddy", "stuffed animal", "stuffed bear"], "hair drier": ["hair dryer", "hair drier", "dryer"],
    "toothbrush": ["toothbrush"],
}


def pattern(words):
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})(?:s|es)?\b", re.IGNORECASE)


PATTERNS = {c: pattern(ws) for c, ws in WORDS.items()}


def mentions(caption, category):
    """Position (0-1, start of the sentence = 0) of the first mention, or None."""
    m = PATTERNS[category].search(caption)
    return None if m is None else m.start() / max(len(caption), 1)


def partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "goxpy_data")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    scenes = {f"s{s['coco_id']}": s for s in json.load(open(HERE / "build/scenes.json"))}
    manifest = list(csv.DictReader(open(HERE / "stimuli/manifest.csv")))
    ours = sorted({r["scene"] for r in manifest})

    caps = defaultdict(list)
    cats_in = defaultdict(set)
    for split in ("val2017", "train2017"):
        ids = {scenes[s]["coco_id"] for s in ours if scenes[s]["split"] == split}
        for a in json.load(open(ANN / f"captions_{split}.json"))["annotations"]:
            if a["image_id"] in ids:
                caps[a["image_id"]].append(a["caption"].strip())
        inst = json.load(open(ANN / f"instances_{split}.json"))
        names = {c["id"]: c["name"] for c in inst["categories"]}
        for a in inst["annotations"]:
            if a["image_id"] in ids:
                cats_in[a["image_id"]].add(names[a["category_id"]])

    # importance of every annotated category, per scene
    rows = []
    importance = {}
    for s in ours:
        i = scenes[s]["coco_id"]
        cs = caps[i][:5]
        for cat in sorted(cats_in[i]):
            pos = [mentions(c, cat) for c in cs]
            hit = [p for p in pos if p is not None]
            importance[s, cat] = len(hit) / len(cs)
            rows.append({"scene": s, "category": cat, "n_captions": len(cs),
                         "mention_rate": round(len(hit) / len(cs), 2),
                         "first_position": round(float(np.mean(hit)), 2) if hit else "",
                         "main": int(cat == scenes[s]["main"]["category"])})
    with open(HERE / "saliency/caption_importance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"{len(ours)} scenes, {len(rows)} (scene, category) pairs, captions per scene: "
             f"{min(len(caps[scenes[s]['coco_id']]) for s in ours)}-{max(len(caps[scenes[s]['coco_id']]) for s in ours)}"]
    main_rate = [r["mention_rate"] for r in rows if r["main"]]
    other_rate = [r["mention_rate"] for r in rows if not r["main"]]
    lines.append(f"Mentioned in the captions: main object {100 * np.mean(main_rate):.0f} % of captions, "
                 f"other objects {100 * np.mean(other_rate):.0f} % "
                 f"(Mann-Whitney p = {mannwhitneyu(main_rate, other_rate).pvalue:.2g})")
    never = sum(r["mention_rate"] == 0 for r in rows if not r["main"])
    lines.append(f"Other objects never mentioned: {never}/{len(other_rate)}")
    lines.append("Examples:")
    for s in ours[:4]:
        i = scenes[s]["coco_id"]
        lines.append(f"  {s}: " + " | ".join(caps[i][:2]))
        lines.append("     " + ", ".join(f"{r['category']}={r['mention_rate']:.1f}" for r in rows if r["scene"] == s))

    # per changed object
    model = {(r["scene"], r["variant"], r["model"]): r for r in csv.DictReader(open(HERE / "saliency/model_saliency.csv"))}
    objs = []
    for v in manifest:
        o = {"scene": v["scene"], "variant": v["variant"], "object": v["object"], "main": v["main"],
             "area": float(v["area_px"]), "imp": importance[v["scene"], v["object"]]}
        for m in MODELS:
            o[f"{m}_ratio"] = float(model[v["scene"], v["variant"], m]["ratio"])
            o[f"{m}_feat"] = float(model[v["scene"], v["variant"], m]["feat_dist"])
        objs.append(o)
    imp = np.array([o["imp"] for o in objs])
    area = np.array([o["area"] for o in objs])
    lines.append("")
    lines.append(f"Changed objects ({len(objs)}): caption importance vs the models")
    for m in MODELS:
        for k, lab in (("ratio", "Grad-CAM ratio"), ("feat", "feat_dist")):
            x = np.array([o[f"{m}_{k}"] for o in objs])
            rho, p = spearmanr(x, imp)
            lines.append(f"  {m:9s} {lab:14s}: rho={rho:+.2f} p={p:.3g}   area partialled out: {partial(x, imp, area):+.2f}")
    rho, p = spearmanr(area, imp)
    lines.append(f"  object area             : rho={rho:+.2f} p={p:.3g}")
    sec = [o for o in objs if o["main"] == "0"]
    si = np.array([o["imp"] for o in sec])
    lines.append(f"  other objects only ({len(sec)}):")
    for m in MODELS:
        for k, lab in (("ratio", "Grad-CAM ratio"), ("feat", "feat_dist")):
            x = np.array([o[f"{m}_{k}"] for o in sec])
            rho, p = spearmanr(x, si)
            lines.append(f"    {m:9s} {lab:14s}: rho={rho:+.2f} p={p:.3g}   area partialled out: "
                         f"{partial(x, si, np.array([o['area'] for o in sec])):+.2f}")

    # eye tracking
    if FIX.exists():
        by = defaultdict(list)
        for r in json.load(open(FIX)):
            by[int(r["name"].split(".")[0])].append(r)
        fx = []
        for o in objs:
            sc = scenes[o["scene"]]
            if sc["coco_id"] not in by:
                continue
            fx.append((o, fixation_share(sc, by[sc["coco_id"]], o)))
        if fx:
            rho, p = spearmanr([o["imp"] for o, _ in fx], [f for _, f in fx])
            lines.append("")
            lines.append(f"Caption importance vs share of eye fixations (COCO-FreeView, {len(fx)} objects): "
                         f"rho={rho:+.2f} p={p:.3g}   area partialled out: "
                         f"{partial([o['imp'] for o, _ in fx], [f for _, f in fx], [o['area'] for o, _ in fx]):+.2f}")

    # participants
    trials = []  # current design only (3 changes per trial, "rank" column), all participants
    for f in sorted(args.data.glob("*Change-Blindness*_sub-*.csv")):
        if "rank" in open(f, encoding="utf-8").readline():
            trials += [t for t in csv.DictReader(open(f, encoding="utf-8")) if t["phase"] == "main"]
    if trials:
        limit = max(float(t["rt_ms"]) for t in trials)
        t_imp = [importance.get((t["scene"], t["object"]), np.nan) for t in trials]
        t_rt = [float(t["rt_ms"]) for t in trials]
        ok = [i for i, v in enumerate(t_imp) if not np.isnan(v)]
        rho, p = spearmanr([t_imp[i] for i in ok], [t_rt[i] for i in ok])
        lines.append(f"Caption importance vs time to find the change ({len(ok)} changes): rho={rho:+.2f} p={p:.3g}")
        for lvl in sorted({t_imp[i] for i in ok}):
            sel = [t_rt[i] for i in ok if t_imp[i] == lvl]
            found = sum(v < limit for v in sel)
            lines.append(f"    mentioned in {lvl * 5:.0f}/5 captions: {len(sel):2d} changes, found {found}, "
                         f"median time {np.median(sel) / 1000:.1f} s")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    groups = sorted(set(imp))
    for ax, m in zip(axes, MODELS):
        data = [[o[f"{m}_feat"] for o in objs if o["imp"] == g] for g in groups]
        ax.boxplot(data, tick_labels=[f"{g * 5:.0f}/5" for g in groups])
        ax.set_yscale("log")
        ax.set_xlabel("objet cité dans … des 5 descriptions")
        ax.set_ylabel(f"distance des représentations ({m})")
        rho, p = spearmanr([o[f"{m}_feat"] for o in objs], imp)
        ax.set_title(f"{m}: rho = {rho:+.2f}")
    fig.suptitle("Importance humaine (descriptions COCO) contre sensibilité des réseaux au changement")
    fig.tight_layout()
    fig.savefig(OUT / "captions.png", dpi=110)
    plt.close(fig)

    (OUT / "captions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def fixation_share(sc, records, o, W=1024, H=768):
    """Share of the FreeView fixations (first one dropped) on the object's
    mask dilated by 15 px; same mapping as compare_eyetracking.py."""
    info_w, info_h = sc["_wh"] if "_wh" in sc else image_size(sc)
    s = min(1680 / info_w, 1050 / info_h)
    ox, oy = (1680 - info_w * s) / 2, (1050 - info_h * s) / 2
    x0, y0, cw, ch = sc["crop"]
    mask = cv2.imread(str(HERE / "stimuli" / o["scene"] / f"{o['variant']}_mask.png"), 0) < 128
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    pts = []
    for r in records:
        for x, y in list(zip(r["X"], r["Y"]))[1:]:
            px, py = ((x - ox) / s - x0) * W / cw, ((y - oy) / s - y0) * H / ch
            if 0 <= px < W and 0 <= py < H:
                pts.append(mask[int(py), int(px)])
    return float(np.mean(pts)) if pts else 0.0


_SIZES = {}


def image_size(sc):
    if sc["coco_id"] not in _SIZES:
        img = cv2.imread(str(HERE / "build/coco" / sc["split"] / sc["file"]))
        _SIZES[sc["coco_id"]] = (img.shape[1], img.shape[0])
    sc["_wh"] = _SIZES[sc["coco_id"]]
    return sc["_wh"]


if __name__ == "__main__":
    main()
