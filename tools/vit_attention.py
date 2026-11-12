# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Self-attention maps of two Vision Transformers, compared with the
participants' change detection and with eye tracking.

Unlike Grad-CAM (a gradient explanation of one class), these maps read the
transformer's own attention, with no class:

  cls      attention of the [CLS] token to the image patches in the last
           layer, averaged over the 12 heads (Caron et al., 2021)
  rollout  attention rollout (Abnar & Zuidema, 2020): the attention of every
           layer, plus the residual connection, multiplied through the
           network, row of the [CLS] token

for two ViT-B/16 that differ only by their training:

  vit   supervised on ImageNet (timm vit_base_patch16_224.augreg2_in21k_ft_in1k,
        the model of model_saliency.py)
  dino  self-supervised, no labels (timm vit_base_patch16_224.dino; Caron et
        al., 2021)

Images are resized to 224x224 (14x14 patches) like in model_saliency.py; maps
are resized back to 1024x768 and written to saliency/attention/<map>/<scene>.png.

For each map, writes build/comparison/attention.txt with, side by side with
Grad-CAM and feat_dist (from saliency/model_saliency.csv):
  - within each trial, how often the change found first is the one the map
    rates highest (chance 33 %), and the rank correlation of the map's value
    on each changed object with the time to find it (object area
    partialled out too);
  - against COCO-FreeView fixations: CC and NSS per photograph, and the
    correlation, across changed objects, with the share of fixations on them.

Usage (from Saliency-Change-Blindness):
  build/venv/Scripts/python tools/vit_attention.py [--data DIR]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
from caption_importance import fixation_share, image_size  # noqa: E402

OUT = HERE / "saliency/attention"
FIX = HERE / "build/eyetracking/COCOFreeView_fixations_trainval.json"
W, H, SIDE = 1024, 768, 224
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {
    "vit": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "dino": "vit_base_patch16_224.dino",
}


class Attention:
    def __init__(self, name):
        self.net = timm.create_model(name, pretrained=True).eval().to(DEV)
        cfg = self.net.pretrained_cfg
        self.mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
        self.maps = []
        for blk in self.net.blocks:
            blk.attn.fused_attn = False  # so the attention matrix is computed
            blk.attn.attn_drop.register_forward_hook(lambda m, i, o: self.maps.append(o.detach()))

    @torch.no_grad()
    def __call__(self, rgb):
        x = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(DEV).permute(2, 0, 1)[None].float() / 255
        self.maps = []
        self.net((x - self.mean) / self.std)
        A = [a[0].mean(0) for a in self.maps]  # per layer, heads averaged: (197, 197)
        g = SIDE // 16
        cls = A[-1][0, 1:]
        roll = torch.eye(A[0].shape[0], device=DEV)
        for a in A:
            a = a + torch.eye(a.shape[0], device=DEV)
            roll = (a / a.sum(-1, keepdim=True)) @ roll
        out = {}
        for k, v in (("cls", cls), ("rollout", roll[0, 1:])):
            m = v.reshape(g, g).cpu().numpy()
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
            out[k] = (m - m.min()) / (m.max() - m.min() + 1e-12)
        return out


def partial(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def norm(m):
    return (m - m.mean()) / (m.std() + 1e-12)


def fixations(sc, records):
    w, h = image_size(sc)
    s = min(1680 / w, 1050 / h)
    ox, oy = (1680 - w * s) / 2, (1050 - h * s) / 2
    x0, y0, cw, ch = sc["crop"]
    pts = []
    for r in records:
        for x, y in list(zip(r["X"], r["Y"]))[1:]:
            px, py = ((x - ox) / s - x0) * W / cw, ((y - oy) / s - y0) * H / ch
            if 0 <= px < W and 0 <= py < H:
                pts.append((px, py))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "goxpy_data")
    args = ap.parse_args()
    manifest = list(csv.DictReader(open(HERE / "stimuli/manifest.csv")))
    scenes = sorted({r["scene"] for r in manifest})
    info = {f"s{s['coco_id']}": s for s in json.load(open(HERE / "build/scenes.json"))}

    # value of every map on every changed object
    value = defaultdict(dict)  # (scene, variant) -> map name -> ratio
    maps = {}                  # (map name, scene) -> map
    for mname, tname in MODELS.items():
        att = Attention(tname)
        for s in scenes:
            img = cv2.cvtColor(cv2.imread(str(HERE / "stimuli" / s / "original.jpg")), cv2.COLOR_BGR2RGB)
            for k, m in att(img).items():
                name = f"{mname}_{k}"
                maps[name, s] = m
                d = OUT / name
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{s}.png"), (m * 255).round().astype(np.uint8))
        del att
        torch.cuda.empty_cache()
    names = sorted({n for n, _ in maps})
    for v in manifest:
        mask = cv2.imread(str(HERE / "stimuli" / v["scene"] / f"{v['variant']}_mask.png"), 0) < 128
        for n in names:
            m = maps[n, v["scene"]]
            value[v["scene"], v["variant"]][n] = float(m[mask].mean() / (m.mean() + 1e-12))
    for r in csv.DictReader(open(HERE / "saliency/model_saliency.csv")):
        key = (r["scene"], r["variant"])
        if key in value:
            value[key][f"gradcam_{r['model']}"] = float(r["ratio"])
            value[key][f"featdist_{r['model']}"] = float(r["feat_dist"])
    measures = names + ["gradcam_resnet50", "gradcam_vit_b16", "featdist_resnet50", "featdist_vit_b16"]
    dino_fd = HERE / "saliency/dino_feat_dist.csv"  # tools/dino_feat_dist.py
    if dino_fd.exists():
        for r in csv.DictReader(open(dino_fd)):
            if (r["scene"], r["variant"]) in value:
                value[r["scene"], r["variant"]]["featdist_dino"] = float(r["feat_dist"])
                value[r["scene"], r["variant"]]["featdist_dino_patches"] = float(r["feat_dist_patches"])
        measures += ["featdist_dino", "featdist_dino_patches"]

    lines = ["Maps: vit = supervised ViT-B/16, dino = self-supervised ViT-B/16; "
             "cls = last-layer [CLS] attention, rollout = attention rollout", ""]

    # participants (3 changes per trial)
    rows = []
    for f in sorted(args.data.glob("*Change-Blindness*_sub-*.csv")):
        if "rank" in open(f, encoding="utf-8").readline():
            rows += [dict(t, file=f.name) for t in csv.DictReader(open(f, encoding="utf-8")) if t["phase"] == "main"]
    if rows:
        by_trial = defaultdict(list)
        for t in rows:
            by_trial[t["file"], t["trial"]].append(t)
        rt = np.array([float(t["rt_ms"]) for t in rows])
        area = np.array([float(t["mask_area"]) for t in rows])
        lines.append(f"PARTICIPANTS: {len({t['file'] for t in rows})} session(s), {len(by_trial)} trials, {len(rows)} changes")
        lines.append(f"  {'measure':22s} {'found first = top':>18s} {'rho time':>9s} {'(area out)':>10s}")
        for n in measures:
            hit = cnt = 0
            for ts in by_trial.values():
                first = [t for t in ts if t["rank"] == "1"]
                if len(ts) == 3 and first:
                    cnt += 1
                    hit += max(ts, key=lambda t: value[t["scene"], t["variant"]][n]) is first[0]
            x = np.array([value[t["scene"], t["variant"]][n] for t in rows])
            rho = spearmanr(x, rt)[0]
            lines.append(f"  {n:22s} {hit:3d}/{cnt} = {100 * hit / cnt:3.0f} %   {rho:+9.2f} {partial(x, rt, area):+10.2f}")
        lines.append("  (chance for 'found first = top' is 33 %; a negative rho = higher value, found sooner)")

    # eye tracking
    if FIX.exists():
        by = defaultdict(list)
        for r in json.load(open(FIX)):
            by[int(r["name"].split(".")[0])].append(r)
        yy, xx = np.mgrid[0:H, 0:W]
        centre = np.exp(-(((xx - W / 2) / (W / 4)) ** 2 + ((yy - H / 2) / (H / 4)) ** 2) / 2)
        cc, nss = defaultdict(list), defaultdict(list)
        for s in scenes:
            pts = fixations(info[s], by[info[s]["coco_id"]])
            dens = np.zeros((H, W))
            for x, y in pts:
                dens[int(y), int(x)] += 1
            dens = cv2.GaussianBlur(dens, (0, 0), 25)
            cands = {n: maps[n, s] for n in names}
            cands["centre_bias"] = centre
            for gm in ("resnet50", "vit_b16"):
                cands[f"gradcam_{gm}"] = cv2.imread(str(HERE / "saliency/maps" / gm / f"{s}.png"), 0) / 255.0
            for n, m in cands.items():
                cc[n].append(float(np.mean(norm(dens) * norm(m))))
                z = norm(m)
                nss[n].append(float(np.mean([z[int(y), int(x)] for x, y in pts])))
        lines.append("")
        lines.append(f"EYE TRACKING (COCO-FreeView, {len(scenes)} photographs): maps vs fixations")
        for n in sorted(cc, key=lambda n: -np.mean(cc[n])):
            lines.append(f"  {n:22s} CC={np.mean(cc[n]):+.2f}  NSS={np.mean(nss[n]):+.2f}")
        share, ar = [], []
        for v in manifest:
            sc = info[v["scene"]]
            share.append(fixation_share(sc, by[sc["coco_id"]], {"scene": v["scene"], "variant": v["variant"]}))
            ar.append(float(v["area_px"]))
        lines.append(f"  per changed object ({len(manifest)}): share of fixations on it vs the map's value")
        for n in measures:
            x = [value[v["scene"], v["variant"]][n] for v in manifest]
            lines.append(f"    {n:22s} rho={spearmanr(x, share)[0]:+.2f}  (area out {partial(x, share, ar):+.2f})")

    out = HERE / "build/comparison/attention.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # a sheet of maps for a few photographs
    rows_img = []
    for s in scenes[:8]:
        img = cv2.imread(str(HERE / "stimuli" / s / "original.jpg"))
        panels = [img]
        for n in ["vit_cls", "vit_rollout", "dino_cls", "dino_rollout"]:
            heat = cv2.applyColorMap((maps[n, s] * 255).astype(np.uint8), cv2.COLORMAP_JET)
            p = cv2.addWeighted(img, 0.45, heat, 0.55, 0)
            cv2.putText(p, n, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
            cv2.putText(p, n, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
            panels.append(p)
        rows_img.append(cv2.resize(np.hstack(panels), (1500, 225)))
    cv2.imwrite(str(HERE / "build/comparison/attention_maps.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
