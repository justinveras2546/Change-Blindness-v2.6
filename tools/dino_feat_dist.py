# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""How much each change moves the representation of DINO, a ViT-B/16
trained without labels (Caron et al., 2021; timm vit_base_patch16_224.dino).

Same measure as feat_dist in model_saliency.py: 1 - cosine similarity of the
final [CLS] embedding (768 numbers) between the original photograph and the
variant, images resized to 224x224. A second value uses the mean of the patch
embeddings instead of [CLS] (feat_dist_patches): it weighs every part of the
image equally.

Writes saliency/dino_feat_dist.csv (scene, variant, feat_dist,
feat_dist_patches), read by vit_attention.py.

Usage (from Saliency-Change-Blindness):
  build/venv/Scripts/python tools/dino_feat_dist.py
"""

import csv
from pathlib import Path

import cv2
import timm
import torch
import torch.nn.functional as F
from stim import variant_rgb

HERE = Path(__file__).resolve().parent.parent
SIDE = 224
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    net = timm.create_model("vit_base_patch16_224.dino", pretrained=True).eval().to(DEV)
    cfg = net.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed_rgb(img):
        x = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(DEV).permute(2, 0, 1)[None].float() / 255
        tokens = net.forward_features((x - mean) / std)  # (1, 197, 768), after the final norm
        return tokens[:, 0], tokens[:, 1:].mean(1)

    rows = []
    cache = {}
    for v in csv.DictReader(open(HERE / "stimuli/manifest.csv")):
        s = v["scene"]
        if s not in cache:
            orig = cv2.cvtColor(cv2.imread(str(HERE / "stimuli" / s / "original.jpg")), cv2.COLOR_BGR2RGB)
            cache = {s: embed_rgb(orig)}
        c0, p0 = cache[s]
        c1, p1 = embed_rgb(variant_rgb(s, v["variant"]))
        rows.append({"scene": s, "variant": v["variant"], "object": v["object"],
                     "feat_dist": round(float(1 - F.cosine_similarity(c0, c1)[0]), 6),
                     "feat_dist_patches": round(float(1 - F.cosine_similarity(p0, p1)[0]), 6)})
    out = HERE / "saliency/dino_feat_dist.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} variants -> {out}")


if __name__ == "__main__":
    main()
