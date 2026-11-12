# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Saliency of every changed object for two ImageNet classifiers.

Models: ResNet-50 (CNN, torchvision IMAGENET1K_V2) and ViT-B/16 (transformer,
timm vit_base_patch16_224.augreg2_in21k_ft_in1k). Both are explained the same
way, with Grad-CAM for the class each model predicts on the original photograph
(ResNet: last block of layer4, 7x7 map; ViT: norm1 of the last block, 14x14
patch map). Images are resized to 224x224 (the 4:3 photographs are squashed,
not cropped, so objects near the border keep their saliency) and the maps are
resized back to 1024x768.

Per variant and model, saliency/model_saliency.csv gives:
  mass       share of the original image's saliency that falls on the object
  ratio      mean saliency on the object / mean over the image (1 = average)
  peak       highest saliency on the object (maps are scaled 0-1)
  feat_dist  1 - cosine similarity of the penultimate features, original vs
             variant: how much the change moves the model's representation
  js         Jensen-Shannon divergence of the class probabilities, original
             vs variant
  top1_changed  the predicted class differs between original and variant
plus the class predicted on the original (top1_class, top1_prob).

Also writes saliency/maps/<model>/<scene>.png (8-bit map, 1024x768) and, for
viewing, build/saliency_overlays/<scene>.jpg and build/saliency_overlays.jpg.

Usage (from the example folder):
  build/venv/Scripts/python tools/model_saliency.py
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from stim import variant_rgb
from torchvision.models import ResNet50_Weights, resnet50

HERE = Path(__file__).resolve().parent.parent
STIM = HERE / "stimuli"
OUT = HERE / "saliency"
VIEW = HERE / "build/saliency_overlays"
W, H, SIDE = 1024, 768, 224

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]  # same order in timm


class Model:
    def __init__(self, name, net, mean, std, layer, reshape, features):
        self.name, self.net = name, net.eval().to(DEV)
        self.mean = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
        self.cam = GradCAM(model=self.net, target_layers=[layer], reshape_transform=reshape)
        self.features = features

    def tensor(self, rgb):
        x = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(DEV).permute(2, 0, 1)[None].float() / 255
        return (x - self.mean) / self.std

    @torch.no_grad()
    def outputs(self, x):
        f = self.features(x)
        return f, F.softmax(self.net(x), dim=1)[0]

    def saliency(self, x, cls):
        m = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR).clip(0, 1)


def load_models():
    r = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    r_body = torch.nn.Sequential(*list(r.children())[:-1])
    resnet = Model("resnet50", r, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
                   r.layer4[-1], None, lambda x: r_body(x).flatten(1))

    v = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
    cfg = v.pretrained_cfg
    grid = SIDE // 16

    def tokens_to_map(t):  # drop [CLS], patches back onto their 14x14 grid
        return t[:, 1:, :].reshape(t.size(0), grid, grid, t.size(2)).permute(0, 3, 1, 2)

    vit = Model("vit_b16", v, cfg["mean"], cfg["std"], v.blocks[-1].norm1, tokens_to_map,
                lambda x: v.forward_head(v.forward_features(x), pre_logits=True))
    return [resnet, vit]


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: torch.sum(a * (torch.log(a + 1e-12) - torch.log(b + 1e-12)))
    return float((kl(p, m) + kl(q, m)) / 2)


def overlay(img, sal):
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0)


def main():
    rows = list(csv.DictReader(open(STIM / "manifest.csv")))
    scenes = sorted({r["scene"] for r in rows})
    models = load_models()
    for m in models:
        (OUT / "maps" / m.name).mkdir(parents=True, exist_ok=True)
    VIEW.mkdir(parents=True, exist_ok=True)

    out, thumbs = [], []
    for si, scene in enumerate(scenes):
        orig = rgb(STIM / scene / "original.jpg")
        variants = [r for r in rows if r["scene"] == scene]
        panels = [cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)]
        for m in models:
            x = m.tensor(orig)
            f0, p0 = m.outputs(x)
            cls = int(p0.argmax())
            sal = m.saliency(x, cls)
            cv2.imwrite(str(OUT / "maps" / m.name / f"{scene}.png"), (sal * 255).round().astype(np.uint8))
            panel = overlay(orig, sal)
            cv2.putText(panel, f"{m.name}: {LABELS[cls]} {float(p0[cls]):.2f}", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
            for r in variants:
                mask = cv2.imread(str(STIM / scene / f"{r['variant']}_mask.png"), cv2.IMREAD_GRAYSCALE) < 128
                f1, p1 = m.outputs(m.tensor(variant_rgb(scene, r["variant"])))
                out.append({
                    "scene": scene, "variant": r["variant"], "object": r["object"],
                    "main": r["main"], "change": r["change"], "area_px": int(mask.sum()),
                    "model": m.name, "top1_class": LABELS[cls], "top1_prob": round(float(p0[cls]), 4),
                    "mass": round(float(sal[mask].sum() / sal.sum()), 5),
                    "ratio": round(float(sal[mask].mean() / sal.mean()), 4),
                    "peak": round(float(sal[mask].max()), 4),
                    "feat_dist": round(float(1 - F.cosine_similarity(f0, f1)[0]), 6),
                    "js": round(js(p0, p1), 6),
                    "top1_changed": int(int(p1.argmax()) != cls),
                })
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel, contours, -1, (255, 255, 255), 2)
            panels.append(panel)
        sheet = np.hstack(panels)
        cv2.imwrite(str(VIEW / f"{scene}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        thumbs.append(cv2.resize(sheet, (sheet.shape[1] // 4, sheet.shape[0] // 4)))
        print(f"{si + 1:2d}/{len(scenes)} {scene}  " +
              "  ".join(f"{r['model']}={r['top1_class']}" for r in out[-len(variants) * len(models)::len(variants)]))

    with open(OUT / "model_saliency.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    cv2.imwrite(str(HERE / "build/saliency_overlays.jpg"), np.vstack(thumbs), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"{len(out)} rows -> {OUT / 'model_saliency.csv'}")


if __name__ == "__main__":
    main()
