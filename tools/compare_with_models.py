# Copyright (2026) Christophe Pallier <christophe@pallier.org>
# Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

"""Compare participants' data with the models' saliency maps.

Reads every data file of Saliency-Change-Blindness and Saliency-Guided-Occlusion
in the data folder (default ~/goxpy_data) and writes to build/comparison/:

  change_blindness.png   detection time of each change against the models'
                         saliency of the changed object (ratio) and against
                         how much the change moves the model's representation
                         (feat_dist)
  occlusion.png          identification accuracy per occlusion condition
  maps_NN.jpg            per photograph: original, then ResNet-50 and
                         ViT-B/16 Grad-CAM maps of the main object, with the
                         participants' clicks on the changes (dots: green =
                         on the change, red = missed) and the occlusion
                         results
  summary.txt            the numbers

Usage (from Saliency-Change-Blindness):
  build/venv/Scripts/python tools/compare_with_models.py [--data DIR]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
OCC = HERE.parent / "Saliency-Guided-Occlusion"
OUT = HERE / "build/comparison"
MODELS = ["resnet50", "vit_b16"]
CONDS = ["intact", "resnet50", "vit_b16", "shuffled"]


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def latest_per_subject(files):
    """Keep the most recent non-empty file of each subject (a subject ID
    replays the same session, so earlier runs are repeats)."""
    best = {}
    for f in sorted(files):
        r = rows(f)
        if r:
            best[r[0]["subject_id"]] = (f, r)
    return best


def overlay(img, sal):
    heat = cv2.applyColorMap((sal * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.45, heat, 0.55, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "goxpy_data")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    # ---- change blindness -------------------------------------------------
    # Only the sessions of the current design (3 changes per trial, a "rank"
    # column); the first version (one change per trial) used other photographs.
    cb = latest_per_subject(f for f in args.data.glob("*Change-Blindness*_sub-*.csv")
                            if "rank" in open(f, encoding="utf-8").readline())
    model = {(r["scene"], r["variant"], r["model"]): r for r in rows(HERE / "saliency/model_saliency.csv")}
    trials = [t for _, rs in cb.values() for t in rs if t["phase"] == "main"]
    lines.append(f"CHANGE BLINDNESS: {len(cb)} participant(s), {len(trials)} changes "
                 f"({len(trials) // 3} photographs x 3 changes)")
    for f, rs in cb.values():
        lines.append(f"  {Path(f).name}")
    found = [t for t in trials if t["found"] == "true"]
    if trials:
        n_false = {(t["subject_id"], t["trial"]): len([c for c in t["false_clicks"].split("|") if c]) for t in trials}
        lines.append(f"  changes found: {len(found)}/{len(trials)}; clicks away from any change: {sum(n_false.values())}")
    for ch in ("disappear", "colour"):
        sub = [t for t in trials if t["change"] == ch]
        if sub:
            lines.append(f"  {ch:9s}: found {sum(t['found'] == 'true' for t in sub)}/{len(sub)}, "
                         f"found first in {sum(t['rank'] == '1' for t in sub)} trials")
    if found:
        lines.append(f"  median time to find a change: {np.median([float(t['rt_ms']) for t in found]):.0f} ms")

    # within each trial: is the change found first the one the model rates highest?
    by_trial = defaultdict(list)
    for t in trials:
        by_trial[t["subject_id"], t["trial"]].append(t)
    lines.append("  Within each photograph, is the change found first the one the model rates highest?"
                 " (chance = 33 %)")
    for m in MODELS:
        for key in ("ratio", "feat_dist"):
            hit = n = 0
            for ts in by_trial.values():
                first = [t for t in ts if t["rank"] == "1"]
                vals = [model.get((t["scene"], t["variant"], m)) for t in ts]
                if not first or None in vals or len(ts) != 3:
                    continue
                best = max(ts, key=lambda t: float(model[t["scene"], t["variant"], m][key]))
                n += 1
                hit += best is first[0]
            if n:
                lines.append(f"    {m:9s} {key:9s}: {hit}/{n} = {100 * hit / n:.0f} %")
    hit = n = 0
    for ts in by_trial.values():
        first = [t for t in ts if t["rank"] == "1"]
        if first and len(ts) == 3:
            n += 1
            hit += max(ts, key=lambda t: float(t["mask_area"])) is first[0]
    if n:
        lines.append(f"    largest object      : {hit}/{n} = {100 * hit / n:.0f} %")

    # time to find, misses counted as the time limit (censored): rank correlation
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for j, m in enumerate(MODELS):
        for i, (key, label) in enumerate([("ratio", "saillance Grad-CAM de l'objet (ratio, échelle log)"),
                                          ("feat_dist", "distance des représentations original/variante (log)")]):
            xs, ys, hit = [], [], []
            for t in trials:
                r = model.get((t["scene"], t["variant"], m))
                if r is None:
                    continue
                xs.append(max(float(r[key]), 1e-6))
                ys.append(float(t["rt_ms"]))
                hit.append(t["found"] == "true")
            if len(xs) < 3:
                continue
            rho, p = spearmanr(xs, ys)
            lines.append(f"  {m:9s} {key:9s} vs time to find (misses = time limit): "
                         f"Spearman rho = {rho:+.2f}, p = {p:.3f}, n = {len(xs)}")
            ax = axes[i, j]
            xs, ys, hit = np.array(xs), np.array(ys), np.array(hit)
            ax.scatter(xs[hit], ys[hit] / 1000, c="tab:green", label="trouvé", s=28)
            ax.scatter(xs[~hit], ys[~hit] / 1000, c="tab:red", marker="x", label="pas trouvé", s=34)
            ax.set_xscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("temps pour trouver (s)")
            ax.set_title(f"{m} — rho = {rho:+.2f} (p = {p:.2f})")
            ax.legend(fontsize=8)
    fig.suptitle("Cécité au changement : temps de détection contre les réseaux\n"
                 "(une corrélation négative = ce que le réseau juge important est trouvé plus vite)")
    fig.tight_layout()
    fig.savefig(OUT / "change_blindness.png", dpi=110)
    plt.close(fig)

    for label, sel in (("main object", "true"), ("other objects", "false")):
        sub = [t for t in trials if t["main_object"] == sel]
        if sub:
            lines.append(f"  {label}: found {sum(t['found'] == 'true' for t in sub)}/{len(sub)}")

    # ---- occlusion ----------------------------------------------------------
    occ = latest_per_subject(args.data.glob("Saliency-Guided-Occlusion_sub-*.csv"))
    otr = [t for _, rs in occ.values() for t in rs if t["phase"] == "main"]
    lines.append("")
    lines.append(f"OCCLUSION: {len(occ)} participant(s), {len(otr)} trials")
    acc = {}
    for c in CONDS:
        sub = [t for t in otr if t["condition"] == c]
        if sub:
            k = sum(t["correct"] == "true" for t in sub)
            cov = np.mean([float(t["main_covered"]) for t in sub])
            acc[c] = (k, len(sub))
            lines.append(f"  {c:9s} {k}/{len(sub)} correct ({100 * k / len(sub):.0f} %), "
                         f"main object hidden on average {100 * cov:.0f} %")
    for t in otr:
        if t["correct"] != "true":
            lines.append(f"  error: {t['scene']} {t['condition']:9s} {t['correct_answer']} -> {t['response']} "
                         f"({100 * float(t['main_covered']):.0f} % of the object hidden)")
    if acc:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [c for c in CONDS if c in acc]
        vals = [100 * acc[c][0] / acc[c][1] for c in names]
        ax.bar(names, vals, color=["0.6", "tab:blue", "tab:orange", "0.3"][:len(names)])
        for x, (c, v) in enumerate(zip(names, vals)):
            ax.text(x, v + 1, f"{acc[c][0]}/{acc[c][1]}", ha="center")
        ax.axhline(25, ls=":", c="k")
        ax.text(len(names) - 0.5, 27, "hasard", ha="right", fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_ylabel("identification correcte (%)")
        ax.set_title("Masquage guidé par les réseaux")
        fig.tight_layout()
        fig.savefig(OUT / "occlusion.png", dpi=110)
        plt.close(fig)

    # ---- maps per photograph -----------------------------------------------
    clicks = defaultdict(list)
    for t in trials:
        if t["found"] == "true":
            clicks[t["scene"]].append((float(t["click_x"]), float(t["click_y"]), True, t["variant"]))
    occ_res = defaultdict(list)
    for t in otr:
        occ_res[t["scene"]].append(f"{t['condition']}: {'OK' if t['correct'] == 'true' else 'FAUX'}")
    scenes = sorted(p.stem for p in (OCC / "saliency/maps/resnet50").glob("*.png"))
    tiles = []
    for s in scenes:
        img = cv2.imread(str(HERE / "build/variants" / s / "original.jpg"))
        panels = [img.copy()]
        for m in MODELS:
            sal = cv2.imread(str(OCC / "saliency/maps" / m / f"{s}.png"), cv2.IMREAD_GRAYSCALE) / 255.0
            panels.append(overlay(img, sal))
        for x, y, hit, v in clicks.get(s, []):
            mask = cv2.imread(str(HERE / "build/variants" / s / f"{v}_mask.png"), cv2.IMREAD_GRAYSCALE) < 128
            cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for p in panels:
                cv2.drawContours(p, cnt, -1, (255, 255, 255), 3)
                cv2.circle(p, (int(x), int(y)), 16, (0, 200, 0) if hit else (0, 0, 255), -1)
                cv2.circle(p, (int(x), int(y)), 16, (255, 255, 255), 3)
        labels = [s, "ResNet-50", "ViT-B/16"]
        for p, lab in zip(panels, labels):
            cv2.putText(p, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
            cv2.putText(p, lab, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
        if occ_res.get(s):
            txt = "masquage " + ", ".join(occ_res[s])
            cv2.putText(panels[0], txt, (14, 740), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7)
            cv2.putText(panels[0], txt, (14, 740), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        tiles.append(cv2.resize(np.hstack(panels), (1200, 300)))
    for k in range(0, len(tiles), 6):
        cv2.imwrite(str(OUT / f"maps_{k // 6:02d}.jpg"), np.vstack(tiles[k:k + 6]), [cv2.IMWRITE_JPEG_QUALITY, 85])

    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
