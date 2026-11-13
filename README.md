# Change Blindness v2.6 — flicker paradigm, 3 changes, click localisation

A photograph flickers between two versions, separated by blanks (Rensink
flicker paradigm). Between the two versions **three objects change** at once:
each disappears, or changes colour. The participant **clicks directly on every
change** they find; each one found is framed in green. The order in which the
3 changes are found (and the time to each) ranks the objects of the
photograph for attention.

Every photograph comes from **COCO-FreeView** (Chen et al., 2022): 10 observers
viewed it freely for 5 s with an eye tracker, so the participants' change
detection can be compared with real fixations, and both with the saliency maps
of image classifiers (ResNet-50, ViT-B/16).

Built with [goxpyriment](https://github.com/chrplr/goxpyriment) (Go, SDL3).

## Repository layout

| Path | Contents |
|------|----------|
| `main.go`, `main_test.go` | The experiment and its tests |
| `stimuli/` | Photographs, variants, masks and regions (embedded in the binary) |
| `tools/` | Python scripts: stimulus pipeline and analyses |
| `saliency/` | Model outputs: Grad-CAM maps and values, attention maps, representation distances, caption importance |
| `data/` | Participants' data files (`.csv`, one row per changed object) |
| `results/` | Results of the analyses (figures and text reports; summary in `results/RESULTATS.md`) |
| `web/`, `.github/workflows/pages.yml` | Browser version (WebAssembly) and its publication on GitHub Pages |

## Building and running

Go 1.25 or later. SDL is embedded in the binary: nothing else to install.

```bash
go run . -w -s 1           # windowed, subject 1
./build.sh                 # or build.bat on Windows: makes a Change-Blindness binary
```

Data files (`Change-Blindness-v2.6_sub-<id>_date-….csv`) are written to
`goxpy_data/` in the home directory; copy the `.csv` files (not the
`-info.txt` files, which hold the computer's user and host names) to `data/`
to analyse them.


## Stimuli

`stimuli/` holds, per scene, `original.jpg`, the variants as patches
`vK_patch.jpg` (each changes one object; only the rectangle where the variant
differs from the original is stored, at `patch_x`, `patch_y` in the manifest,
which keeps the stimuli at 16 MB instead of 61 MB), the mask of that object `vK_mask.png` (black on white,
used to score the clicks) and `vK_region.png`, the region the program pastes
from the patch onto the original (the mask widened by 25 px, which covers
every pixel the change touched, split with the neighbours where two regions
meet). All are 1024×768. `stimuli/manifest.csv` lists the variants: changed
object (COCO category), whether it is the main object, and the type of change.

They are built from COCO 2017 (val and train) by the scripts in `tools/`:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `select_scenes.py --freeview` | Among the COCO-FreeView photographs, keeps those with one large, central, non-person object and at least 4 other small objects clear of it. Excludes "NoDerivs" licences. |
| 2 | `make_variants.py` | Crops to 4:3, resizes to 1024×768, and makes one variant per object: hue rotation (man-made, saturated objects only), or erasure with LaMa inpainting (small objects only). Outside the changed region a variant is the original, pixel for pixel. |
| 3 | `contact_sheets.py` | Close-ups original / variant of every change, for review. |
| 4 | `export_stimuli.py --freeview` | Drops the variants listed in `tools/rejected.txt` (bad inpainting, food recoloured, etc.), keeps the COCO-FreeView scenes with at least 3 changes left, writes the regions, copies them to `stimuli/`. |

The working files (COCO download, Python environment, LaMa weights,
eye-tracking file) go to `build/`, which git ignores. To rebuild from scratch (Windows paths):

```bash
python -m venv build/venv
build/venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
build/venv/Scripts/python -m pip install numpy pillow opencv-python-headless pycocotools
# COCO annotations_trainval2017.zip and val2017.zip -> build/coco/
# big-lama.pt (github.com/enesmsahin/simple-lama-inpainting releases) -> build/models/
build/venv/Scripts/python tools/select_scenes.py --freeview
build/venv/Scripts/python tools/make_variants.py
build/venv/Scripts/python tools/contact_sheets.py
build/venv/Scripts/python tools/export_stimuli.py --freeview
build/venv/Scripts/python tools/make_patches.py
```

`make_patches.py` replaces each full variant `vK.jpg` by its patch;
`tools/stim.py` rebuilds full variants for the analysis scripts.

`select_scenes.py --freeview` needs the COCO-FreeView fixation file (see
below). It relaxes the selection criteria: COCO-FreeView holds the
photographs of a visual-search study, mostly kitchens and desks. 46 scenes
remain after review, with 3 to 8 changes each (198 variants: 101 erasures,
97 colour changes, 21 of them on the main object).

`stimuli/ATTRIBUTION.csv` gives the source and licence of every photograph
(Creative Commons, mostly non-commercial: fine for research and teaching).

## Comparing with the participants and with eye tracking

| Script | What it compares |
|--------|------------------|
| `compare_with_models.py` | Participants' detection times with the models' saliency of each changed object and with `feat_dist`; within each trial, how often the change found first is the one the model rates highest (chance 33 %). Writes `build/comparison/`. |
| `compare_eyetracking.py` | COCO-FreeView fixations, mapped onto our crops, with the models' maps (CC, NSS, against a centre-bias baseline); object by object, with the models' saliency, `feat_dist` and the participants' detection times (also with the object's area partialled out); within each trial, how often the change found first is the most fixated. Needs the fixation file (about 50 MB) in `build/eyetracking/`: `http://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json`. |

```bash
build/venv/Scripts/python tools/compare_with_models.py --data data
build/venv/Scripts/python tools/compare_eyetracking.py --data data
build/venv/Scripts/python tools/caption_importance.py --data data
build/venv/Scripts/python tools/dino_feat_dist.py
build/venv/Scripts/python tools/vit_attention.py --data data
```

They write to `build/comparison/`; the outputs of the last run are copied in
`results/` (see `results/RESULTATS.md`). `vit_attention.py` compares the
self-attention maps of a supervised ViT-B/16 and of DINO (self-supervised)
and, with `dino_feat_dist.py`, DINO's representation distance.

`caption_importance.py` adds a third human measure: how many of the 5 COCO
captions of each photograph mention each object (needs
`captions_{train,val}2017.json` from `annotations_trainval2017.zip` in
`build/coco/annotations/`).

## Model saliency

`tools/model_saliency.py` computes, for every variant, how salient the
changed object is for two ImageNet classifiers, ResNet-50 (CNN) and ViT-B/16
(transformer), both explained with Grad-CAM for the class they predict on the
original photograph (images resized to 224×224). It needs two more packages:

```bash
build/venv/Scripts/python -m pip install timm grad-cam
build/venv/Scripts/python tools/model_saliency.py
```

It writes `saliency/maps/<model>/<scene>.png` (map scaled 0–255, 1024×768) and
`saliency/model_saliency.csv`, one row per variant and model:

| Column | Description |
|--------|-------------|
| `top1_class`, `top1_prob` | Class predicted on the original, and its probability |
| `mass` | Share of the map's total saliency that falls on the object |
| `ratio` | Mean saliency on the object / mean over the image (1 = average) |
| `peak` | Highest saliency on the object |
| `feat_dist` | 1 − cosine similarity of the penultimate features, original vs variant |
| `js` | Jensen–Shannon divergence of the class probabilities, original vs variant |
| `top1_changed` | The predicted class differs between original and variant |

Join it to the participants' data on `scene` + `variant`. Overlays for viewing
go to `build/saliency_overlays/`.

## Design

- 2 practice scenes, with feedback (changes found framed in green, missed
  ones in red), then **25 scenes** (about 6 minutes) in random order, no
  feedback.
- **The 25 scenes are drawn at random for each participant, weighted by how
  many changes they offer**: weight = changes − 2 (a scene offering 3 changes
  weighs 1, 5 changes 3, 8 changes 6; weighted draw without replacement). All
  44 scenes get used, the richer ones more often:

  | Changes offered | Scenes | Shown in (share of sessions) |
  |---|---|---|
  | 3 | 11 | 33 % |
  | 4 | 15 | 55 % |
  | 5 | 13 | 69 % |
  | 6–8 | 5 | 82–91 % |

  The 2 practice scenes, the same for everyone, are among those offering the
  fewest changes (3). `-n` sets how many scenes are drawn.
- **For each scene, 3 of its variants are drawn at random**, any type, a
  different set for each participant when the scene has more than 3.
  **Big changes** (the main object, or an object covering more than 2.5 % of
  the photograph: 33 of the 198 variants) are found by nearly everyone, first,
  in about a second, so they are drawn with a quarter of the weight of the
  others (`bigWeight`): 9 % of the changes shown instead of 17 %, and at least
  one in 27 % of the photos instead of 46 %. The `main_object` and
  `mask_area` columns identify them in the data.
- Whether the original or the changed version comes first in the cycle is
  random (for an erased object: it disappears or appears).
- Every draw is seeded by the subject ID: the same ID reproduces the session.

## Trial

| Phase | Duration |
|-------|----------|
| Fixation cross | 800 ms |
| Image A | 240 ms |
| Blank (gray) | 80 ms |
| Image A′ | 240 ms |
| Blank (gray) | 80 ms |
| … repeated until the 3 changes are found, or 15 s | |
| Inter-trial blank | 700 ms |

Durations are rounded to whole display frames; the loop is VSYNC-locked and
each time is the mouse event timestamp minus the first image's flip
timestamp. A click within `-tolerance` px of a change not yet found finds it;
other clicks are recorded as false clicks. Both images are re-encoded
losslessly by the same encoder, so outside the 3 changed regions they are
identical to the pixel (checked by `main_test.go`).

## Running

```bash
go run . -w -s 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `-w` | off | Windowed mode |
| `-d N` | -1 | Display index |
| `-s N` | — | Subject ID (selects the changes and seeds the draw) |
| `-n N` | 25 | Scenes per participant |
| `-max MS` | 15000 | Trial timeout |
| `-tolerance PX` | 25 | A click within this distance of a changed object finds it |
| `-practice` | true | Run the practice trials (`-practice=false` to skip) |

## Output

One row **per changed object** (3 per trial) in `goxpy_data/…csv`:

| Column | Description |
|--------|-------------|
| `phase`, `trial` | `practice` or `main`; trial number within the phase |
| `scene`, `changes` | Scene ID (`s` + COCO image ID); the 3 variants shown, separated by `\|` |
| `order` | `original_first` or `changed_first` |
| `variant`, `object` | This row's variant and the COCO category of its object |
| `main_object` | `true` if the change is on the scene's main object |
| `change` | `colour` or `disappear` |
| `found`, `rank` | Whether it was found; 1, 2 or 3 = found first, second, third (0 = not found) |
| `rt_ms` | Time from the first image onset to the click that found it (the timeout if not found) |
| `click_x`, `click_y`, `dist_px` | That click in image pixels, origin top-left, and its distance to the object (−1 if not found) |
| `mask_cx`, `mask_cy`, `mask_area` | Object centroid and area in pixels |
| `n_found` | Changes found in the trial (0–3) |
| `false_clicks` | Clicks away from every change, `x:y:ms` separated by `\|` |
| `trial_ms`, `cycles` | Trial duration; completed A–blank–A′–blank cycles |

`dist_px` is stored so the criterion can be changed at analysis time.

## References

Rensink, R. A., O'Regan, J. K., & Clark, J. J. (1997). To see or not to see:
The need for attention to perceive changes in scenes. *Psychological Science*,
8(5), 368–373.

Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context.
*ECCV 2014*, 740–755.

Chen, Y., Yang, Z., Chakraborty, S., Mondal, S., Ahn, S., Samaras, D.,
Hoai, M., & Zelinsky, G. (2022). Characterizing target-absent human
attention. *CVPR Workshops 2022* (COCO-FreeView eye-tracking data).

Suvorov, R., et al. (2022). Resolution-robust large mask inpainting with
Fourier convolutions (LaMa). *WACV 2022*, 2149–2159.
