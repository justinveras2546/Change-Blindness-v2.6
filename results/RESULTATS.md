# Résultats

Analyses du 12 novembre 2026, sur **une session pilote** (sujet 997, 40 photos,
3 changements par photo, soit 120 changements) et sur l'oculométrie de
COCO-FreeView (10 observateurs par photo, 5 s de regard libre). Avec un seul
participant, ce sont des pistes à confirmer, pas des conclusions.

Les fichiers de ce dossier sont les sorties des scripts de `tools/` :

| Fichier | Script | Contenu |
|---------|--------|---------|
| `summary.txt`, `change_blindness.png` | `compare_with_models.py` | Détection contre saillance Grad-CAM et distance des représentations |
| `eyetracking.txt`, `eyetracking.png`, `eyetracking_maps.jpg` | `compare_eyetracking.py` | Regard humain contre cartes des réseaux et détection |
| `captions.txt`, `captions.png` | `caption_importance.py` | Objets cités dans les descriptions COCO |
| `attention.txt`, `attention_maps.jpg` | `vit_attention.py` (+ `dino_feat_dist.py`) | Cartes d'attention des transformers (ViT supervisé, DINO) |

## Détection des changements

- 101 changements trouvés sur 120 (84 %), 4 clics à côté.
- Temps médians : 1,4 s pour le 1er changement trouvé, 3,5 s pour le 2e, 7,3 s pour le 3e.
- L'objet principal est trouvé à chaque fois (11/11), en premier 10 fois.

## Quel changement est trouvé en premier ?

Pour chaque photo, la mesure prédit-elle le changement trouvé en premier ?
Le hasard est à 33 %.

| Mesure | Bonne prédiction | Corrélation avec le temps (taille de l'objet contrôlée) |
|--------|-----------------|------------------|
| Distance des représentations, ViT supervisé | 60 % | −0,45 |
| Distance des représentations, DINO | 60 % | −0,49 |
| Distance des représentations, ResNet-50 | 58 % | −0,50 |
| Objet le plus regardé (COCO-FreeView) | 52 % | −0,26 |
| Plus gros objet | 48 % | — |
| Attention [CLS], ViT supervisé | 45 % | −0,16 |
| Attention [CLS], DINO | 35 % | −0,28 |
| Grad-CAM, ResNet-50 | 35 % | −0,13 |
| Grad-CAM, ViT-B/16 | 32 % | +0,02 |

La distance des représentations (de combien le changement modifie le résumé
interne de la photo par le réseau) prédit nettement ce qui est remarqué, et
garde son effet sur les objets du fond seuls (109 changements : ρ ≈ −0,42 à
−0,51, taille contrôlée). Les cartes (Grad-CAM, attention) prédisent mal.

## Où regardent les humains (COCO-FreeView, 46 photos)

| Carte | CC | NSS |
|-------|----|-----|
| Attention [CLS], DINO | 0,47 | 1,03 |
| Biais central (gaussienne au centre) | 0,42 | 0,85 |
| Grad-CAM, ResNet-50 | 0,29 | 0,56 |
| Grad-CAM, ViT-B/16 | 0,06 | 0,08 |
| Attention [CLS], ViT supervisé | 0,01 | 0,03 |

Seule l'attention de DINO (entraîné sans étiquettes) fait mieux que le biais
central. Le ViT supervisé, de même architecture, ne ressemble pas au regard
humain : son attention se concentre sur quelques zones vides (jetons
« registres », Darcet et al., 2024).

## Liens entre les mesures humaines

- Objets plus regardés en oculométrie → changements trouvés plus vite
  (ρ = −0,41 ; −0,26 taille contrôlée).
- Objets cités dans les descriptions COCO → plus regardés (ρ = +0,45) et
  trouvés plus vite (ρ ≈ −0,3).

## Limites

- Un seul participant, qui connaît l'expérience.
- L'objet principal est gros, central, trouvé en premier, et perturbe beaucoup
  les réseaux : les analyses doivent être refaites sur les objets du fond seuls
  et avec la taille en covariable (modèles mixtes une fois qu'il y aura
  plusieurs participants).
- Les photos COCO-FreeView sont surtout des intérieurs (bureaux, cuisines).
