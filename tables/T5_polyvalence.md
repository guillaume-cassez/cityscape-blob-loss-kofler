# T5 — Polyvalence (P3.17) : position de G sur 36 endpoints × 13 bras

36 endpoints (19 IoU par classe + 17 métriques métier) × 13 bras, holdout first:500, bootstrap apparié B=10 000 + Holm. `dommage max` = pire Δ d'un endpoint vs contrôle (plus c'est proche de 0, plus le bras est sans point faible). Critère T0 pré-enregistré : ΔmIoU significatif ET aucun endpoint >1 pt sous la référence ET aucune classe dégradée >0,5 pt.

| # | Bras | percentile moyen | pire rang | #1 | dommage max (pt) | z | pic max (pt) | ΔmIoU (p / Holm) | gains sig (brut/Holm) | pertes sig (brut/Holm) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MoE-V3-CS (4 experts) | 60.4 | 11 | 0 | **-0.53** (instances_foule_det05) | -0.35 | +4.24 (IoU_truck) | +0.45 (0.0066 / 0.092) | 7/1 | 2/1 | non dominé |
| 2 | consensus C⊘B | 44.0 | 13 | 0 | **-1.90** (IoU_bus) | -1.47 | +3.44 (IoU_truck) | +0.07 (0.7866 / 1.000) | 2/1 | 6/3 | dominé |
| 3 | consensus Cp⊘B | 35.7 | 13 | 0 | **-2.78** (IoU_bus) | -2.16 | +2.65 (instances_foule_det05) | -0.15 (0.3532 / 1.000) | 0/0 | 7/2 | dominé |
| 4 | C · CE+Dice+EDT | 47.5 | 13 | 0 | **-3.21** (precision_ped_pixels) | -0.50 | +3.63 (IoU_truck) | +0.06 (0.8182 / 1.000) | 2/0 | 4/3 | dominé |
| 5 | B · CE+Dice | 51.2 | 13 | 2 | **-4.06** (precision_ped_pixels) | -0.63 | +2.65 (instances_foule_det05) | -0.08 (0.6592 / 1.000) | 11/8 | 6/6 | dominé |
| 6 | consensus Dp⊘B | 69.0 | 12 | 4 | **-4.77** (precision_ped_pixels) | -0.74 | +4.38 (IoU_truck) | +0.48 (0.0376 / 0.414) | 11/6 | 5/4 | non dominé |
| 7 | Cp · CE+Dice+SDT | 34.5 | 13 | 0 | **-6.05** (precision_ped_pixels) | -0.94 | +2.65 (instances_foule_det05) | -0.28 (0.0618 / 0.618) | 0/0 | 6/3 | dominé |
| 8 | G · CE+Dice+Blob ← **G (ce papier)** | 41.9 | 13 | 7 | **-6.76** (precision_ped_pixels) | -1.05 | +3.43 (IoU_truck) | +0.09 (0.6592 / 1.000) | 4/1 | 13/12 | dominé |
| 9 | consensus D⊘B | 71.9 | 12 | 3 | **-8.70** (precision_ped_pixels) | -1.35 | +4.94 (IoU_wall) | +0.49 (0.0082 / 0.107) | 14/9 | 5/4 | non dominé |
| 10 | Dp · CE+SDT | 72.7 | 13 | 3 | **-15.02** (precision_ped_pixels) | -2.32 | +5.19 (IoU_truck) | +0.47 (0.0330 / 0.396) | 16/11 | 4/3 | non dominé |
| 11 | D · CE+EDT | 77.3 | 13 | 16 | **-22.20** (precision_ped_pixels) | -3.44 | +5.49 (IoU_wall) | +0.52 (0.0050 / 0.075) | 17/13 | 4/3 | non dominé |

## Profil G (bras de ce papier)

- percentile moyen **41.9** · rang moyen 8.2 · pire rang **13/13** · dernier sur 16/36 endpoints
- **#1 sur 7/36 endpoints** (spécialiste : c'est plus que le MoE, 0) — mais dommage maximal **-6.76 pt** (precision_ped_pixels, z = -1.05 σ)
- prix consenti par point de mIoU gagné : **-71.7 pt** de dommage (contre −1,2 pour le MoE-V3-CS)
- gains significatifs (brut) : IoU_pole, IoU_traffic light, IoU_sky, boundary_f1_3px ; pertes : 13 endpoints dont 12 survivent à Holm
- **dominé** au sens de Pareto par : B, C, C⊘B, Dp⊘B, contrôle, MoE-V3-CS
- familles : meilleure = contours (+0.31 pt moyen), pire = **piéton** (-5.09 pt)

## Verdict T0 (pré-enregistré) : True

Lecture pour le papier : G est l'anti-thèse du critère T0 — un spécialiste dominé dont la spécialité (objets fins) est précisément ce que le MoE-V3-CS absorbe sans hériter des dégâts (papier 3). Sa place dans le classement est donnée telle quelle, sans re-classement.

Provenance : `results/moe_v3_cs/p317_polyvalence/table_polyvalence.json` (P3.17, 2026-09-29).
