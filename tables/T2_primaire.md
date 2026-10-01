# T2 — Endpoint primaire pré-enregistré : G vs B apparié (160 époques, seule la loss change)

Protocole : mIoU dataset-level (cityscapesScripts, 19 classes), holdout first:500, seeds 42/123/456 moyennés dans chaque réplicat, bootstrap apparié B=10 000 (seed 20260618), p bilatéral, paire unique → Holm = p.

| Quantité | G (CE+Dice+0,5·blob) | B (CE+Dice) |
|---|---|---|
| mIoU point | **81.263** | **81.093** |
| IC95 | [79.815 ; 82.431] | [79.694 ; 82.204] |
| mIoU seed 42 | 81.297 | 81.429 |
| mIoU seed 123 | 81.291 | 80.861 |
| mIoU seed 456 | 81.202 | 80.989 |

**Δ(G−B) = +0.173 pt · IC95 [-0.229 ; +0.554] · p = 0.3952 (Holm paire unique = 0.7904) → NON SIGNIFICATIF.**

Lecture honnête : l'endpoint primaire est **nul**. Le blob loss n'améliore pas le mIoU en segmentation sémantique exclusive pleine résolution. La contribution du papier est le diagnostic du trade-off (T3, T4) et la position de polyvalence (T5).

## Contexte 13 bras (référence contrôle 80 époques — NON apparié en budget, donné pour le plateau)

| # | Bras | mIoU | Δ vs contrôle | p | Holm (famille 12 paires, P3.14) | Holm (famille 15 paires, P3.16) |
|---|---|---|---|---|---|---|
| 1 | D · CE+EDT | 81.69 | +0.52 [+0.14, +0.91] | 0.0048 | 0.058 | 0.075 |
| 2 | consensus D⊘B | 81.65 | +0.48 [+0.11, +0.87] | 0.0082 | 0.082 | 0.107 |
| 3 | consensus Dp⊘B | 81.65 | +0.48 [+0.02, +0.96] | 0.0396 | 0.317 | 0.414 |
| 4 | Dp · CE+SDT | 81.64 | +0.47 [+0.04, +0.93] | 0.0312 | 0.281 | 0.396 |
| 5 | MoE-V3-CS (4 experts) | 81.62 | +0.45 [+0.11, +0.82] | 0.0066 | 0.073 | 0.092 |
| 6 | A · CE seule | 81.28 | +0.11 [-0.27, +0.52] | 0.5722 | 1.000 | n.c. (a) |
| 7 | G · CE+Dice+Blob ← **G (ce papier)** | 81.26 | +0.10 [-0.28, +0.47] | 0.6552 | 1.000 | 1.000 |
| 8 | consensus C⊘B | 81.24 | +0.07 [-0.34, +0.47] | 0.7758 | 1.000 | 1.000 |
| 9 | C · CE+Dice+EDT | 81.23 | +0.06 [-0.34, +0.47] | 0.8096 | 1.000 | 1.000 |
| 10 | B · CE+Dice | 81.09 | -0.08 [-0.45, +0.28] | 0.6748 | 1.000 | 1.000 |
| 11 | consensus Cp⊘B | 81.01 | -0.15 [-0.52, +0.17] | 0.3498 | 1.000 | 1.000 |
| 12 | Cp · CE+Dice+SDT | 80.89 | -0.28 [-0.60, +0.02] | 0.0628 | 0.440 | 0.618 |
| — | contrôle (réf) | 81.17 | — | — | — | — |

(a) A est hors de la famille 15 paires (couverture partielle, `annexe_couverture_partielle` de P3.17 = ['A']) : son Holm n'y est pas calculable.

**Aucun bras ne passe Holm 0,05 dans AUCUNE des deux familles** (meilleur 0.0576 sur 12 paires, 0.0750 sur 15 paires) ; G est 7ᵉ/12 par amplitude (7ᵉ/13 contrôle inclus — le contrôle, Δ = 0, se classe 10ᵉ ; table P3.14).

Les deux colonnes sont des familles de multiplicité DIFFÉRENTES et les deux sont données : 12 paires = les 12 bras contre le contrôle (`p314/master_table.json`), 15 paires = la famille exploratoire du programme, qui ajoute les 4 comparaisons fusion-contre-expert (`metiers_experts/table_metiers_experts.json`, reprise par P3.17). Tailles calculées depuis les artefacts, Holm recomputé depuis les p bruts et comparé aux valeurs stockées (sanity bloquante) — jamais recopié d'une étiquette.

Provenance : `results/moe_v3_cs/harness/table_Gseul_P310.json` (primaire cité ci-dessus) et `results/moe_v3_cs/p314/master_table.json` (contexte). La ligne mIoU de T4 est régénérée depuis les npz P3.16 (autre forward GPU des mêmes checkpoints) : écart ≤ 0,001 pt avec le harness (non-déterminisme cuDNN/BF16, `deterministic: false` ; même ordre que la sanity P3.17, 4,1e-3 pt) — mesuré dans SANITY.md, le verdict ns est identique dans les deux sources.
