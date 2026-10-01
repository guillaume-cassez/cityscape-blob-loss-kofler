# SANITY — P4.02 (régénération des statistiques du paper 4)

Date : 2026-10-01T16:56:20 · B=10000 · bootstrap_seed=20260618

| Check | Résultat | Détail |
|---|---|---|
| frag_classes somme == frag npy (G) | ✅ | (G) somme des 19 classes bit-à-bit égale au count total (mean 933.9/image) |
| frag_classes somme == frag npy (B) | ✅ | (B) somme des 19 classes bit-à-bit égale au count total (mean 615.1/image) |
| frag_classes somme == frag npy (controle) | ✅ | (controle) somme des 19 classes bit-à-bit égale au count total (mean 519.3/image) |
| P3.16 G_vs_controle reproduit | ✅ | 19 métriques bit-à-bit (delta/IC/p, mêmes index bootstrap) |
| harness P3.10 points G/B (tolérance 1e-4, 2 forwards) | ✅ | npz G=0.8126227 vs harness 0.8126326 (écart 9.9e-06) ; B écart 4.1e-05 — non-déterminisme forward, cf P3.17 sanity 4,1e-3 pt |
| harness P3.10 primaire G−B (tolérance) | ✅ | npz Δ=+0.1733pt p=0.3952 vs harness Δ=+0.1702pt p=0.4032 — même verdict (ns) |
| P3.14 master G_vs_controle (tolérance) | ✅ | npz Δ=+0.0943pt p=0.6592 vs P3.14 Δ=+0.0950pt p=0.6552 |
| mIoU seed 42 G vs harness (tolérance) | ✅ | npz 0.8129533 vs harness 0.8129699 (écart 1.7e-05) |
| mIoU seed 123 G vs harness (tolérance) | ✅ | npz 0.8128984 vs harness 0.8129079 (écart 9.5e-06) |
| mIoU seed 456 G vs harness (tolérance) | ✅ | npz 0.8120164 vs harness 0.8120201 (écart 3.7e-06) |
| fragments controle = P3.16 (mêmes npy) | ✅ | controle=519.26 vs P3.16 519.26 |
| Holm P3.14 recomputé == stocké (A_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (B_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (C_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (Cp_vs_controle) | ✅ | 0.4396 vs 0.4396 |
| Holm P3.14 recomputé == stocké (D_vs_controle) | ✅ | 0.0576 vs 0.0576 |
| Holm P3.14 recomputé == stocké (Dp_vs_controle) | ✅ | 0.2808 vs 0.2808 |
| Holm P3.14 recomputé == stocké (fused_CpvetoB_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (fused_CvetoB_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (fused_DpvetoB_vs_controle) | ✅ | 0.3168 vs 0.3168 |
| Holm P3.14 recomputé == stocké (fused_DvetoB_vs_controle) | ✅ | 0.082 vs 0.082 |
| Holm P3.14 recomputé == stocké (G_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.14 recomputé == stocké (moe_v3cs_vs_controle) | ✅ | 0.0726 vs 0.0726 |
| Holm P3.16 recomputé == stocké (B_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (C_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (Cp_vs_controle) | ✅ | 0.618 vs 0.618 |
| Holm P3.16 recomputé == stocké (D_vs_controle) | ✅ | 0.075 vs 0.075 |
| Holm P3.16 recomputé == stocké (Dp_vs_controle) | ✅ | 0.396 vs 0.396 |
| Holm P3.16 recomputé == stocké (G_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (fused_CvetoB_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (fused_DvetoB_vs_controle) | ✅ | 0.1066 vs 0.1066 |
| Holm P3.16 recomputé == stocké (fused_CpvetoB_vs_controle) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (fused_DpvetoB_vs_controle) | ✅ | 0.4136 vs 0.4136 |
| Holm P3.16 recomputé == stocké (fused_CvetoB_vs_C) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (fused_DvetoB_vs_D) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (fused_CpvetoB_vs_Cp) | ✅ | 0.7668 vs 0.7668 |
| Holm P3.16 recomputé == stocké (fused_DpvetoB_vs_Dp) | ✅ | 1 vs 1 |
| Holm P3.16 recomputé == stocké (moe_v3cs_vs_controle) | ✅ | 0.0924 vs 0.0924 |
| famille P3.14 = 12 paires (12 bras vs contrôle) | ✅ | len(pairwise) = 12 |
| famille P3.16 = 15 paires (12 vs contrôle + 3 fusion-vs-expert) | ✅ | len(pairwise mIoU) = 15, len(pairs) = 15 |
| Holm mIoU polyvalence == famille 15 paires (B) | ✅ | 1 vs 1 |
| Holm mIoU polyvalence == famille 15 paires (C) | ✅ | 1 vs 1 |
| Holm mIoU polyvalence == famille 15 paires (Cp) | ✅ | 0.618 vs 0.618 |
| Holm mIoU polyvalence == famille 15 paires (D) | ✅ | 0.075 vs 0.075 |
| Holm mIoU polyvalence == famille 15 paires (Dp) | ✅ | 0.396 vs 0.396 |
| Holm mIoU polyvalence == famille 15 paires (fused_CpvetoB) | ✅ | 1 vs 1 |
| Holm mIoU polyvalence == famille 15 paires (fused_CvetoB) | ✅ | 1 vs 1 |
| Holm mIoU polyvalence == famille 15 paires (fused_DpvetoB) | ✅ | 0.4136 vs 0.4136 |
| Holm mIoU polyvalence == famille 15 paires (fused_DvetoB) | ✅ | 0.1066 vs 0.1066 |
| Holm mIoU polyvalence == famille 15 paires (G) | ✅ | 1 vs 1 |
| Holm mIoU polyvalence == famille 15 paires (moe_v3cs) | ✅ | 0.0924 vs 0.0924 |
| classement 13 bras trié par Δ décroissant | ✅ | 13 bras, Δ de +0.52 à -0.28 pt |
| rang G contrôle inclus == rang G décalé de la position du contrôle | ✅ | rang_G=7/12, rang_G_13=7/13, rang_contrôle=10 |
| frag 16/19 classes en hausse vs B | ✅ | 16/19 hausses vs B, 19/19 vs contrôle, 14 Holm-significatives ; ne montent pas vs B : bus -0.0, wall -0.0, terrain -1.8 |
| frag seule baisse matérielle = terrain | ✅ | baisses Holm-significatives : ['terrain'] |

Tous les checks sont bloquants : le script sort en erreur au premier échec, aucune table/figure n'est écrite sur une sanité rouge.

Deux familles de tolérance, déclarées : (1) **bit-à-bit** vs P3.16 — mêmes npz, mêmes masques 12 bras, mêmes index bootstrap → la régénération est une reproduction exacte ; (2) **tolérance 1e-4 (0,01 pt)** vs harness P3.10 / master P3.14 — ces tables proviennent d'un AUTRE forward GPU des mêmes checkpoints (non-déterminisme cuDNN/BF16, `deterministic: false`), écart du même ordre que la sanity P3.17 déjà documentée (mIoU consolidé = P3.14 à 4,1e-3 pt). Le primaire pré-enregistré cité dans T2 est la table harness ; les deux sources donnent le même verdict (non significatif).
