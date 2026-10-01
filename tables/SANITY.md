# SANITY — P4.02 (régénération des statistiques du paper 4)

Date : 2026-10-01T14:59:26 · B=10000 · bootstrap_seed=20260618

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
| frag 16/19 classes en hausse vs B | ✅ | 16/19 hausses vs B, 19/19 vs contrôle, 14 Holm-significatives ; ne montent pas vs B : bus -0.0, wall -0.0, terrain -1.8 |
| frag seule baisse matérielle = terrain | ✅ | baisses Holm-significatives : ['terrain'] |

Tous les checks sont bloquants : le script sort en erreur au premier échec, aucune table/figure n'est écrite sur une sanité rouge.

Deux familles de tolérance, déclarées : (1) **bit-à-bit** vs P3.16 — mêmes npz, mêmes masques 12 bras, mêmes index bootstrap → la régénération est une reproduction exacte ; (2) **tolérance 1e-4 (0,01 pt)** vs harness P3.10 / master P3.14 — ces tables proviennent d'un AUTRE forward GPU des mêmes checkpoints (non-déterminisme cuDNN/BF16, `deterministic: false`), écart du même ordre que la sanity P3.17 déjà documentée (mIoU consolidé = P3.14 à 4,1e-3 pt). Le primaire pré-enregistré cité dans T2 est la table harness ; les deux sources donnent le même verdict (non significatif).
