# T4 — Métriques officielles et métier, paires du papier (RÉGÉNÉRÉ depuis les npz par image)

Protocole identique P3.15/P3.16 (mêmes npz, mêmes masques 12 bras, mêmes index bootstrap B=10 000 seed 20260618). `Holm(2)` = Holm sur la famille de CE papier (les 2 paires G_vs_B et G_vs_controle) ; `Holm P3.16` = cross-référence du Holm de la famille exploratoire 15 paires (table P3.16, G_vs_controle). **gras** = Holm(2) < 0,05. Unité : points (×100), sauf `fragments` = composantes connexes par image (count).

| Métrique | n | G | B | ctrl | Δ(G−B) [IC95] | p | Holm(2) | Δ(G−ctrl) [IC95] | p | Holm(2) | Holm P3.16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 500 | 81.26 | 81.09 | 81.17 | +0.173 [-0.229 ; +0.554] | 0.3952 | 0.7904 | +0.094 [-0.281 ; +0.466] | 0.6592 | 0.7904 | 1.000 |
| IoU_person | 500 | 85.22 | 84.88 | 85.15 | **+0.344** [+0.092 ; +0.578] | 0.0088 | 0.0176 | +0.067 [-0.182 ; +0.302] | 0.5838 | 0.5838 | 0.694 |
| IoU_rider | 500 | 67.58 | 68.49 | 68.92 | -0.911 [-2.003 ; +0.036] | 0.0612 | 0.0612 | **-1.339** [-2.413 ; -0.352] | 0.0068 | 0.0136 | 0.095 |
| boundary_f1_3px | 500 | 77.00 | 76.37 | 76.63 | **+0.630** [+0.454 ; +0.806] | 0 | 0 | **+0.376** [+0.216 ; +0.537] | 0 | 0 | 0.000 |
| boundary_f1_3px_pieds | 500 | 67.73 | 67.30 | 67.49 | +0.431 [-0.119 ; +0.953] | 0.1204 | 0.2408 | +0.247 [-0.201 ; +0.702] | 0.2814 | 0.2814 | 1.000 |
| rappel_strict_instances | 441 | 72.84 | 77.17 | 76.25 | **-4.335** [-4.948 ; -3.729] | 0 | 0 | **-3.417** [-3.967 ; -2.868] | 0 | 0 | 0.000 |
| precision_ped_pixels | 473 | 63.27 | 65.96 | 70.03 | **-2.699** [-3.484 ; -1.951] | 0 | 0 | **-6.763** [-7.800 ; -5.785] | 0 | 0 | 0.000 |
| instances_toutes_rappel | 441 | 76.55 | 80.99 | 79.98 | **-4.439** [-5.000 ; -3.884] | 0 | 0 | **-3.423** [-3.911 ; -2.942] | 0 | 0 | 0.000 |
| instances_toutes_det05 | 441 | 85.12 | 88.05 | 87.11 | **-2.936** [-3.817 ; -2.122] | 0 | 0 | **-1.994** [-2.696 ; -1.328] | 0 | 0 | 0.000 |
| instances_individuelles_rappel | 440 | 76.74 | 81.11 | 80.10 | **-4.376** [-4.935 ; -3.823] | 0 | 0 | **-3.361** [-3.853 ; -2.866] | 0 | 0 | 0.000 |
| instances_individuelles_det05 | 440 | 85.17 | 88.08 | 87.18 | **-2.903** [-3.819 ; -2.069] | 0 | 0 | **-2.002** [-2.712 ; -1.302] | 0 | 0 | 0.000 |
| instances_foule_rappel | 63 | 69.97 | 75.71 | 74.58 | **-5.736** [-7.864 ; -4.033] | 0 | 0 | **-4.612** [-6.531 ; -2.940] | 0 | 0 | 0.000 |
| instances_foule_det05 | 63 | 82.01 | 86.24 | 83.60 | **-4.233** [-7.937 ; -1.058] | 0.0046 | 0.0092 | -1.587 [-4.233 ; +0.529] | 0.225 | 0.225 | 1.000 |
| instances_taille_T1_rappel | 315 | 60.91 | 66.56 | 64.99 | **-5.645** [-6.650 ; -4.627] | 0 | 0 | **-4.080** [-5.092 ; -3.091] | 0 | 0 | 0.000 |
| instances_taille_T1_det05 | 315 | 68.26 | 72.84 | 71.73 | **-4.574** [-6.218 ; -2.890] | 0 | 0 | **-3.466** [-5.017 ; -1.960] | 0 | 0 | 0.000 |
| instances_taille_T2_rappel | 320 | 80.88 | 85.46 | 84.53 | **-4.584** [-5.316 ; -3.880] | 0 | 0 | **-3.656** [-4.370 ; -2.938] | 0 | 0 | 0.000 |
| instances_taille_T2_det05 | 320 | 91.08 | 93.78 | 93.15 | **-2.695** [-3.893 ; -1.609] | 0 | 0 | **-2.064** [-3.196 ; -0.928] | 0.0004 | 0.0004 | 0.006 |
| instances_taille_T3_rappel | 335 | 91.68 | 93.85 | 93.47 | **-2.175** [-2.478 ; -1.875] | 0 | 0 | **-1.798** [-2.090 ; -1.510] | 0 | 0 | 0.000 |
| instances_taille_T3_det05 | 335 | 98.78 | 98.96 | 98.96 | -0.180 [-0.697 ; +0.238] | 0.4654 | 0.4654 | -0.177 [-0.468 ; +0.054] | 0.1586 | 0.3172 | 1.000 |
| fragments | 500 | 933.88 | 615.13 | 519.26 | **+318.8** [+305.4 ; +331.9] | 0 | 0 | **+414.6** [+397.8 ; +431.1] | 0 | 0 | — |

## Fragmentation par classe — composantes connexes 8-connexes par image (diagnostic neuf, G vs B)

Mesure inédite du programme (P3.16 n'avait les fragments que pour contrôle/MoE). Holm sur la famille des 19 classes. **gras** = Holm < 0,05.

| Classe | G | B | ctrl | Δ(G−B) | IC95 | p | Holm(19) |
|---|---|---|---|---|---|---|---|
| car | 117.3 | 67.9 | 57.4 | **+49.4** [+46.5 ; +52.2] | 0 | 0 |
| pole | 176.4 | 131.1 | 115.9 | **+45.4** [+42.9 ; +47.8] | 0 | 0 |
| vegetation | 107.7 | 63.8 | 58.8 | **+43.8** [+41.6 ; +46.1] | 0 | 0 |
| road | 82.4 | 43.8 | 47.1 | **+38.6** [+36.3 ; +40.9] | 0 | 0 |
| traffic sign | 92.3 | 53.9 | 43.2 | **+38.4** [+36.1 ; +40.7] | 0 | 0 |
| building | 116.9 | 84.4 | 75.4 | **+32.5** [+29.6 ; +35.4] | 0 | 0 |
| sidewalk | 106.2 | 75.4 | 64.6 | **+30.8** [+28.7 ; +33.0] | 0 | 0 |
| person | 53.2 | 24.3 | 16.4 | **+28.9** [+27.3 ; +30.5] | 0 | 0 |
| sky | 27.7 | 23.2 | 11.9 | **+4.6** [+3.2 ; +5.9] | 0 | 0 |
| fence | 10.9 | 8.2 | 4.9 | **+2.7** [+2.1 ; +3.4] | 0 | 0 |
| bicycle | 11.8 | 9.6 | 6.2 | **+2.2** [+1.7 ; +2.7] | 0 | 0 |
| rider | 4.1 | 2.0 | 1.9 | **+2.2** [+2.0 ; +2.4] | 0 | 0 |
| traffic light | 6.9 | 6.0 | 4.6 | **+0.9** [+0.6 ; +1.1] | 0 | 0 |
| truck | 0.6 | 0.3 | 0.3 | **+0.2** [+0.1 ; +0.4] | 0 | 0 |
| motorcycle | 0.7 | 0.6 | 0.6 | +0.0 [-0.0 ; +0.1] | 0.494 | 1 |
| train | 0.3 | 0.2 | 0.2 | +0.0 [-0.0 ; +0.0] | 0.3292 | 1 |
| bus | 0.5 | 0.5 | 0.4 | -0.0 [-0.1 ; +0.1] | 0.7028 | 1 |
| wall | 7.2 | 7.2 | 4.0 | -0.0 [-0.5 ; +0.5] | 0.9496 | 1 |
| terrain | 10.9 | 12.8 | 5.5 | **-1.8** [-2.6 ; -1.1] | 0 | 0 |

Lecture (thèse du papier) : le terme blob **achète** l'IoU pixel des objets fins (traffic light, pole, bicycle, person, truck — T3) et un BF1 3px supérieur, et le **paie** en rappel instance piéton (strict, T1, foule) et en précision pixel piéton. Les deux paires racontent la même histoire ; la paire appariée en budget (G−B) est celle du primaire.

Lecture du diagnostic de fragmentation (mesure neuve, 500 images × 3 seeds) : le terme blob ne **compacte pas** les prédictions — il les **fragmente**. Les composantes connexes augmentent dans 16 classes sur 19 vs B apparié (14 hausses Holm-significatives) et dans 19 sur 19 vs le bras contrôle. Classes qui ne montent pas vs B : bus -0.0, wall -0.0, terrain -1.8 — seule terrain est une baisse matérielle (Holm < 0,001), les autres sont nulles (p ≥ 0,70). Grandes classes mouchetées (car +49,4, pole +45,4, vegetation +43,8, road +38,6 composantes/image) et masques piétons morcelés (person 24,3 → 53,2, ×2,2), cohérent avec les pixels perdus à l'intérieur des instances GT (rappel strict −4,3 ; petits T1 −5,6 — un masque troué se coupe en composantes disjointes). Les classes fines gagnent aussi des composantes (traffic light +0,9, bicycle +2,2, Holm-significatifs) tout en gagnant de l'IoU pixel : le terme achète de la couverture pixel des objets fins, pas leur intégrité topologique. Total : 933,9 composantes/image (G) contre 615,1 (B), ×1,52.

Note mIoU : cette ligne vient du forward des npz P3.16 ; le primaire pré-enregistré (T2) vient du forward harness P3.10 — écart ≤ 0,001 pt (non-déterminisme cuDNN/BF16), verdict ns identique, écart mesuré dans SANITY.md. Toutes les autres lignes sont exactement la source P3.16/P3.17.

Provenance : npz `results/moe_v3_cs/metiers_experts/metriques_*_seed*.npz` + fragments `frag_{G,B,controle}_seed*.npy` et décomposition par classe `results/moe_v3_cs/paper4_blob/frag_classes_*_seed*.npy` (G/B calculés ce jour depuis les preds /mnt/data8t/cityscape_p316, contrôle depuis results/moe_v3_cs/metiers) ; bootstraps bruts `results/moe_v3_cs/paper4_blob/table_metiers_G_pairees.json` + `table_frag_classes_G_vs_B.json`.
