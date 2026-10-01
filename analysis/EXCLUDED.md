# Artefacts volontairement NON embarqués

Le manuscrit le déclare (§8, limite 10) : les artefacts lourds ne sont pas distribués.
Rien n'est tu — chaque exclusion est nommée, avec sa raison et son chemin de
régénération. Tout ce qui est nécessaire pour re-dériver les statistiques publiées
(tableaux par image, matrices de confusion, tables de bootstrap) EST embarqué dans
`analysis/`.

| Artefact | Nature | Pourquoi il n'est pas ici | Comment le régénérer |
|---|---|---|---|
| `results/moe_v3_cs/metiers/pred_moe_v3cs_seed{42,123,456}.npy` | cartes de prédiction plein cadre 500×1024×2048 uint8 | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/pred_controle_seed{42,123,456}.npy` | idem, bras contrôle | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/gt_trainids.npy` | ground-truth train ids du holdout | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers/gt_instances_ped.npz` | instances piétonnes officielles | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers_experts/metriques_*_seed*.npz (36 fichiers)` | métriques métier par image des 12 bras — le bras G et sa paire B sont embarqués | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `results/moe_v3_cs/metiers_experts/frag_*_seed*.npz` | fragments par classe des autres bras | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `checkpoints/pilot_fullres_G_blob_seed{42,123,456}/epoch_160.pth` | poids d'entraînement (31,5 Go de VRAM, ~87 h GPU pour les 3 seeds) | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |
| `paquetages .blob.npz (2 975 fichiers)` | packs d'instances précalculés — parité bit-à-bit vérifiée sur les 2 975 le 2026-09-26 | volume (≥ 1 Go par carte de prédiction plein cadre) | scripts publiés dans `scripts/`, à partir de Cityscapes + des checkpoints |

## Ce qui EST embarqué et suffit à re-dériver chaque nombre publié

- `analysis/harness/table_Gseul_P310.json` — le critère primaire pré-enregistré
  (Δ(G−B), IC95, p bilatéral, bootstrap apparié par image).
- `analysis/harness/*.cm.npy` — matrices de confusion par seed (G et contrôle),
  d'où la mIoU officielle est re-calculable.
- `analysis/paper4_blob/` — fragmentation par classe (G, B, contrôle × 3 seeds) et
  métriques métier appariées du bras G.
- `analysis/p314/master_table.json` — la table maîtresse 13 bras / 36 endpoints.
- `analysis/p316/attribution_perclass_vs_B.json` — attribution per-class appariée.
- `analysis/p317_polyvalence/table_polyvalence.json` — classement de polyvalence.
- `tables/paper4_tables.json` — la table de consolidation T1-T6 du papier.

Le test de cohérence `scripts/p4_check_numbers.py` (50 checks) relie chaque nombre du
manuscrit à ces artefacts et échoue à la moindre dérive.

## Licence

Les manuscrits (`paper.md`, `paper_fr.md`, `paper.pdf`, `paper_fr.pdf`), les tableaux
et les figures sont sous **CC-BY-4.0** (c'est la licence du dépôt Zenodo). Le code
(`src/`, `scripts/`, `tests/`, `configs/`) est sous **MIT** (`LICENSE`). Les données
d'entraînement appartiennent au jeu Cityscapes et restent soumises à sa propre licence.
