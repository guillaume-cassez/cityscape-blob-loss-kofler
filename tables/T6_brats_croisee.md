# T6 — Cross-dataset : le même verdict sur BRATS-2023 (chiffres CITÉS, non re-mesurés ici)

Le bras G existe aussi côté BRATS (region-based/sigmoid, MedNeXt, recette nnU-Net, CV 5 folds, n = 1196) — deuxième dataset, deuxième régime de probabilités. Les chiffres sont cités depuis les artefacts du programme BRATS (vérifiés par grep le 2026-09-30), pas re-mesurés dans ce dépôt.

| Bras BRATS | Résultat | Source exacte |
|---|---|---|
| G blob loss **seul** (vs baseline nnU-Net) | **−0,00376 Dice** (CV 5 folds ; 0/5 folds gagnées — dernier des 4 experts) | BRATS repo `papers/paper3/versions/v3/NOTE_DECISION_V3.md` ligne 46 + `tables/v3_vs_consensus.md` ligne 38 |
| G comme **expert 3** du MoE-V3 gagnant | gate V3 : **+0,00566 Dice** vs sa propre baseline, **1ᵉʳ des 29 bras** (5/5 folds) | NOTE_DECISION_V3.md lignes 52, 115 ; Zenodo DOI 10.5281/zenodo.22903668 (abstract, API 2026-09-30) |

Même conclusion qu'ici : le blob loss seul n'apporte pas de gain global (nul/négatif), mais il contribue comme expert initialisé d'un mélange. Les deux programmes sont indépendants (datasets, architectures, métriques, régimes de probabilités différents) — la convergence des verdicts est l'argument de généralité du papier 4.

Papiers BRATS du programme : MoE « The Gate Does Not Choose » DOI 10.5281/zenodo.22903668 (concept 10.5281/zenodo.22776410) ; kervadec DOI 10.5281/zenodo.22906447 ; consensus DOI 10.5281/zenodo.22904810 ; distmap DOI 10.5281/zenodo.20110976.
