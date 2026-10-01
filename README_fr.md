# Le poids égal par instance ne paie pas seul

**Un auxiliaire blob loss en pleine résolution Cityscapes achète l'IoU des objets fins avec le
rappel piéton, et ne paie que comme expert d'un mélange d'experts.**

Guillaume Cassez — chercheur indépendant · ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) ·
[guillaume-cassez.fr](https://guillaume-cassez.fr/voiture-autonome/)

Preprint, CC-BY-4.0. À lire : [`paper_fr.pdf`](paper_fr.pdf) (FR, 15 p.) ou
[`paper.pdf`](paper.pdf) (EN, 15 p.).

---

## De quoi il s'agit

Évaluation contrôlée et pré-enregistrée de la **blob loss** de Kofler *et al.* (IPMI 2023) — un
terme auxiliaire qui donne à chaque instance de la vérité terrain le même poids quel que soit
son nombre de pixels — portée à la **segmentation multi-classes exclusive en résolution
Cityscapes native** (1024×2048, 19 classes, softmax). Le bras **G** (CE + Dice + 0,5·blob) est
comparé à sa référence **B** exactement appariée (CE + Dice) : mêmes ConvNeXt-V2-Base + UPerNet,
même recette de 160 époques, même augmentation, trois seeds partagés (42, 123, 456) — **seule la
loss change**.

## Le résultat principal est nul, et c'est l'objet du papier

| | valeur |
|---|---|
| **Critère primaire** (pré-enregistré) | mIoU officielle au niveau dataset, G vs B apparié |
| Δ(G−B) | **+0.170 pt** |
| IC95 | [-0.233 ; +0.551] |
| p bilatéral, bootstrap apparié par image | **0.4032** |
| bootstrap | B = 10,000 réplicats, seed 20260618, holdout first:500 |
| mIoU (G / B) | 81.26 / 81.09 |
| robustesse (2ᵉ forward indépendant) | Δ +0.173 pt, p = 0.3952 — même verdict, écart ≤ 0,01 pt |

Le poids égal par instance **n'améliore pas** la mIoU globale dans ce régime. Le papier rapporte
ce que le terme fait à la place : un **arbitrage** mesuré.

**Acheté** — couverture pixel des objets fins : traffic light +0,98, pole +0,58, bicycle +0,57
IoU (significatifs sous Holm dans la famille des 19 classes), truck +3,78 (IC excluant 0),
Boundary F1 (3 px) +0,63.

**Payé** — intégrité des instances piétonnes : rappel piéton strict −4,34, plus petit tercile
(T1) −5,65, instances en groupe −5,74, précision pixel piéton −2,70 (tous vs B apparié,
Holm = 0).

**Et un mécanisme que les tables contredisent.** Les composantes connexes par image passent de
615.1 à 933.9 (**×1.52**), en hausse dans
**16 classes sur 19** contre le bras B apparié (dont 14
Holm-significatives) et dans **19 sur 19** contre le bras contrôle ;
masques person ×2.2. Seule baisse matérielle : terrain
(-1.8, Holm < 0,001) — wall et bus sont nulles (p ≥ 0,70). L'hypothèse de
travail du programme — qu'un terme équilibrant par instance *compacte* les masques en élaguant
les petites composantes — est **explicitement réfutée** par cette décomposition par classe
(§6.4) : le terme crée des trous, il n'élague pas les fragments.

**Polyvalence.** Sur le critère pré-enregistré du programme (36 endpoints × 13 bras), G est un
**spécialiste dominé** : percentile moyen 41.9, pire rang 13/13,
dommage maximal -6.76 pt, 13 pertes significatives contre 4 gains
significatifs.

**Là où il paie.** Comme *expert initialisé d'un mélange*. Sur BRATS 2023 (sigmoïde, par
région) le bras blob seul valait −0,00376 Dice, mais comme expert 3 du MoE-V3 gagnant il a
contribué +0,00566 Dice, premier sur 29 bras
([DOI 10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668)). Le MoE-V3-CS
Cityscapes absorbe la spécialité objets fins de G (T1 +0,80) sans hériter de son dommage
(dommage maximal −0,53 pt ; ΔmIoU +0,45 pt, p = 0,0066 vs son contrôle apparié).

## Résumé

> We report a pre-registered, controlled evaluation of the **blob loss** of Kofler *et al.* [IPMI
> 2023] — an auxiliary term that gives every ground-truth instance the same weight regardless of
> its pixel count — ported to **exclusive multi-class semantic segmentation at the native
> Cityscapes resolution** (1024×2048, 19 classes, softmax regime). Arm **G** (CE + Dice +
> 0.5·blob) is compared against its exact paired reference **B** (CE + Dice): same
> ConvNeXt-V2-Base + UPerNet architecture, same 160-epoch recipe, same augmentation distribution,
> three shared seeds (42, 123, 456) — only the loss changes. The **pre-registered primary endpoint
> is null**: dataset-level official mIoU on the shared 500-image holdout, Δ(G−B) = **+0.170 pt**,
> 95 % CI [−0.233 ; +0.551], two-sided paired image-bootstrap p = **0.4032** (B = 10 000
> replicates). Equal weight per instance does **not** improve global mIoU in this regime. What the
> term actually does is a **measured trade-off**. It *buys* thin-object pixel coverage — traffic
> light +0.98, pole +0.58, bicycle +0.57 IoU vs B (Holm-significant within the 19-class family),
> truck +3.78 (CI excluding zero), Boundary F1 (3 px) +0.63 — and it *pays* in pedestrian instance
> integrity: strict pedestrian recall **−4.34**, small-instance stratum (T1) **−5.65**,
> crowd-group instances **−5.74**, pedestrian pixel precision **−2.70** (all vs paired B, Holm =
> 0), and a general **fragmentation** of the masks: connected components rise from 615.1 to 933.9
> per image (**×1.52**), up in **16 of 19 classes** against the paired arm B (14 Holm-significant
> rises; the only material decrease is terrain, −1.8) and in **19 of 19** against the control arm
> (person masks ×2.2) — a new per-class decomposition that *revises* the working mechanistic
> hypothesis of the program (the term does not prune small components; it creates holes). On the
> program's pre-registered versatility criterion (36 endpoints × 13 arms), G is a **dominated
> specialist**: mean percentile 41.9, worst rank 13/13, maximal damage −6.76 pt, 13 significant
> losses against 4 significant gains. The same verdict holds on a second dataset and a second
> probability regime: on BRATS 2023 (region-based sigmoid, MedNeXt, nnU-Net recipe, 5-fold CV, n =
> 1 196), the blob arm alone scored **−0.00376 Dice** vs baseline, yet as **expert 3 of the
> winning MoE-V3** it contributed **+0.00566 Dice, first of 29 arms** [DOI
> 10.5281/zenodo.22903668]. The Cityscapes companion mixture **MoE-V3-CS** (four experts
> initialised from B, D, Dp, G; top-2 patch-wise gate) tells the same story: it absorbs G's
> thin-object speciality — small-instance recall T1 +0.80 (significant), traffic light not
> degraded — without inheriting its damage (maximal damage −0.53 pt; ΔmIoU +0.45 pt, p = 0.0066 vs
> its paired control). **Contributions.** (1) A pre-registered **null primary** for
> equal-weight-per-instance auxiliary loss in full-resolution exclusive-softmax segmentation,
> reported as such. (2) A complete **trade-off diagnostic**: per-class IoU, seven business-metric
> families on pedestrian instances, and a **new per-class connected-component decomposition**
> (933.9 vs 615.1 components/image) that contradicts the initially hypothesised compaction
> mechanism — the paper reports what the tables show. (3) An **exact port** of Kofler's algebra
> (eq. 1) to the exclusive-softmax regime with precomputed instance packs: +1–2 % epoch time
> against +870 s/epoch for the naive path, naive↔precomputed parity locked by unit tests, and the
> horizontal-flip alignment pitfall documented. (4) Evidence, on two datasets and two probability
> regimes, that the term **pays as an initialised expert of a mixture** while null-or-harmful
> alone. (5) Public release of code, configs, tables, figures and regeneration scripts. ---

## Contenu du dépôt

| chemin | contenu |
|---|---|
| `paper_fr.md` / `paper_fr.pdf` | manuscrit français (15 p.) |
| `paper.md` / `paper.pdf` | manuscrit anglais (15 p.) |
| `build.sh` + `header.tex` | la recette exacte qui reconstruit les deux PDF (pandoc → XeLaTeX, letter, marges 1,7 cm, Liberation Serif) |
| `tables/` | T1-T6 (md + csv) et `paper4_tables.json`, la table de consolidation |
| `figures/` | F1 forest plot par classe, F2 arbitrage, F3 position polyvalence (png + pdf) |
| `src/losses/blob_loss.py` | port exact de l'algèbre de Kofler (eq. 1) au régime softmax exclusif |
| `src/losses/blob_lab.py` | précalcul des packs d'instances (`cc3d`, 22,4 ms/image) |
| `src/postprocessing/consensus.py` | `count_fragments` — composantes 8-connexes |
| `src/metrics/segmentation_metrics.py` | mIoU officielle, Boundary F1, métriques d'instances piétonnes |
| `configs/` | configs Hydra des bras G et B et du contrôle |
| `scripts/p4_blob_tables_figures.py` | régénère toutes les tables et figures du papier |
| `scripts/p4_check_numbers.py` | **50 checks** reliant chaque nombre du manuscrit à son artefact |
| `scripts/p4_check_manuscrit_gate.py` | rejoue les regex du gate DOI sur les manuscrits publiés |
| `tests/` | parité blob naïf↔précalculé, mIoU officielle |
| `analysis/` | la chaîne de provenance complète des nombres publiés |
| `analysis/EXCLUDED.md` | les artefacts lourds **non** embarqués, nommés un par un avec leur chemin de régénération |

## Reproduire

```bash
bash build.sh                              # reconstruit les deux PDF
python3 scripts/p4_check_numbers.py        # 50 checks manuscrit<->artefacts
python3 scripts/p4_check_manuscrit_gate.py # regex du gate DOI sur les manuscrits
python3 -m pytest tests/test_blob_loss.py  # parite naive <-> precalcule
```

Régénérer les tables et figures demande les artefacts par image d'`analysis/` plus le jeu
Cityscapes ; régénérer *le bras lui-même* demande ~87 h de GPU (3 seeds × 160 époques en
1024×2048 sur une RTX PRO 6000 96 Go) — voir `tables/T1_protocole_cout.md`.

**Déclaration de reproductibilité (déclarée, pas sur-vendue).** cuDNN benchmark reste activé
(`deterministic: false`, BF16) : la reproductibilité bit-à-bit sur GPU n'est **pas** revendiquée.
Ce qui est vérifié : mêmes artefacts → mêmes statistiques bit-à-bit, et forwards indépendants
des mêmes checkpoints → accord à 0,01 pt près (14/14 sanity checks bloquants).

## Papiers compagnons du programme

| papier | dataset | DOI |
|---|---|---|
| Régression auxiliaire par carte de distance | Cityscapes | [10.5281/zenodo.21006236](https://doi.org/10.5281/zenodo.21006236) |
| Ablation boundary loss (source du bras B apparié) | Cityscapes | [10.5281/zenodo.21006393](https://doi.org/10.5281/zenodo.21006393) |
| **Ce papier** — blob loss seul (bras G) | Cityscapes | voir `CITATION.cff` |
| Mélange d'experts initialisé par experts | BRATS 2023 | [10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668) |
| Boundary loss à poids fixe | BRATS 2023 | [10.5281/zenodo.22906447](https://doi.org/10.5281/zenodo.22906447) |
| Consensus de composantes connexes | BRATS 2023 | [10.5281/zenodo.22904810](https://doi.org/10.5281/zenodo.22904810) |
| Loss auxiliaire par carte de distance | BRATS 2023 | [10.5281/zenodo.20110976](https://doi.org/10.5281/zenodo.20110976) |

## Licences

Manuscrits, tables et figures : **CC-BY-4.0** (la licence du dépôt Zenodo). Code (`src/`,
`scripts/`, `tests/`, `configs/`) : **MIT** ([`LICENSE`](LICENSE)). Les données Cityscapes
restent soumises à leur propre licence.

## Citer

Voir [`CITATION.cff`](CITATION.cff). Auteur : Guillaume Cassez, chercheur indépendant
(ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931)).
