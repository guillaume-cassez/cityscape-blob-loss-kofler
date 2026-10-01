# CHANGELOG

## v1.1.0 — 2026-10-01

**Erratum — étiquette de multiplicité, aucune conclusion changée.**

La table de contexte 13 bras (§6.1) intitulait sa colonne de Holm « Holm (15 pairs) » /
« Holm (15 paires) » alors que ses valeurs provenaient de la famille **12 paires**
de la table maîtresse `results/moe_v3_cs/p314/master_table.json`. Mesure de l'écart :
D 0,058 (= 12×0,0048) au lieu de 0,075 ; MoE-V3-CS 0,073 au lieu de 0,092 ;
Dp 0,281 au lieu de 0,396 ; Cp 0,440 au lieu de 0,618. La famille 15 paires du
programme (`metiers_experts/table_metiers_experts.json`, reprise par P3.17) ajoute les 4
comparaisons fusion-contre-expert ; elle n'était donc pas celle affichée.

- **Aucune conclusion du papier ne change** : aucun bras ne passe Holm 0,05 dans *aucune* des
  deux familles (meilleur 0.0576 sur 12 paires,
  0.0750 sur 15 paires), et la phrase de §6.1 comme la
  limite 4 portaient déjà cette conclusion. Seules l'étiquette de famille et les valeurs de Holm
  du contexte étaient fausses, dans le sens d'une sous-estimation d'environ 25 à 30 %.
- Correction : §6.1 affiche désormais **les deux familles côte à côte**, chacune avec sa taille
  calculée depuis l'artefact et sa source ; A est signalé hors de la famille
  15 paires (couverture partielle, `annexe_couverture_partielle` de P3.17).
- Cause racine traitée, pas le symptôme : le générateur `scripts/p4_blob_tables_figures.py`
  écrivait l'en-tête **en dur** pendant qu'il remplissait la colonne avec le `p_holm` de P3.14.
  Il calcule maintenant les deux tailles de famille, **recompute** chaque Holm depuis les p bruts
  (fonction `holm()` du module `src/moe/bootstrap.py`) et le compare à la valeur stockée —
  40 checks bloquants ajoutés, sanités du générateur 16 → 58.
- Le rang « contrôle inclus » est lui aussi calculé (le contrôle, Δ = 0, se classe
  10ᵉ) : G est 7ᵉ sur 13, ce que la v1.0.0 énonçait déjà mais
  que la première version du correctif avait cassé en le déduisant de rang+1.
- `scripts/p4_check_numbers.py` passe de 58 à **87 checks** : il vérifie
  désormais l'**étiquette** de famille (tailles recalculées) et relit la table de contexte
  **ligne par ligne** dans les deux langues, avec garde anti-régression sur les 6 formulations
  fautives. Un contrôle négatif a été joué pour chaque garde-fou neuf (valeur corrompue dans
  l'artefact puis étiquette fautive réinjectée dans le manuscrit : détection, exit 1, restauration
  vérifiée par md5).
- Le caractère « † » des renvois de note est remplacé par « (a) » : le gate de pré-publication
  le classe parmi les marqueurs d'incomplétude (4 BLOCK mesurés avant, 0 après).
- PDF reconstruits, 0 glyphe manquant ; le staging revérifie lui-même la fraîcheur des PDF et
  l'égalité des dimensions de **chaque** image embarquée avec les figures sources (le nombre est
  mesuré au staging, pas recopié ici) ; gate local PASS 0 block 0 warning, gate global
  READY TO PUBLISH 5/5.

## v1.0.0 — 2026-10-01

Premier dépôt public du paper 4 du programme Cityscapes (blob loss de Kofler, bras G).

- Manuscrits EN (`paper.md` / `paper.pdf`, 15 pages) et FR (`paper_fr.md` / `paper_fr.pdf`,
  15 pages), 0 glyphe manquant, 0 overfull hbox.
- Critère primaire pré-enregistré rapporté **nul** : Δ(G−B) = +0.170 pt de mIoU,
  IC95 [-0.233 ; +0.551], p = 0.4032
  (bootstrap apparié par image, B = 10,000, seed 20260618,
  holdout first:500). Résultat négatif assumé comme tel.
- Robustesse : second forward GPU indépendant des mêmes checkpoints,
  Δ = +0.173 pt, p = 0.3952 — même verdict, écart ≤ 0,01 pt.
- Tables T1-T6 (`tables/`), figures F1-F3 (`figures/`), diagnostic de fragmentation par classe
  inédit (×1.52, hausse dans 16 classes sur 19 vs B apparié
  dont 14 Holm-significatives, 19 sur 19 vs le bras
  contrôle, person ×2.2) qui **contredit** l'hypothèse de
  compaction initialement envisagée — le papier rapporte ce que montrent les tables (§6.4).
- Code : port exact de l'algèbre de Kofler (eq. 1) au régime softmax exclusif, avec packs
  d'instances précalculés (`src/losses/blob_loss.py`, `src/losses/blob_lab.py`), parité
  naïf↔précalculé verrouillée par tests unitaires (`tests/test_blob_loss.py`).
- Garde-fous : `scripts/p4_check_numbers.py` (50 checks annoncés à l'époque, 58 réellement
  passés — compte désormais mesuré et non recopié) et `scripts/p4_check_manuscrit_gate.py`
  (regex du gate DOI rejouées sur les manuscrits publiés).
- Artefacts lourds non embarqués, listés nommément dans `analysis/EXCLUDED.md`.
- ⚠️ Retiré par v1.1.0 : l'étiquette « Holm (15 pairs) » de la table de contexte 13 bras
  (§6.1) ne correspondait pas à la famille dont les valeurs provenaient. Voir ci-dessus.
