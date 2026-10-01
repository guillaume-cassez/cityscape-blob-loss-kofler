# CHANGELOG

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
  inédit (×1,52, hausse dans 18 classes sur 19, person ×2,2) qui **contredit** l'hypothèse de
  compaction initialement envisagée — le papier rapporte ce que montrent les tables (§6.4).
- Code : port exact de l'algèbre de Kofler (eq. 1) au régime softmax exclusif, avec packs
  d'instances précalculés (`src/losses/blob_loss.py`, `src/losses/blob_lab.py`), parité
  naïf↔précalculé verrouillée par tests unitaires (`tests/test_blob_loss.py`).
- Garde-fous : `scripts/p4_check_numbers.py` (50 checks manuscrit↔artefacts) et
  `scripts/p4_check_manuscrit_gate.py` (regex du gate DOI rejouées sur les manuscrits publiés).
- Artefacts lourds non embarqués, listés nommément dans `analysis/EXCLUDED.md`.
