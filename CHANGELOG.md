# CHANGELOG

## v1.2.0 — 2026-10-04

**Premier dépôt public depuis la v1.1.0** (record Zenodo 23085640, erratum du 2026-10-01).
Cette version rassemble les deux étapes correctives qui n'avaient été que stagées
localement (v1.1.1 du 2026-10-03, v1.1.2 du 2026-10-04). **Aucun chiffre du papier ne
change** : le critère primaire reste nul (Δ(G−B) = +0.170 pt, IC95
[-0.233 ; +0.551], p = 0.4032) et le diagnostic de
fragmentation reste celui de l'Annexe B. La seule modification de MÉTADONNÉE est la liste
des auteurs.

- **Auteurs — Stanislas Larnier en 2ᵉ position dans le record Zenodo lui-même.** Les
  versions publiées avant le 2026-10-04 (v1.0.0 record 23083560, v1.1.0 record 23085640)
  portent **Guillaume Cassez seul** : mesuré sur l'API publique le 2026-10-04
  (`creators = ['Cassez, Guillaume']`), alors que `paper.yaml` en portait déjà deux.
  Cause : la liste était écrite **en dur** dans le stager et dans le script de dépôt.
  Elle est désormais lue depuis `papers/paper4/paper.yaml` par le module
  `paper_authors.py` du dépôt de recherche, source unique du payload Zenodo, de
  `CITATION.cff`, de `.zenodo.json` et des deux README — plus aucune seconde copie ne
  peut diverger. Ce module n'est PAS embarqué ici : il lit `paper.yaml`, qui contient une
  adresse e-mail personnelle et ne part donc pas dans un dépôt public. Un record publié
  étant immuable, les deux versions antérieures restent en ligne à un seul auteur, et
  cette entrée le dit explicitement plutôt que de le laisser découvrir.
- **Le point B n'est plus masqué par le texte du point G** (figure F3, page 10 EN / 11 FR)
  et **le tableau métier n'a plus de lignes dédoublées** (page 8) : ce sont les deux
  défauts que Guillaume signalait le 2026-10-04 15h09, corrigés à la cause et documentés
  dans l'entrée v1.1.2 ci-dessous (mesures avant/après au renderer matplotlib).
- **Figures dessinées À LA LARGEUR IMPRIMÉE** (7,16 in = `\linewidth`, échelle 1,000) :
  les polices du code SONT les tailles sur le papier. Voir v1.1.2 pour les échelles
  mesurées avant (0,426 à 0,788) et le gate `verifier_taille_imprimee()` qui bloque
  désormais sous 6,5 pt imprimé.
- **Pagination mesurée, plus jamais recopiée.** Les PDF de cette version font
  **15 pages (EN)** et **16 pages (FR)**, lus par `pdfinfo`. La v1.1.0
  publique (record 23085640, re-téléchargée sans jeton, md5 `956cf11e…` EN / `4d659219…`
  FR) faisait **15 + 15** : le manuscrit français gagne donc **une page**, les figures
  dessinées à taille réelle étant plus hautes. Les deux README de ce bundle annonçaient
  « 15 p. » pour les DEUX langues — le FR était **faux**. Cause : les comptes étaient
  écrits en dur dans la prose du stager. `scripts/sync_bundle_pages.py` réécrit désormais
  chaque mention depuis `pdfinfo`, est appelé par `build.sh` après le build, et son mode
  `--check` sort 1 au premier écart (contrôle négatif probant : un « 12 p. » injecté dans
  un README est détecté, puis la restauration vérifiée par md5).
- Gates rejoués sur les fichiers de CETTE version : layout **exit 0** sur les 4 PDF des
  deux papiers (paper4 EN 15 p. 7435 mots, FR 16 p. 8222 mots ; 0
  débordement à droite, 0 hors papier en bas), figures embarquées **3/3 pixel-égales au
  canon** dans chaque langue, nombres **87/87**,
  manuscrits **PASS** (0 marqueur, 0 regex Forbidden, auteurs 2 ✓).

## v1.1.2 — 2026-10-04

**Figures à taille réelle, tableau métier sans lignes dédoublées, deux auteurs :
aucun chiffre changé.**
Statut : **rendu public par la v1.2.0 du 2026-10-04** — cette étape corrective
n'avait été que stagée localement, en attente du feu vert de Guillaume pour un acte
public engageant un tiers ; ce feu vert est venu le 2026-10-04 23h43. Le dépôt v1.1.0
(record 23085640) porte **Guillaume Cassez seul** : un record publié est immuable, il
reste donc en ligne tel quel, et la liste à deux auteurs ci-dessous n'est publique
qu'à partir de la v1.2.0.

- **Le point B n'est plus masqué par le texte du point G** (F3, page 10 EN / 11 FR).
  Guillaume : « le point B est masqué partiellement par le texte du point G ».
  Mesure AVANT au renderer matplotlib : l'annotation de G — 3 lignes, `bbox` blanc
  **opaque** `alpha=1.0` de **230,4 × 64,1 px**, posée à `xytext=(14, 18)` donc en
  haut à droite de G (41,93 ; −6,76) et s'étendant jusqu'à x = 54,4 — recouvrait
  **75 % de la boîte du marqueur de B** (51,23 ; −4,06) en FR et **73 %** en EN.
  Cause lue dans le gate : `verifier_figure` ne comparait que texte × texte et
  texte × `Line2D` ; un `scatter` n'est ni l'un ni l'autre, et la règle
  `_opaque_au_dessus` **exemptait** au contraire le fond opaque. Correctif :
  annotation déplacée dans le quadrant bas-droit de G (mesuré vide de tout point,
  le plus proche — consensus D⊘B — étant à x = 71,9), étiquettes de bras recentrées
  (`xytext=(6, 0)`, `va="center"`) pour ne plus frôler la pointillée du seuil T0,
  libellé du seuil recoupé sur 2 lignes avec fond blanc opaque en zorder 5. Mesure
  APRÈS : **0 % de recouvrement, point B entièrement visible**, en FR comme en EN.
  Gate ajouté : `verifier_occlusion_marqueurs()` dans `scripts/fig_overlap_gate.py`
  compare la boîte d'encre de tout texte à fond opaque au disque de chaque marqueur
  (`PathCollection` et `Line2D` à marker), en excluant la cible de l'annotation.
- **Taille du texte des figures — même cause racine que sur le papier 3.** Mesure
  AVANT sur les PDF livrés : échelles de placement **0,426** (F2_tradeoff, 51 textes
  imprimés sous 6,5 pt, minimum **3,80 pt**), **0,696** (F3_polyvalence_G) et
  **0,787** (F1_forest_perclass). Les figures étaient dessinées à 9-17 in de large
  puis réduites à `\linewidth` = 7,16 in : chaque `fontsize` du code était multiplié
  par ce facteur, et aucun gate ne le mesurait. Correctif : les 3 figures sont
  dessinées **À LA LARGEUR IMPRIMÉE**, échelle **1,000** mesurée sur les 6 PDF
  reconstruits (EN et FR), taille imprimée minimale **6,80 pt**, **0 texte sous
  6,5 pt**. Les libellés de classes de F2 passent en version courte (`classe_court`
  : la glose française seule en FR) — 29 caractères bilingues laissaient 1,06 in sur
  2,39 pour tracer l'intervalle de confiance ; F1, une page plus tôt, porte les 19
  classes dans les deux langues. Les libellés de valeurs de F2 sont décrochés en
  **points typographiques** (`offset points`) et plus en unités de données : à
  2,39 in de panneau, 3 % de l'amplitude valait 1,7 px, moins que la garde du gate
  plus le rayon du cap d'IC (mesuré : « ligne × texte traverse '+3,78' » et 5 autres).
- **Tableau des métriques métier (§6.3, page 8 EN / 9 FR) : 20 cellules d'IC95
  wrappées sur 2 lignes → 0.** Guillaume : « il faut plus serrer les colonnes pour
  éviter les doubles lignes dans les lignes du tableau ». Deux causes mesurées et
  corrigées dans `scripts/tables_latex_postprocess.py` :
  1. **le critère de choix du corps était le mauvais** — la boucle s'arrêtait à la
     PREMIÈRE taille dont la hauteur et la largeur tenaient dans la page. `\small`
     tenait, donc la table restait en `\small` avec 20 replis, et un corps plus
     petit n'était jamais essayé. Le critère devient `(nombre de sauts de lignes,
     rang du corps)` : le moins de lignes dédoublées d'abord, la plus grande police
     ensuite à égalité, en descendant jusqu'à 0 saut. La table passe en
     `\footnotesize`, **0 saut**.
  2. **`\tabcolsep` était écrasé en silence** — `header.tex` portait
     `\AtBeginEnvironment{tabular}{\small\setlength{\tabcolsep}{4pt}}`, or ce hook se
     déclenche À L'OUVERTURE de l'environnement, donc APRÈS le `\setlength` émis par
     le post-processeur. Les largeurs étaient optimisées avec une séparation, la
     composition en employait une autre. La séparation est désormais adaptative
     (4 pt ≤ 8 colonnes, 3 pt ≤ 10, 2 pt au-delà — à 4 pt, une table à 12 colonnes
     consacrait 88 pt, 17 % du budget, à du blanc), le hook du `header.tex` ne fixe
     plus que le corps de repli, et le post-processeur **refuse de composer**
     (exit 2) si un header force encore `\tabcolsep` dans un hook. Sans ce second
     correctif la table de polyvalence débordait la marge droite de 5 à 28 pt
     (7 mots à x2 = 569-594 pt pour une limite à 565,8), attrapés par
     `check_pdf_layout.py`.
- **Angle mort du gate tableaux corrigé** (`scripts/measure_table_cuts.py`) : la
  détection de cellule IC wrappée n'acceptait le fragment ouvrant qu'EN FIN DE
  LIGNE. Dans un tableau à 12 colonnes la cellule wrappée est au MILIEU de la ligne
  (les colonnes suivantes continuent), et le gate déclarait « 0 symptôme » sur les
  20 replis du PDF livré. La détection porte désormais sur toute la ligne, et un
  repli n'est BLOQUANT que si la continuation ne peut pas être un début de phrase
  (chiffre, signe, fermante) — les replis de prose d'une cellule de texte longue
  (table des critères T1-T4 du papier 3, colonne « Résultat recalculé » : Σ naturel
  833 pt pour un budget de 492) sont rapportés mais admis.
- **Auteurs — Stanislas Larnier en 2ᵉ position**, sur instruction de Guillaume du
  2026-10-04, aligné sur les quatre papiers BRATS du programme dont il est déjà
  co-auteur. La décision inverse (« papier mono-auteur, pas de gift authorship »,
  2026-09-30) est **rapportée et datée** dans `paper.yaml`, pas effacée.
  Contreparties d'intégrité : section **Contributions des auteurs** après §9,
  mention en page de titre de la date d'établissement de la liste et du fait que les
  dépôts Zenodo antérieurs portent Guillaume seul, contrôle **positif** dans
  `scripts/p4_check_manuscrit_gate.py`, regex Forbidden retirée des deux SPEC avec
  la décision datée à sa place, et `CITATION.cff` / `.zenodo.json` / `README.md` /
  `README_fr.md` du bundle à deux auteurs.
- Pagination : FR **16 pages** (inchangé), EN 14 → **15** (figures plus hautes).
- Gates rejoués sur les PDF finis : layout **exit 0** (8222 mots FR, 7435 EN, 0 à
  droite, 0 en bas), tableaux **exit 0** (0 longtable, 10 `table[H]`, 0 en-tête
  répété, **0 cellule IC wrappée**, 0 ligne dédoublée bloquante), figures
  embarquées **3/3 pixel-égales au canon** dans les deux langues, nombres **87/87**,
  manuscrits **PASS** (0 marqueur, 0 regex Forbidden, auteurs 2 ✓), figures
  **6/6 à 0 superposition d'encre** et autocontrôles **9/9**.

## v1.1.1 — 2026-10-03

**Chronologie du récit + mise en page et figures : aucun chiffre changé.**
Statut : **rendu public par la v1.2.0 du 2026-10-04** — étape corrective stagée
localement le 2026-10-03, jamais déposée seule.

- Correctif de build (21:35) : les corrections de figures ci-dessous n'étaient PAS
  dans les PDF construits à 19:59. Les manuscrits référencent `figures/…` en chemin
  relatif, donc le build lit les copies de `publish/repo/figures/` — restées au
  2026-10-02 : les 3 figures FR embarquées étaient périmées (mesure : raster p9
  3019 px de large contre 3008 px au canon ; F1 et F3 avaient des dimensions
  IDENTIQUES entre copie périmée et canon, seul le pixel les sépare). Désormais
  `build.sh` synchronise le canon (`scripts/sync_paper_figures.py`, copie vérifiée
  md5, fail-closed) puis le gate `scripts/check_figures_embedded.py` compare chaque
  raster du PDF FINI à la figure canonique — avant : 3/3 périmées, après : 3/3
  pixel-égales (EN comme FR). Un contrôle canon ↔ bundle a été ajouté au stager.

- Récit : ce papier est rédigé comme s'il précédait l'étude du mélange d'experts
  Cityscapes. Disparaissent donc les formules qui citaient un papier compagnon déjà
  écrit (« le mélange compagnon raconte la même histoire », « mesuré dans l'étude
  compagne », « ce que le mélange absorbe ») : le bras MoE-V3-CS n'intervient plus
  qu'en **contexte de plateau** (tables maîtresses P3.14/P3.17, artefacts du
  programme), et la moitié *usage* de la thèse s'appuie sur le volet BRATS **publié**
  [DOI 10.5281/zenodo.22903668]. Même tableau, mêmes valeurs, autre statut narratif.
- Tableaux : build post-traité (`scripts/tables_latex_postprocess.py`) — plus de
  table sciée sans en-tête répété ; la seule table de 20 lignes (§6.3) passe sur deux
  pages AVEC en-tête répété ; largeurs de colonnes calculées sur le contenu.
- Figures : libellés de F1/F2 sortis des lignes d'IC (à droite du cap haut si Δ > 0,
  à gauche du cap bas sinon) ; F3 : annotation MoE-V3-CS déplacée à droite du losange
  (elle était barrée par la ligne de seuil T0), libellé de seuil passé à gauche,
  titre débarrassé de la mention « (papier 3) ».
- Gates rejoués : layout 0 débordement (8 114 mots FR lus), `p4_check_numbers` 87/87,
  gate manuscrit PASS.

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
