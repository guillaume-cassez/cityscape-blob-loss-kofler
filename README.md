# Equal Weight per Instance Does Not Pay Alone

**A blob-loss auxiliary on full-resolution Cityscapes buys thin-object IoU with pedestrian
recall, and only pays as an expert of a mixture-of-experts.**

**Guillaume Cassez** · **Stanislas Larnier** — independent research · ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) (G.C.) · [HAL stanislas-larnier](https://cv.hal.science/stanislas-larnier) (S.L.) · [guillaume-cassez.fr](https://guillaume-cassez.fr/voiture-autonome/)

Preprint, CC-BY-4.0. Read [`paper.pdf`](paper.pdf) (EN, 15 p.) or
[`paper_fr.pdf`](paper_fr.pdf) (FR, 16 p.).

---

## What this is

A pre-registered, controlled evaluation of the **blob loss** of Kofler *et al.* (IPMI 2023) —
an auxiliary term giving every ground-truth instance the same weight regardless of its pixel
count — ported to **exclusive multi-class segmentation at native Cityscapes resolution**
(1024×2048, 19 classes, softmax). Arm **G** (CE + Dice + 0.5·blob) is compared with its exact
paired reference **B** (CE + Dice): same ConvNeXt-V2-Base + UPerNet, same 160-epoch recipe, same
augmentation, three shared seeds (42, 123, 456) — **only the loss differs**.

## The headline result is null, and that is the point

| | value |
|---|---|
| **Primary endpoint** (pre-registered) | dataset-level official mIoU, G vs paired B |
| Δ(G−B) | **+0.170 pt** |
| 95 % CI | [-0.233 ; +0.551] |
| two-sided paired image-bootstrap p | **0.4032** |
| bootstrap | B = 10,000 replicates, seed 20260618, holdout first:500 |
| mIoU (G / B) | 81.26 / 81.09 |
| robustness (2nd independent forward) | Δ +0.173 pt, p = 0.3952 — same verdict, ≤ 0.01 pt apart |

Equal weight per instance does **not** improve global mIoU in this regime. The paper reports
what the term actually does instead: a measured **trade-off**.

**Bought** — thin-object pixel coverage: traffic light +0.98, pole +0.58, bicycle +0.57 IoU
(Holm-significant within the 19-class family), truck +3.78 (CI excluding zero), Boundary F1
(3 px) +0.63.

**Paid** — pedestrian instance integrity: strict pedestrian recall −4.34, smallest size tercile
(T1) −5.65, crowd-group instances −5.74, pedestrian pixel precision −2.70 (all vs paired B,
Holm = 0).

**And a mechanism the tables contradict.** Connected components per image rise from
615.1 to 933.9 (**×1.52**), up in
**16 of 19 classes** against the paired arm B (14 of them
Holm-significant) and in **19 of 19** against the control arm; person
masks ×2.2. The only material decrease is terrain
(-1.8, Holm < 0.001) — wall and bus are null (p ≥ 0.70). The working
hypothesis of the program — that an instance-balancing term *compacts* masks by pruning small
components — is **explicitly refuted** by this per-class decomposition (§6.4): the term creates
holes, it does not prune fragments.

**Versatility.** On the program's pre-registered criterion (36 endpoints × 13 arms), G is a
**dominated specialist**: mean percentile 41.9, worst rank
13/13, maximal damage -6.76 pt, 13 significant losses against
4 significant gains.

**Where it does pay.** As an *initialised expert of a mixture*. On BRATS 2023 (sigmoid,
region-based) the blob arm alone scored −0.00376 Dice, yet as expert 3 of the winning MoE-V3 it
contributed +0.00566 Dice, first of 29 arms ([DOI 10.5281/zenodo.22903668]). The Cityscapes
companion MoE-V3-CS absorbs G's thin-object speciality (T1 +0.80) without inheriting its damage
(maximal damage −0.53 pt; ΔmIoU +0.45 pt, p = 0.0066 vs its paired control).

## Abstract

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
> 10.5281/zenodo.22903668]. On the Cityscapes program plateau, the expert-mixture arm initialised
> from these specialists (MoE-V3-CS, master table P3.14) points the same way, as context: ΔmIoU
> +0.45 pt [+0.11 ; +0.82], p = 0.0066 vs its recipe-paired control, maximal damage −0.53 pt,
> small-instance recall T1 +0.80, traffic light not degraded — the profile of an arm that puts G's
> thin-object speciality to use without carrying its damage. The dedicated analysis of that
> mixture arm (routing diagnostics, multiplicity families) is not the subject of this paper.
> **Contributions.** (1) A pre-registered **null primary** for equal-weight-per-instance auxiliary
> loss in full-resolution exclusive-softmax segmentation, reported as such. (2) A complete
> **trade-off diagnostic**: per-class IoU, seven business-metric families on pedestrian instances,
> and a **new per-class connected-component decomposition** (933.9 vs 615.1 components/image) that
> contradicts the initially hypothesised compaction mechanism — the paper reports what the tables
> show. (3) An **exact port** of Kofler's algebra (eq. 1) to the exclusive-softmax regime with
> precomputed instance packs: +1–2 % epoch time against +870 s/epoch for the naive path,
> naive↔precomputed parity locked by unit tests, and the horizontal-flip alignment pitfall
> documented. (4) Evidence, on two datasets and two probability regimes, that the term **pays as
> an initialised expert of a mixture** while null-or-harmful alone. (5) Public release of code,
> configs, tables, figures and regeneration scripts. ---

## Repository layout

| path | content |
|---|---|
| `paper.md` / `paper.pdf` | manuscript, English (15 p.) |
| `paper_fr.md` / `paper_fr.pdf` | manuscript, French (16 p.) |
| `build.sh` + `header.tex` | the exact recipe that rebuilds both PDFs (pandoc → XeLaTeX, letter, 1.7 cm margins, Liberation Serif) |
| `tables/` | T1-T6 (md + csv) and `paper4_tables.json`, the consolidation table |
| `figures/` | F1 per-class forest plot, F2 trade-off, F3 versatility position (png + pdf) |
| `src/losses/blob_loss.py` | exact port of Kofler's algebra (eq. 1) to the exclusive-softmax regime |
| `src/losses/blob_lab.py` | instance-pack precomputation (`cc3d`, 22.4 ms/image) |
| `src/postprocessing/consensus.py` | `count_fragments` — 8-connected components |
| `src/metrics/segmentation_metrics.py` | official-metric mIoU, Boundary F1, pedestrian instance metrics |
| `configs/` | Hydra configs of arms G and B and of the control |
| `scripts/p4_blob_tables_figures.py` | regenerates every table and figure of the paper |
| `scripts/p4_check_numbers.py` | **50 checks** binding each manuscript number to its artifact |
| `scripts/p4_check_manuscrit_gate.py` | re-runs the DOI gate's regexes on the published manuscripts |
| `tests/` | naive↔precomputed blob parity, official mIoU |
| `analysis/` | the full provenance chain of the published numbers |
| `analysis/EXCLUDED.md` | heavy artifacts **not** shipped, named one by one with their regeneration path |

## Reproducing

```bash
bash build.sh                              # rebuild both PDFs
python3 scripts/p4_check_numbers.py        # 50 manuscript<->artifact checks
python3 scripts/p4_check_manuscrit_gate.py # DOI-gate regexes on the manuscripts
python3 -m pytest tests/test_blob_loss.py  # naive <-> precomputed parity
```

Regenerating the tables and figures needs the per-image artifacts of `analysis/` plus the
Cityscapes dataset; regenerating the *arm itself* needs ~87 h of GPU (3 seeds × 160 epochs at
1024×2048 on one RTX PRO 6000 96 GB) — see `tables/T1_protocole_cout.md`.

**Reproducibility statement (declared, not overclaimed).** cuDNN benchmark is left on
(`deterministic: false`, BF16), so bit-exact GPU reruns are *not* claimed. What is verified:
same artifacts → same statistics bit-for-bit, and independent forwards of the same checkpoints
→ agreement within 0.01 pt (14/14 blocking sanity checks).

## Companion papers of the program

| paper | dataset | DOI |
|---|---|---|
| Distance-map auxiliary regression | Cityscapes | [10.5281/zenodo.21006236](https://doi.org/10.5281/zenodo.21006236) |
| Boundary-loss ablation (source of the paired arm B) | Cityscapes | [10.5281/zenodo.21006393](https://doi.org/10.5281/zenodo.21006393) |
| **This paper** — blob loss alone (arm G) | Cityscapes | see `CITATION.cff` |
| Expert-initialised mixture-of-experts | BRATS 2023 | [10.5281/zenodo.22903668](https://doi.org/10.5281/zenodo.22903668) |
| Fixed-weight boundary loss | BRATS 2023 | [10.5281/zenodo.22906447](https://doi.org/10.5281/zenodo.22906447) |
| Connected-component consensus | BRATS 2023 | [10.5281/zenodo.22904810](https://doi.org/10.5281/zenodo.22904810) |
| Distance-map auxiliary loss | BRATS 2023 | [10.5281/zenodo.20110976](https://doi.org/10.5281/zenodo.20110976) |

## Licences

Manuscripts, tables and figures: **CC-BY-4.0** (the licence of the Zenodo deposit). Code
(`src/`, `scripts/`, `tests/`, `configs/`): **MIT** ([`LICENSE`](LICENSE)). Cityscapes data
remains under its own licence.

## Citation

See [`CITATION.cff`](CITATION.cff). Authors: Guillaume Cassez, Stanislas Larnier, independent researchers (ORCID [0009-0007-0987-3931](https://orcid.org/0009-0007-0987-3931) for Guillaume Cassez).
