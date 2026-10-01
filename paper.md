---
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \usepackage{booktabs}
---

# Equal Weight per Instance Does Not Pay Alone: A Blob-Loss Auxiliary on Full-Resolution Cityscapes Buys Thin-Object IoU with Pedestrian Recall, and Only Pays as an Expert of a Mixture-of-Experts

*Cityscapes val · ConvNeXt-V2-Base + UPerNet · CE + Dice + 0.5·blob (Kofler, IPMI 2023) vs CE + Dice · 3 seeds × 160 epochs at 1024×2048*

---

## Abstract

We report a pre-registered, controlled evaluation of the **blob loss** of Kofler *et al.* [IPMI 2023] — an auxiliary term that gives every ground-truth instance the same weight regardless of its pixel count — ported to **exclusive multi-class semantic segmentation at the native Cityscapes resolution** (1024×2048, 19 classes, softmax regime). Arm **G** (CE + Dice + 0.5·blob) is compared against its exact paired reference **B** (CE + Dice): same ConvNeXt-V2-Base + UPerNet architecture, same 160-epoch recipe, same augmentation distribution, three shared seeds (42, 123, 456) — only the loss changes.

The **pre-registered primary endpoint is null**: dataset-level official mIoU on the shared 500-image holdout, Δ(G−B) = **+0.170 pt**, 95 % CI [−0.233 ; +0.551], two-sided paired image-bootstrap p = **0.4032** (B = 10 000 replicates). Equal weight per instance does **not** improve global mIoU in this regime. What the term actually does is a **measured trade-off**. It *buys* thin-object pixel coverage — traffic light +0.98, pole +0.58, bicycle +0.57 IoU vs B (Holm-significant within the 19-class family), truck +3.78 (CI excluding zero), Boundary F1 (3 px) +0.63 — and it *pays* in pedestrian instance integrity: strict pedestrian recall **−4.34**, small-instance stratum (T1) **−5.65**, crowd-group instances **−5.74**, pedestrian pixel precision **−2.70** (all vs paired B, Holm = 0), and a general **fragmentation** of the masks: connected components rise from 615.1 to 933.9 per image (**×1.52**), up in **16 of 19 classes** against the paired arm B (14 Holm-significant rises; the only material decrease is terrain, −1.8) and in **19 of 19** against the control arm (person masks ×2.2) — a new per-class decomposition that *revises* the working mechanistic hypothesis of the program (the term does not prune small components; it creates holes). On the program's pre-registered versatility criterion (36 endpoints × 13 arms), G is a **dominated specialist**: mean percentile 41.9, worst rank 13/13, maximal damage −6.76 pt, 13 significant losses against 4 significant gains.

The same verdict holds on a second dataset and a second probability regime: on BRATS 2023 (region-based sigmoid, MedNeXt, nnU-Net recipe, 5-fold CV, n = 1 196), the blob arm alone scored **−0.00376 Dice** vs baseline, yet as **expert 3 of the winning MoE-V3** it contributed **+0.00566 Dice, first of 29 arms** [DOI 10.5281/zenodo.22903668]. The Cityscapes companion mixture **MoE-V3-CS** (four experts initialised from B, D, Dp, G; top-2 patch-wise gate) tells the same story: it absorbs G's thin-object speciality — small-instance recall T1 +0.80 (significant), traffic light not degraded — without inheriting its damage (maximal damage −0.53 pt; ΔmIoU +0.45 pt, p = 0.0066 vs its paired control).

**Contributions.** (1) A pre-registered **null primary** for equal-weight-per-instance auxiliary loss in full-resolution exclusive-softmax segmentation, reported as such. (2) A complete **trade-off diagnostic**: per-class IoU, seven business-metric families on pedestrian instances, and a **new per-class connected-component decomposition** (933.9 vs 615.1 components/image) that contradicts the initially hypothesised compaction mechanism — the paper reports what the tables show. (3) An **exact port** of Kofler's algebra (eq. 1) to the exclusive-softmax regime with precomputed instance packs: +1–2 % epoch time against +870 s/epoch for the naive path, naive↔precomputed parity locked by unit tests, and the horizontal-flip alignment pitfall documented. (4) Evidence, on two datasets and two probability regimes, that the term **pays as an initialised expert of a mixture** while null-or-harmful alone. (5) Public release of code, configs, tables, figures and regeneration scripts.

---

## 1. Introduction

Voxel-wise losses — cross-entropy, Dice — weight every ground-truth instance by its volume. Small instances (rare classes, distant or partially occluded objects) contribute few pixels to the loss and are, in a strict optimisation sense, *rationally sacrificed*. Kofler *et al.* [2023] proposed the **blob loss**: an auxiliary term that computes a soft binary dice **per ground-truth instance**, over a domain restricted to that instance plus all pixels outside other instances of the same class, and averages instances → class → image → batch. Every instance then carries the same total weight regardless of its pixel count. The term was validated in biomedical segmentation under the nnU-Net **region-based** recipe (overlapping WT/TC/ET targets, sigmoid channels), where it improved small-structure metrics.

This paper asks the transfer question on the other side of the fence:

> Does equal weight per instance pay in **exclusive multi-class** semantic segmentation — softmax over 19 flat classes — at the **native Cityscapes resolution**, under a fixed training budget and a pre-registered primary endpoint?

The setting is a controlled arm of a four-paper loss program on Cityscapes at 1024×2048 with ConvNeXt-V2-Base + UPerNet: a distance-map auxiliary regression study [Cassez 2026a, DOI 10.5281/zenodo.21006236], a boundary-loss ablation whose arm B (CE + Dice, 160 epochs, seeds 42/123/456) is the exact paired reference used here [Cassez 2026b, DOI 10.5281/zenodo.21006393], this blob-loss study, and a companion mixture-of-experts report (MoE-V3-CS) in which the present arm G serves as expert 3. The program mirrors a completed BRATS track [DOI 10.5281/zenodo.22903668; 10.5281/zenodo.22906447; 10.5281/zenodo.22904810; 10.5281/zenodo.20110976], so every blob arm exists on both datasets under two probability regimes.

The answer, measured before this manuscript was written and pre-registered in the program's artifact chain, is **no — the primary endpoint is null** (Δ mIoU +0.170 pt, p = 0.4032). This is therefore a negative-result paper in the house discipline of the program: negative and neutral results are reported as such, all official metrics are shown, and the contribution is the **diagnostic** (what the term buys, what it breaks, and the mechanism the tables actually support), the **exact port** (algebra, cost, pitfalls), and the **usage** (an initialised expert of a mixture, measured on two datasets).

---

## 2. Related work

**Class imbalance in semantic segmentation.** Cross-entropy with class re-weighting is the standard remedy; we use ISNS weights (inverse square-root of the number of samples) in *both* arms, so the comparison isolates the blob term on top of an already re-weighted CE. Dice loss [Milletari 2016] optimises region overlap and is class-balanced at the *class* level, not at the *instance* level: within a class, one large instance still dominates many small ones. Focal loss [Lin 2017] re-weights hard pixels, again pixel-wise. The blob loss is the imbalance remedy that operates **per instance**.

**Blob loss.** Kofler *et al.* [2023, IPMI; arXiv:2205.08209] introduce the term for biomedical segmentation under the nnU-Net region-based recipe [Isensee 2021], with total loss $\mathcal{L} = \mathcal{L}_{seg} + \beta\,\mathcal{L}_{blob}$ and the paper's ratio $\alpha:\beta = 2:1$. Their validation regime is **sigmoid** (independent per-region channels, overlapping regions). The exclusive-softmax regime studied here puts classes in direct competition for every pixel; §7 discusses why the verdict differs from the intuition the term invites.

**Boundary-aware and distance-based companions.** Within the same program and hardware: the Kervadec boundary loss [Kervadec 2019] ablation [Cassez 2026b] and the distance-map auxiliary regression [Cassez 2026a] both report their pre-registered endpoints on the identical holdout and bootstrap protocol, which makes the present arm directly comparable to them (13-arm context table, §6.1).

**Fragmentation and consensus filtering.** Connected-component counts are a spatial-coherence proxy to which dataset-level mIoU is blind. The companion boundary paper measures a CC **consensus veto** (C⊘B) that prunes −18.7 % of spurious fragments at no mIoU cost [Cassez 2026b]; on BRATS the analogous rule is published in [DOI 10.5281/zenodo.22904810]. The per-class fragment decomposition introduced here (§6.4) is, to our knowledge, the first measurement of how an instance-balancing loss moves fragmentation *per class* on Cityscapes.

**Mixture-of-experts from initialised specialists.** The BRATS MoE-V3 [DOI 10.5281/zenodo.22903668] trains a patch-wise gate over four experts initialised from independently trained specialist arms — including the blob arm. The Cityscapes companion MoE-V3-CS transposes the design (four experts B, D, Dp, G; top-2 gate over 3×3 patches; 80 epochs from expert weights). This paper cites both for the *usage* half of its thesis; the MoE-V3-CS results themselves are reported in the companion study, not re-measured here.

---

## 3. Method — the blob loss in the exclusive-softmax regime

Let $\Omega$ be the image domain, $C = 19$ the number of classes, $p_c(x) \in [0,1]$ the **softmax** probability of class $c$ at pixel $x$ (so $\sum_c p_c(x) = 1$ — the regime difference with BRATS), and $y(x) \in \{0,\dots,18,255\}$ the ground-truth train id (255 = void/ignore). For class $c$, the ground-truth mask splits into connected **instances** $n \in I_c$ (8-connectivity on the label map; group annotations of the Cityscapes instance maps are handled at the business-metric level, §5.3, not inside the loss).

**Per-instance domain.** Following Kofler *et al.* eq. 1 exactly, the domain of instance $n$ of class $c$ is the whole image *minus the other instances of the same class*:

$$\Omega_n = \bigl(\{x : y(x) \neq 255\} \smallsetminus \bigcup_{m \in I_c,\, m \neq n} m\bigr) \cup n .$$

**Per-instance soft binary dice.** With $t_n$ the indicator of instance $n$, $s_n = \sum_{x \in n} p_c(x)$ the predicted mass on the instance, $c_n = |n|$ its pixel count, and $S_0^{(c)} = \sum_{x \in \Omega_n \smallsetminus n} p_c(x)$ the mass on the instance-free part of its domain (so $\sum_{x \in \Omega_n} p_c = S_0^{(c)} + s_n$):

$$\ell_n = 1 - \frac{2 s_n + \varepsilon}{S_0^{(c)} + s_n + c_n + \varepsilon}, \qquad \varepsilon = 1.0 \text{ (nnU-Net convention).}$$

Ignored pixels (255) fall in a trash bucket: they are excluded from both $S_0^{(c)}$ and the instances, and are never counted as false negatives (the `valid` semantics of the BRATS port).

**Two-level mean.** Faithful to the BRATS `RegionBlobLoss` the port comes from:

$$\mathcal{L}_{blob} = \frac{1}{|\mathcal{B}|} \sum_{\text{img} \in \mathcal{B}} \frac{1}{|C_{img}|} \sum_{c \in C_{img}} \frac{1}{|I_c|} \sum_{n \in I_c} \ell_n ,$$

where $C_{img}$ is the set of classes present in the image. Every instance of every present class carries the same total weight per image — the defining property of the term. `min_blob_pixels = 0`: no instance is filtered by size, since small instances are precisely what the term is meant to rescue.

**Composite losses (arms G and B).** With CE at ISNS class weights and Dice (smooth 1.0), both ignoring 255:

$$\mathcal{L}_G = 0.5\,\mathcal{L}_{CE}^{ISNS} + 0.5\,\mathcal{L}_{Dice} + 0.5\,\mathcal{L}_{blob}, \qquad \mathcal{L}_B = 0.5\,\mathcal{L}_{CE}^{ISNS} + 0.5\,\mathcal{L}_{Dice}.$$

$\beta = 0.5$ is Kofler's default ($\alpha:\beta = 2:1$ with the segmentation loss weighing 1.0 in total). **No sweep of $\beta$ is performed** — the fixed-budget comparison stance of the whole program (same position as the companion papers on $\lambda_b$ and the distance-map weight).

---

## 4. Implementation — precomputed instance packs

**Two execution paths, same algebra.** (1) *Precomputed* (used for training arm G): per-image instance labels are packed offline into `.blob.npz` files (`blob_glob` / `blob_counts` / `blob_csr`, built by `src/losses/blob_lab.py` via `scripts/precompute_blob_lab.py`) and delivered by the dataloader; the forward pass is then elementary GPU scatter-add kernels — the CPU connected-component pass and the 26 small per-class GPU synchronisations disappear from the training loop. (2) *On-the-fly fallback*: without a pack, the same packing functions recompute it from the target — this is the correctness reference used by tests and evaluation harnesses, giving structural parity between the two paths.

**Measured cost** (RTX PRO 6000 Blackwell, full-resolution ground truth, 2026-09-17): the naive on-the-fly path costs **+870 s/epoch** — unacceptable; the precomputed path costs **+1–2 %/epoch**. The connected-component labelling itself runs at **22.4 ms/image** with `cc3d` (8-connectivity; 16 classes present, 118 instances on a representative full-resolution image) against 59.7 ms with `scipy.ndimage.label`. Packs are computed once for the 2 975-image train split (labels are shared across seeds).

**Parity and regeneration.** Naive↔precomputed parity is locked by `tests/test_blob_loss.py`. All 2 975 packs were regenerated **bit-identical** on 2026-09-26 (verified over the full set). The packs themselves are heavy and are not shipped in the release bundle; `scripts/precompute_blob_lab.py` regenerates them identically.

**The flip pitfall (documented).** The instance pack must undergo the **same** horizontal flip as image and label. In arm G the dataset applies the flip itself (p = 0.5, jointly on image + label + pack) and the albumentations flip is **disabled** in the config, so the blob glob can never be misaligned. The augmentation *distribution* is identical to arm B's (hflip p = 0.5, photometric jitter, Gaussian blur); only the application point differs. A sanity inherited from the program's SDT audit: all three resolved G configs have the structural flip OFF — the arm is unaffected by the misalignment bug found in earlier boundary/distmap arms.

---

## 5. Experimental setup

### 5.1 Model, recipe, budget

**Architecture** (identical in both arms): ConvNeXt-V2-Base [Woo 2023] pretrained ImageNet-22K (FCMAE) then fine-tuned on ImageNet-1K, + UPerNet head [Xiao 2018] with auxiliary FCN deep supervision (weight 0.4), 19 logits per pixel at full input resolution. **Training**: 160 epochs, batch 2 × gradient accumulation 4 (effective 8), AdamW (lr 6×10⁻⁵, weight decay 0.01), polynomial decay (power 1.0, 1-epoch warmup), BF16 autocast, channels_last, inputs at native 1024×2048 without cropping. Seeds **42, 123, 456** in both arms. Arm G trained 17–20 September 2026.

**Measured cost of arm G** (per seed, `results/p3_queue/pilot_fullres_G_blob_seed{42,123,456}.log`): 160/160 epochs, median **656 s/epoch** [655–669], VRAM **31.5 GB**, final train loss 0.3743 / 0.3735 / 0.3765, ≈ **29.2 h per seed**, ≈ **87 h GPU** for the arm.

### 5.2 Data and holdout

Cityscapes fine annotations [Cordts 2016]: 2 975 train / 500 val images, 19 evaluation classes. All evaluations run on the program's **pre-specified shared holdout**: the first 500 val images in loader order (`first:500`) — identical for all 13 arms of the program, fixed before any arm was trained. The official Cityscapes leaderboard was **not** submitted to (declared limitation, §8).

### 5.3 Metrics

* **mIoU** — dataset-level, from a single aggregated confusion matrix, estimator validated **bit-identical to the official `cityscapesScripts`** routine (`tests/test_official_miou.py`). Per-class IoU from the same matrices.
* **Boundary F1 (3 px)** — per-class F1 of predicted vs ground-truth contours within a 3-pixel tolerance [protocol of Perazzi 2016], averaged over classes present per image; also restricted to {person, rider}.
* **Pedestrian instance metrics** — from the official `*_gtFine_instanceIds.png` maps, classes {person, rider}: pixel recall *any-ped* (prediction $\in$ {person, rider}) and *strict* (prediction = the instance's own class); detection rate at recall ≥ 0.5; pedestrian **pixel precision** (fraction of predicted-pedestrian pixels falling inside a ground-truth pedestrian instance).
* **Strata** — all instances / individual instances (instId ≥ 1000) / **crowd-group** instances (instId < 1000 — the official ambiguous/tightly-packed pedestrian groups, the best available proxy for partially occluded pedestrians) / **size terciles** of individual instances, T1 = smallest tercile (proxy for distance and occlusion), T3 = largest.
* **Fragments** — 8-connected components per image summed over the 19 class masks (`count_fragments`, `src/postprocessing/consensus.py`), plus a **new per-class decomposition** introduced by this paper (components per class per image, arms G/B/control).

### 5.4 Pre-registered primary endpoint and statistics

The single pre-registered primary endpoint is the **dataset-level official mIoU, arm G vs its paired reference B** (paired = same architecture, recipe, budget, augmentation, seeds — only the loss differs). Test: **paired image-bootstrap** over the 500 holdout images — positions resampled with replacement, **B = 10 000** replicates, bootstrap seed **20260618**, the three seeds averaged *within each replicate*, 95 % percentile CI, two-sided p = 2·min(frac Δ ≤ 0, frac Δ ≥ 0). The primary family is the single pair, so Holm = p.

Secondary/exploratory families, each with its Holm correction declared where reported: per-class IoU (19-class family per arm pair), business metrics (Holm(2) over this paper's two pairs G-vs-B and G-vs-control), per-class fragmentation (19-class family), 13-arm mIoU context (**two** families, both given side by side in §6.1: the 12-pair family of the master table `results/moe_v3_cs/p314/master_table.json` and the program's 15-pair exploratory family of `results/moe_v3_cs/metiers_experts/table_metiers_experts.json`).

**Two significance criteria, never conflated.** *Holm-significant* (corrected p < 0.05 within the stated family) and *CI-excluding-zero* (raw 95 % CI) are reported **separately** everywhere — bold in tables marks Holm significance only, and the p / Holm columns carry the exact values. This distinction matters for this arm: e.g. truck +3.78 IoU vs B excludes zero but does not survive the 19-class Holm (0.460), while traffic light +0.98 survives (Holm < 0.001).

**The two comparison pairs.** The budget-paired reference is **B** (160 epochs) — the pair of the primary endpoint. The 13-arm program context uses a different reference, the MoE **control** (CE + Dice, 80 epochs, B-initialised) — *not* budget-paired with G, always labelled as context, never substituted for the paired pair. Both are reported side by side throughout.

**Non-determinism, declared.** cuDNN benchmark is on (`deterministic: false`, BF16). The primary numbers come from the harness forward of the pre-registered artifact; the business-metric regeneration (§6.3) comes from a second independent GPU forward of the *same* checkpoints. The two agree within **≤ 0.01 pt** on every shared quantity (per-seed mIoU gaps ≤ 2×10⁻⁴ pt; primary Δ +0.170 vs +0.173, p 0.4032 vs 0.3952 — same verdict), of the same order as the 4.1×10⁻³ pt already documented in the program's consolidation sanity.

---

## 6. Results

### 6.1 Primary endpoint: null

Pre-registered primary, paired image-bootstrap over the 500-image holdout (artifact `results/moe_v3_cs/harness/table_Gseul_P310.json`, 2026-09-22):

| Quantity | G — CE+Dice+0.5·blob | B — CE+Dice |
|---|---|---|
| mIoU (point) | **81.263** | **81.093** |
| 95 % CI | [79.815 ; 82.431] | [79.694 ; 82.204] |
| mIoU seed 42 | 81.297 | 81.429 |
| mIoU seed 123 | 81.291 | 80.861 |
| mIoU seed 456 | 81.202 | 80.989 |

**Δ(G−B) = +0.170 pt · 95 % CI [−0.233 ; +0.551] · p = 0.4032 (single pair, Holm = p) → not significant.**

The honest reading: the primary endpoint is **null**. The blob loss does not improve mIoU in full-resolution exclusive-softmax semantic segmentation. Per-seed values do not hide a consistent direction either: G wins seed 123 (+0.43) and seed 456 (+0.21), loses seed 42 (−0.13).

The second, independent GPU forward of the same checkpoints (business-metric regeneration path, §6.3) gives Δ = +0.173 pt [−0.229 ; +0.554], p = 0.3952 — same verdict, gap ≤ 0.01 pt (declared cuDNN/BF16 non-determinism).

**13-arm program context** (reference: MoE control, CE + Dice 80 epochs — *not* budget-paired with G; artifact `results/moe_v3_cs/p314/master_table.json`):

| # | Arm | mIoU | Δ vs control | p | Holm (12-pair family) | Holm (15-pair family) |
|---|---|---|---|---|---|---|
| 1 | D · CE+Kervadec EDT | 81.69 | +0.52 [+0.14 ; +0.91] | 0.0048 | 0.058 | 0.075 |
| 2 | consensus D⊘B | 81.65 | +0.48 [+0.11 ; +0.87] | 0.0082 | 0.082 | 0.107 |
| 3 | consensus Dp⊘B | 81.65 | +0.48 [+0.02 ; +0.96] | 0.0396 | 0.317 | 0.414 |
| 4 | Dp · CE+SDT (distmap) | 81.64 | +0.47 [+0.04 ; +0.93] | 0.0312 | 0.281 | 0.396 |
| 5 | MoE-V3-CS (4 experts) | 81.62 | +0.45 [+0.11 ; +0.82] | 0.0066 | 0.073 | 0.092 |
| 6 | A · CE alone | 81.28 | +0.11 [−0.27 ; +0.52] | 0.5722 | 1.000 | n.a. (a) |
| 7 | **G · CE+Dice+Blob (this paper)** | **81.26** | **+0.10 [−0.28 ; +0.47]** | **0.6552** | **1.000** | **1.000** |
| 8 | consensus C⊘B | 81.24 | +0.07 [−0.34 ; +0.47] | 0.7758 | 1.000 | 1.000 |
| 9 | C · CE+Dice+Kervadec EDT | 81.23 | +0.06 [−0.34 ; +0.47] | 0.8096 | 1.000 | 1.000 |
| 10 | B · CE+Dice (160 ep) | 81.09 | −0.08 [−0.45 ; +0.28] | 0.6748 | 1.000 | 1.000 |
| 11 | consensus Cp⊘B | 81.01 | −0.15 [−0.52 ; +0.17] | 0.3498 | 1.000 | 1.000 |
| 12 | Cp · CE+Dice+SDT | 80.89 | −0.28 [−0.60 ; +0.02] | 0.0628 | 0.440 | 0.618 |
| — | control (ref, CE+Dice 80 ep) | 81.17 | — | — | — | — |

(a) A is outside the 15-pair family (partial coverage; `annexe_couverture_partielle` of P3.17 = ['A']), so no Holm value exists for it there.

**Two distinct multiplicity families, both reported.** The *12-pair* family is the master table itself (each of the 12 arms against the control, `results/moe_v3_cs/p314/master_table.json`). The *15-pair* family is the program's exploratory family, which adds the 4 fusion-versus-expert comparisons (`results/moe_v3_cs/metiers_experts/table_metiers_experts.json`, carried over by P3.17). Family sizes are computed from the artifacts and each Holm value is recomputed from the raw p-values and compared against the stored one (blocking check, `papers/paper4/tables/SANITY.md`).

G ranks **7th of 13** by amplitude (the control, Δ = 0, ranks 10th). **No arm passes Holm 0.05 in either family** — best 0.0576 over 12 pairs, 0.0750 over 15 pairs — stated as-is: on mIoU, the conclusions of this program rest on **amplitudes and their CIs**, not on post-Holm significance.

### 6.2 Per-class IoU: thin-object coverage bought

![F1 — Per-class forest plot, G vs B paired](figures/F1_forest_perclass_GvsB.png)

*Figure 1: per-class Δ IoU, arm G vs paired reference B (bootstrap apparié, B = 10 000). Filled markers = Holm-significant within the 19-class family; open markers = CI excluding zero without Holm survival; the two criteria are plotted separately, never merged.*

Full 19-class table (paired G−B; context G−control from the same artifact family). Bold = Holm (19-class family) < 0.05:

| Class | IoU G | IoU B | Δ(G−B) | 95 % CI | p | Holm | Δ(G−ctrl) | 95 % CI | p | Holm |
|---|---|---|---|---|---|---|---|---|---|---|
| road | 98.45 | 98.44 | +0.01 | [−0.05 ; +0.06] | 0.7970 | 1.000 | +0.04 | [−0.02 ; +0.10] | 0.1874 | 1.000 |
| sidewalk | 87.04 | 87.07 | −0.03 | [−0.44 ; +0.38] | 0.9184 | 1.000 | +0.06 | [−0.31 ; +0.48] | 0.7642 | 1.000 |
| building | 93.32 | 93.27 | +0.05 | [−0.13 ; +0.18] | 0.5170 | 1.000 | +0.07 | [−0.03 ; +0.18] | 0.1612 | 1.000 |
| wall | 57.21 | 57.92 | −0.70 | [−4.97 ; +2.43] | 0.7974 | 1.000 | +1.07 | [−1.32 ; +4.05] | 0.4916 | 1.000 |
| fence | 65.35 | 66.07 | −0.72 | [−2.47 ; +0.71] | 0.3874 | 1.000 | −0.46 | [−1.29 ; +0.34] | 0.2668 | 1.000 |
| pole | 70.34 | 69.76 | **+0.58** | [+0.34 ; +0.81] | <0.0001 | <0.001 | +0.29 | [+0.06 ; +0.51] | 0.0136 | 0.231 |
| traffic light | 77.83 | 76.85 | **+0.98** | [+0.58 ; +1.37] | <0.0001 | <0.001 | +0.47 | [+0.12 ; +0.82] | 0.0108 | 0.194 |
| traffic sign | 83.92 | 83.52 | +0.40 | [−0.04 ; +0.83] | 0.0686 | 0.706 | +0.21 | [−0.13 ; +0.54] | 0.2182 | 1.000 |
| vegetation | 93.15 | 93.09 | +0.06 | [−0.00 ; +0.13] | 0.0588 | 0.706 | +0.02 | [−0.03 ; +0.07] | 0.4260 | 1.000 |
| terrain | 66.40 | 66.10 | +0.30 | [−0.46 ; +1.06] | 0.4454 | 1.000 | +0.01 | [−0.61 ; +0.60] | 0.9850 | 1.000 |
| sky | 95.64 | 95.60 | +0.05 | [−0.05 ; +0.15] | 0.3538 | 1.000 | +0.11 | [+0.00 ; +0.22] | 0.0410 | 0.656 |
| person | 85.22 | 84.90 | +0.32 | [+0.07 ; +0.56] | 0.0146 | 0.234 | +0.07 | [−0.18 ; +0.30] | 0.5824 | 1.000 |
| rider | 67.58 | 68.49 | −0.90 | [−1.98 ; +0.03] | 0.0632 | 0.706 | −1.34 | [−2.42 ; −0.35] | 0.0072 | 0.137 |
| car | 95.71 | 95.46 | +0.25 | [+0.03 ; +0.54] | 0.0212 | 0.297 | +0.14 | [−0.13 ; +0.45] | 0.3136 | 1.000 |
| truck | 84.03 | 80.24 | +3.78 | [+0.09 ; +8.30] | 0.0354 | 0.460 | +3.43 | [−1.22 ; +8.31] | 0.1826 | 1.000 |
| bus | 90.07 | 89.75 | +0.32 | [−1.38 ; +2.20] | 0.7154 | 1.000 | −1.48 | [−4.74 ; +1.01] | 0.3274 | 1.000 |
| train | 81.20 | 83.74 | −2.54 | [−5.59 ; −0.34] | 0.0176 | 0.264 | −1.04 | [−2.79 ; +0.86] | 0.2966 | 1.000 |
| motorcycle | 71.15 | 70.70 | +0.46 | [−0.90 ; +2.31] | 0.5112 | 1.000 | +0.07 | [−0.93 ; +0.99] | 0.9362 | 1.000 |
| bicycle | 80.38 | 79.81 | **+0.57** | [+0.28 ; +0.90] | <0.0001 | <0.001 | +0.05 | [−0.17 ; +0.29] | 0.6634 | 1.000 |

Against paired **B**: the Holm survivors are exactly the thin, signal-rich classes — **traffic light +0.98, pole +0.58, bicycle +0.57**. Truck (+3.78), person (+0.32) and car (+0.25) exclude zero but do not survive the 19-class Holm (0.460 / 0.234 / 0.297); train (−2.54) is the single degradation excluding zero (Holm 0.264). Against the **control**, G is the **only arm of the 13** that improves traffic light (+0.47, CI excluding zero; the boundary/distmap arms all *degrade* it by −2.3 to −2.8), at the price of rider (−1.34, CI excluding zero; Holm(2) 0.0136 in §6.3).

### 6.3 Business metrics: pedestrian integrity paid

![F2 — The measured trade-off](figures/F2_tradeoff.png)

*Figure 2: the trade-off, both pairs. Left/middle: thin-object pixel coverage (Boundary F1, per-class IoU) goes up; right: pedestrian instance recall (strict, size strata, crowd) and pedestrian pixel precision go down. CIs from the paired image-bootstrap.*

Regenerated from the per-image npz artifacts (500 images × 3 seeds per arm; bit-identical to the program's cross-arm attribution artifacts on 19 shared metrics; ≤ 0.01 pt vs the primary harness forward on the shared mIoU row — 14/14 blocking sanity checks, `papers/paper4/tables/SANITY.md`). Units: points (×100), except fragments = components per image. **Bold = Holm(2) < 0.05** over this paper's two pairs; n = images contributing to the metric:

| Metric | n | G | B | ctrl | Δ(G−B) [95 % CI] | p | Holm(2) | Δ(G−ctrl) [95 % CI] | p | Holm(2) |
|---|---|---|---|---|---|---|---|---|---|---|
| mIoU | 500 | 81.26 | 81.09 | 81.17 | +0.173 [−0.229 ; +0.554] | 0.3952 | 0.7904 | +0.094 [−0.281 ; +0.466] | 0.6592 | 0.7904 |
| IoU person | 500 | 85.22 | 84.88 | 85.15 | **+0.344** [+0.092 ; +0.578] | 0.0088 | 0.0176 | +0.067 [−0.182 ; +0.302] | 0.5838 | 0.5838 |
| IoU rider | 500 | 67.58 | 68.49 | 68.92 | −0.911 [−2.003 ; +0.036] | 0.0612 | 0.0612 | **−1.339** [−2.413 ; −0.352] | 0.0068 | 0.0136 |
| Boundary F1 3 px | 500 | 77.00 | 76.37 | 76.63 | **+0.630** [+0.454 ; +0.806] | <0.0001 | <0.001 | **+0.376** [+0.216 ; +0.537] | <0.0001 | <0.001 |
| Boundary F1 3 px, ped | 500 | 67.73 | 67.30 | 67.49 | +0.431 [−0.119 ; +0.953] | 0.1204 | 0.2408 | +0.247 [−0.201 ; +0.702] | 0.2814 | 0.2814 |
| Ped. strict recall | 441 | 72.84 | 77.17 | 76.25 | **−4.335** [−4.948 ; −3.729] | <0.0001 | <0.001 | **−3.417** [−3.967 ; −2.868] | <0.0001 | <0.001 |
| Ped. pixel precision | 473 | 63.27 | 65.96 | 70.03 | **−2.699** [−3.484 ; −1.951] | <0.0001 | <0.001 | **−6.763** [−7.800 ; −5.785] | <0.0001 | <0.001 |
| All-instances recall | 441 | 76.55 | 80.99 | 79.98 | **−4.439** [−5.000 ; −3.884] | <0.0001 | <0.001 | **−3.423** [−3.911 ; −2.942] | <0.0001 | <0.001 |
| All-instances det@0.5 | 441 | 85.12 | 88.05 | 87.11 | **−2.936** [−3.817 ; −2.122] | <0.0001 | <0.001 | **−1.994** [−2.696 ; −1.328] | <0.0001 | <0.001 |
| Individual recall | 440 | 76.74 | 81.11 | 80.10 | **−4.376** [−4.935 ; −3.823] | <0.0001 | <0.001 | **−3.361** [−3.853 ; −2.866] | <0.0001 | <0.001 |
| Individual det@0.5 | 440 | 85.17 | 88.08 | 87.18 | **−2.903** [−3.819 ; −2.069] | <0.0001 | <0.001 | **−2.002** [−2.712 ; −1.302] | <0.0001 | <0.001 |
| Crowd recall | 63 | 69.97 | 75.71 | 74.58 | **−5.736** [−7.864 ; −4.033] | <0.0001 | <0.001 | **−4.612** [−6.531 ; −2.940] | <0.0001 | <0.001 |
| Crowd det@0.5 | 63 | 82.01 | 86.24 | 83.60 | **−4.233** [−7.937 ; −1.058] | 0.0046 | 0.0092 | −1.587 [−4.233 ; +0.529] | 0.2250 | 0.2250 |
| T1 (smallest) recall | 315 | 60.91 | 66.56 | 64.99 | **−5.645** [−6.650 ; −4.627] | <0.0001 | <0.001 | **−4.080** [−5.092 ; −3.091] | <0.0001 | <0.001 |
| T1 det@0.5 | 315 | 68.26 | 72.84 | 71.73 | **−4.574** [−6.218 ; −2.890] | <0.0001 | <0.001 | **−3.466** [−5.017 ; −1.960] | <0.0001 | <0.001 |
| T2 recall | 320 | 80.88 | 85.46 | 84.53 | **−4.584** [−5.316 ; −3.880] | <0.0001 | <0.001 | **−3.656** [−4.370 ; −2.938] | <0.0001 | <0.001 |
| T2 det@0.5 | 320 | 91.08 | 93.78 | 93.15 | **−2.695** [−3.893 ; −1.609] | <0.0001 | <0.001 | **−2.064** [−3.196 ; −0.928] | 0.0004 | 0.0004 |
| T3 (largest) recall | 335 | 91.68 | 93.85 | 93.47 | **−2.175** [−2.478 ; −1.875] | <0.0001 | <0.001 | **−1.798** [−2.090 ; −1.510] | <0.0001 | <0.001 |
| T3 det@0.5 | 335 | 98.78 | 98.96 | 98.96 | −0.180 [−0.697 ; +0.238] | 0.4654 | 0.4654 | −0.177 [−0.468 ; +0.054] | 0.1586 | 0.3172 |
| Fragments (comp./img) | 500 | 933.88 | 615.13 | 519.26 | **+318.8** [+305.4 ; +331.9] | <0.0001 | <0.001 | **+414.6** [+397.8 ; +431.1] | <0.0001 | <0.001 |

The trade-off in one sentence: the term **buys** contour quality (Boundary F1 +0.63 vs B, Holm-significant) and thin-object pixel coverage (§6.2), and **pays** in every pedestrian-instance integrity metric — strict recall −4.34, smallest tercile −5.65, crowd groups −5.74, pixel precision −2.70, detection@0.5 −2.94 (all vs paired B, Holm(2) < 0.001 except crowd det@0.5 at 0.0092). The payment is ordered by instance size: T1 −5.65 > T2 −4.58 > T3 −2.18 — **the smallest instances lose the most coverage**, the exact opposite of the term's intent. Person IoU does rise (+0.344, Holm(2)-significant vs B): the pixel *mass* of the person class grows while the *integrity* of individual pedestrian masks degrades — IoU and instance recall move in opposite directions, which is precisely why the mIoU-centric reading alone would have missed the effect.

### 6.4 Fragmentation: the mechanism the tables show

New per-class measurement of this paper (500 images × 3 seeds, arms G / B / control, 8-connected components per class per image, paired bootstrap + 19-class Holm; full table in Appendix B):

| Reading | Value (G vs B) |
|---|---|
| Total components per image | **933.9 vs 615.1 — ×1.52** (+318.8 [+305.4 ; +331.9], Holm < 0.001) |
| Classes with more components | **16 of 19** vs paired B (14 Holm-significant rises); **19 of 19** vs the control arm. The three that do not rise vs B: terrain **−1.8** (Holm < 0.001, the only material decrease), wall −0.02 (p = 0.95) and bus −0.02 (p = 0.70), both null |
| Largest absolute rises | car +49.4, pole +45.4, vegetation +43.8, road +38.6, traffic sign +38.4, building +32.5, sidewalk +30.8 |
| Pedestrian masks | person 24.3 → 53.2 components/image (**×2.2**); rider 2.0 → 4.1 |
| Thin classes that *gain* IoU | also gain components: traffic light +0.9, bicycle +2.2 (Holm-significant) |

**Revision note (house discipline: the paper reports what the tables show).** The program's initial working hypothesis — inherited from the BRATS intuition that an instance-balancing term suppresses small spurious components in favour of compact thin objects — is **contradicted** by this decomposition: components *increase* almost everywhere, including in the classes whose pixel IoU improves. The mechanism the measurements support is the opposite of compaction: in the exclusive-softmax regime, the equal-weight-per-instance gradient buys **dispersed pixel coverage** for thin and ample classes (per-class IoU ↑, Boundary F1 ↑) at the price of **holes inside ground-truth instances** — a holed mask splits into disjoint components (fragments ×1.52), which mechanically lowers instance-level recall (strict −4.34, T1 −5.65) and pedestrian pixel precision (−2.70). No gradient-level analysis is performed here; §8 lists this as a limitation.

### 6.5 Versatility ranking: a dominated specialist

The program's pre-registered versatility criterion: 36 endpoints (19 per-class IoU + 17 business metrics) × 13 arms, same holdout and bootstrap; "maximal damage" = the worst endpoint-level Δ vs control (closer to 0 = no weak point). G's row and its neighbours (full ranking, artifact `results/moe_v3_cs/p317_polyvalence/table_polyvalence.json`):

| Rank | Arm | Mean percentile | Worst rank | #1 | Max damage (pt) | z | Max peak (pt) | ΔmIoU (p / Holm) | Gains sig. (raw/Holm) | Losses sig. (raw/Holm) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MoE-V3-CS | 60.4 | 11 | 0 | **−0.53** (crowd det@0.5) | −0.35 | +4.24 (truck IoU) | +0.45 (0.0066 / 0.092) | 7/1 | 2/1 | non-dominated |
| 5 | B · CE+Dice | 51.2 | 13 | 2 | −4.06 (ped precision) | −0.63 | +2.65 (crowd det@0.5) | −0.08 (0.6592 / 1.000) | 11/8 | 6/6 | dominated |
| **8** | **G · CE+Dice+Blob (this paper)** | **41.9** | **13** | **7** | **−6.76** (ped precision) | **−1.05** | **+3.43** (truck IoU) | **+0.09 (0.6592 / 1.000)** | **4/1** | **13/12** | **dominated** |
| 11 | D · CE+Kervadec EDT | 77.3 | 13 | 16 | −22.20 (ped precision) | −3.44 | +5.49 (wall IoU) | +0.52 (0.0050 / 0.075) | 17/13 | 4/3 | non-dominated |

G's profile: mean percentile **41.9** (mean rank 8.2), worst rank **13/13**, last on 16 of 36 endpoints — yet **#1 on 7 of 36 endpoints**, more than any arm except D (16): a genuine specialist. The damage is concentrated on the pedestrian family (worst endpoint: pedestrian pixel precision **−6.76 pt**, z = −1.05 inter-arm σ; worst family mean: pedestrian −5.09; best family: contours +0.31). The **price per mIoU point** — maximal damage divided by ΔmIoU — is **−71.7 pt** for G against **−1.2 pt** for the MoE-V3-CS: the mixture buys the same plateau without the hole. G is **Pareto-dominated** (by B, C, C⊘B, Dp⊘B, the control and MoE-V3-CS). The pre-registered T0 verdict (significant ΔmIoU *and* no endpoint > 1 pt below reference *and* no class degraded > 0.5 pt) is reported **as computed: True for the mixture, and G is its antithesis** — a dominated specialist whose speciality (thin objects) is exactly what the mixture absorbs.

![F3 — G in the versatility ranking](figures/F3_polyvalence_G.png)

*Figure 3: position of arm G in the 36-endpoint × 13-arm ranking, against the mixture MoE-V3-CS and the plateau. Specialist peak (#1 on 7 endpoints) and maximal damage (−6.76 pt) on the same arm.*

### 6.6 Cross-dataset: the same verdict on BRATS

Numbers **cited** from the published BRATS program (second dataset, second probability regime — region-based sigmoid, MedNeXt, nnU-Net recipe, 5-fold CV over n = 1 196; sources verified 2026-09-30, not re-measured here):

| BRATS arm | Result | Source |
|---|---|---|
| blob **alone** vs nnU-Net baseline | **−0.00376 Dice** (5-fold CV; 0/5 folds won; last of the 4 expert arms) | BRATS `papers/paper3/versions/v3/NOTE_DECISION_V3.md` l.46 + `tables/v3_vs_consensus.md` l.38 |
| blob as **expert 3 of the winning MoE-V3** | gate V3: **+0.00566 Dice** vs its own baseline, **1st of 29 arms** (5/5 folds) | same note l.52, l.115; [DOI 10.5281/zenodo.22903668] |

Two independent programs — different datasets, architectures, metrics and probability regimes — converge on the same verdict: **null-or-negative alone, first-of-the-field as an initialised expert of a mixture**. That convergence is this paper's generality argument.

---

## 7. Discussion

### 7.1 Why equal weight per instance does not pay alone, in this regime

The measurements support a coherent reading, offered as a hypothesis grounded in the tables (no gradient-level evidence is collected here). Under **exclusive softmax**, every probability mass given to class $c$ inside instance $n$ is taken from the other 18 classes at the same pixel. The per-instance term weights each instance's *presence* equally, so its gradient keeps pushing class-c mass onto small instances long after the region-wise losses (CE-ISNS, Dice) have settled — mass that lands as **dispersed coverage**: enough scattered correct pixels to raise thin-class IoU (+0.98 traffic light) and contour F1 (+0.63), not enough to keep small masks **connected**. The per-class fragment decomposition (§6.4) shows the cost: holes multiply inside ground-truth instances of 16 classes of 19 vs the paired arm B — of 19 of 19 vs the control (person ×2.2), and a holed mask loses instance recall (−4.34 strict), small-instance recall first (T1 −5.65 > T2 −4.58 > T3 −2.18) and pedestrian pixel precision (−2.70). The class-level mIoU integrates the scattered gains and the scattered losses into a **null** (+0.170, p = 0.4032): the term redistributes correctness, it does not create it. On BRATS (sigmoid, independent channels), the same term was also null alone — the competition regime is not the whole story, but the *usage* verdict replicated exactly.

### 7.2 Where it pays: an initialised expert of a mixture

The companion Cityscapes mixture **MoE-V3-CS** initialises four experts from the independently trained arms B, D, Dp, **G** and trains a top-2 gate over 3×3 patches for 80 epochs (control: CE+Dice 80 epochs, B-initialised). Measured in the companion study (cited here, artifacts of the same program chain): ΔmIoU **+0.45 pt [+0.11 ; +0.82], p = 0.0066** vs control; small-instance recall **T1 +0.80 (significant)**; traffic light **not degraded** (the boundary/distmap arms lose 2.3–2.8 on it); maximal damage **−0.53 pt**; fragments 488/image — *below* the control's 519. In other words, the gate routes the thin-object speciality of G to the patches where it wins, and routes away from G where its hole-making behaviour would cost — the speciality is absorbed, the damage is not inherited. The BRATS MoE-V3 did the same with the same expert (1st of 29 arms, 5/5 folds). **The blob loss is a good expert and a bad generalist** — and mixture-of-experts training is the mechanism that converts one into the other.

### 7.3 Practical takeaways

* **Do not ship the blob term alone** in exclusive-softmax semantic segmentation at fixed budget: expect a null mIoU, better thin-object pixel IoU, and degraded pedestrian instance integrity. If pedestrian *detection* feeds a downstream planner, the −4.3 strict-recall points are disqualifying on their own.
* **Do consider it as an expert candidate** for a patch-wise mixture: it is the only arm of the 13 that improves traffic light vs the control, and its speciality is exactly what the measured mixture absorbs.
* **If you keep it, pair it with a CC filter.** The fragmentation diagnosis (+318.8 components/image) says a post-hoc connected-component consensus veto — the mIoU-neutral C⊘B of [Cassez 2026b], −18.7 % fragments — targets precisely this arm's failure mode. (Reported as a design implication; the G⊘-style veto on this arm is not measured in this paper.)
* **The engineering cost is solved**: precomputed instance packs make the term +1–2 %/epoch (against +870 s/epoch naive), parity-tested, with the flip-alignment pitfall documented (§4).

### 7.4 Relation to Kofler et al.

Nothing here contradicts the published medical result: under the region-based sigmoid recipe the term improved the small-structure metrics it targets [Kofler 2023]. What this paper adds is the **regime boundary**: transposed *exactly* (same algebra, same β = 0.5, same two-level mean) to exclusive-softmax full-resolution Cityscapes, the term's promise inverts into the trade-off above. Loss-term transfers across probability regimes should be measured, not assumed — on both of our datasets the *alone* verdict was null while the *expert* verdict was first-of-the-field.

---

## 8. Limitations

1. **The primary endpoint is null.** This is a negative-result paper by design; the contribution is the diagnostic, the port and the usage — not a performance claim.
2. **β not swept.** β = 0.5 fixed at Kofler's default (α:β = 2:1); no λ sweep, same fixed-budget stance as the program's companion papers. The trade-off could move with β; that surface is not measured.
3. **Holdout `first:500`.** A pre-specified subset of val, identical for all 13 arms and fixed before training — but not the official leaderboard, which was not submitted to.
4. **Holm leaves no arm significant on mIoU in either multiplicity family** (best raw p = 0.0048 → Holm 0.058 over the 12-pair master-table family, 0.075 over the program's 15-pair exploratory family). Program-wide conclusions on mIoU rest on amplitudes and CIs, stated as such.
5. **Budget mismatch with the 13-arm context.** G trains 160 epochs; the context control 80. The paired comparison of G is B (160 epochs); the control pair is always labelled *context*.
6. **Mechanism is a measurement-grounded hypothesis.** The fragmentation decomposition is new and blocking-checked, but no gradient-level or per-layer analysis is performed; §7.1 remains an interpretation.
7. **n = 3 seeds.** Sub-0.5-pt effects are underpowered at the seed level; the primary inference is the 500-image paired bootstrap, which probes evaluation-set sampling, not seed variance.
8. **cuDNN/BF16 non-determinism declared.** Two independent forwards of the same checkpoints differ by ≤ 0.01 pt (verdicts identical); bit-exact reruns are not claimed anywhere in the program.
9. **One architecture, two datasets.** ConvNeXt-V2-Base + UPerNet only; Cityscapes + BRATS only. Other heads (Mask2Former-style mask attention re-weights boundaries internally) may react differently.
10. **Instance packs not shipped** (heavy); regenerated bit-identical by the released script, parity verified over all 2 975 packs on 2026-09-26.

---

## 9. Conclusion

Given every ground-truth instance the same weight, whatever its pixel count, and full-resolution exclusive-softmax Cityscapes answers: **not worth it alone**. The pre-registered primary endpoint is null (ΔmIoU +0.170 pt [−0.233 ; +0.551], p = 0.4032, G vs paired B, 160 epochs, 3 seeds). The term buys exactly what it promises — thin-object pixel coverage (traffic light +0.98, pole +0.58, bicycle +0.57, Holm-significant; Boundary F1 +0.63) — and pays for it in pedestrian instance integrity (strict recall −4.34, smallest tercile −5.65, crowd −5.74) and in mask topology (components ×1.52, 16 classes of 19 up vs paired B and 19 of 19 vs the control, person masks ×2.2), which leaves it a **Pareto-dominated specialist** on the program's 36-endpoint versatility criterion. The same term, as an initialised expert of a patch-wise mixture, is **first of 29 arms on BRATS** and part of the **only non-dominated, significantly-positive arm on Cityscapes** (MoE-V3-CS, ΔmIoU +0.45, p = 0.0066, maximal damage −0.53). Equal weight per instance does not pay alone — it pays as an expert.

Code, configs, per-arm artifacts pointers, tables, figures and regeneration scripts: **github.com/guillaume-cassez/cityscape-blob-loss-kofler**. Program companions: Cityscapes distmap [DOI 10.5281/zenodo.21006236], Cityscapes boundary ablation [DOI 10.5281/zenodo.21006393], BRATS MoE-V3 [DOI 10.5281/zenodo.22903668], BRATS boundary [DOI 10.5281/zenodo.22906447], BRATS consensus [DOI 10.5281/zenodo.22904810], BRATS distmap [DOI 10.5281/zenodo.20110976].

---

## Appendix A — Runtime and reproducibility

\begin{table}[H]
\centering
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{@{}p{4.6cm}p{3.6cm}p{1.6cm}p{5.2cm}@{}}
\toprule
\textbf{Stage} & \textbf{Hardware} & \textbf{Time} & \textbf{Output} \\
\midrule
Instance-pack precomputation (one-shot, train split)
& 8 P-cores, cc3d
& 22.4 ms/img
& \texttt{.blob.npz} per image \newline (\texttt{blob\_glob/counts/csr}) \\
\addlinespace
Training 160 ep, 1 seed (arm G)
& 1 \(\times\) RTX PRO 6000 \newline 96 GB Blackwell
& 29.2 h \newline (656 s/ep, 31.5 GB)
& \texttt{checkpoints/pilot\_fullres\_G\_blob\_seed\{s\}/epoch\_160.pth} \\
\addlinespace
Arm G total (3 seeds)
& same
& \(\approx\) 87 h GPU
& final train loss 0.3743 / 0.3735 / 0.3765 \\
\addlinespace
Primary bootstrap (harness)
& CPU
& minutes
& \texttt{results/moe\_v3\_cs/harness/table\_Gseul\_P310.json} \\
\addlinespace
Business metrics + fragments (2 arms \(\times\) 3 seeds)
& 1 GPU + 6 workers
& \(\approx\) 2.6 h (12-arm job)
& \texttt{metriques\_G\_seed*.npz}, \texttt{frag\_*\_seed*.npy} \\
\addlinespace
Tables + figures of this paper
& CPU
& minutes
& \texttt{papers/paper4/\{tables,figures\}/} \\
\bottomrule
\end{tabular}
\end{table}

**Reproducibility seeds.** Training seeds 42, 123, 456 are set globally (PyTorch, NumPy, Python `random`, CUDA). The bootstrap seed is fixed (20260618, B = 10 000) and shared by every table of the program, so all CIs and p-values are exactly re-derivable from the released per-image artifacts. cuDNN benchmark is left **on** (`deterministic: false`) for training speed; bit-exact GPU reruns are therefore not claimed — the declared reproducibility statement is: same artifacts → same statistics bit-for-bit (verified), and independent forwards of the same checkpoints → agreement within 0.01 pt (measured, 14/14 blocking sanity checks).

---

## Appendix B — Per-class fragmentation, G vs B (components per image)

8-connected components per class per image, 500 images × 3 seeds, paired bootstrap B = 10 000 (seed 20260618), Holm over the 19-class family. Bold = Holm < 0.05. New measurement of this paper (the program had fragments only at arm level before it).

| Class | G | B | ctrl | Δ(G−B) | 95 % CI | p | Holm(19) |
|---|---|---|---|---|---|---|---|
| car | 117.3 | 67.9 | 57.4 | **+49.4** | [+46.5 ; +52.2] | <0.0001 | <0.001 |
| pole | 176.4 | 131.1 | 115.9 | **+45.4** | [+42.9 ; +47.8] | <0.0001 | <0.001 |
| vegetation | 107.7 | 63.8 | 58.8 | **+43.8** | [+41.6 ; +46.1] | <0.0001 | <0.001 |
| road | 82.4 | 43.8 | 47.1 | **+38.6** | [+36.3 ; +40.9] | <0.0001 | <0.001 |
| traffic sign | 92.3 | 53.9 | 43.2 | **+38.4** | [+36.1 ; +40.7] | <0.0001 | <0.001 |
| building | 116.9 | 84.4 | 75.4 | **+32.5** | [+29.6 ; +35.4] | <0.0001 | <0.001 |
| sidewalk | 106.2 | 75.4 | 64.6 | **+30.8** | [+28.7 ; +33.0] | <0.0001 | <0.001 |
| person | 53.2 | 24.3 | 16.4 | **+28.9** | [+27.3 ; +30.5] | <0.0001 | <0.001 |
| sky | 27.7 | 23.2 | 11.9 | **+4.6** | [+3.2 ; +5.9] | <0.0001 | <0.001 |
| fence | 10.9 | 8.2 | 4.9 | **+2.7** | [+2.1 ; +3.4] | <0.0001 | <0.001 |
| bicycle | 11.8 | 9.6 | 6.2 | **+2.2** | [+1.7 ; +2.7] | <0.0001 | <0.001 |
| rider | 4.1 | 2.0 | 1.9 | **+2.2** | [+2.0 ; +2.4] | <0.0001 | <0.001 |
| traffic light | 6.9 | 6.0 | 4.6 | **+0.9** | [+0.6 ; +1.1] | <0.0001 | <0.001 |
| truck | 0.6 | 0.3 | 0.3 | **+0.2** | [+0.1 ; +0.4] | <0.0001 | <0.001 |
| motorcycle | 0.7 | 0.6 | 0.6 | +0.0 | [−0.0 ; +0.1] | 0.4940 | 1.000 |
| train | 0.3 | 0.2 | 0.2 | +0.0 | [−0.0 ; +0.0] | 0.3292 | 1.000 |
| bus | 0.5 | 0.5 | 0.4 | −0.0 | [−0.1 ; +0.1] | 0.7028 | 1.000 |
| wall | 7.2 | 7.2 | 4.0 | −0.0 | [−0.5 ; +0.5] | 0.9496 | 1.000 |
| terrain | 10.9 | 12.8 | 5.5 | **−1.8** | [−2.6 ; −1.1] | <0.0001 | <0.001 |

Sanity: the per-class sums equal the arm-level component counts **bit-for-bit** (G 933.9, B 615.1, control 519.3 per image). Source: `results/moe_v3_cs/paper4_blob/table_frag_classes_G_vs_B.json`.

---

\newpage

## References

* Cordts *et al.* (2016). *The Cityscapes dataset for semantic urban scene understanding*. CVPR.
* Cassez (2026a). *Distance-Map Auxiliary Regression for Full-Resolution Cityscapes Segmentation*. Zenodo. DOI 10.5281/zenodo.21006236.
* Cassez (2026b). *Boundary Loss Ablation for Full-Resolution Cityscapes Segmentation: When Dice Helps and When It Doesn't*. Zenodo. DOI 10.5281/zenodo.21006393.
* Cassez & Larnier (2026c). *The Gate Does Not Choose: An Expert-Initialised Mixture-of-Experts Outperforms 24 Arms on BRATS 2023*. Zenodo. DOI 10.5281/zenodo.22903668.
* Cassez & Larnier (2026d). *Precision Pays: A Fixed-Weight Boundary Loss Anchors the Precision End* (BRATS). Zenodo. DOI 10.5281/zenodo.22906447.
* Cassez & Larnier (2026e). *Two Models That Agree Beat the Best of Them Alone: Parameter-Free Connected-Component Consensus* (BRATS). Zenodo. DOI 10.5281/zenodo.22904810.
* Cassez (2025). *Distance Map Auxiliary Loss for Brain Tumor Segmentation: A Fragment-Centric Study*. Zenodo. DOI 10.5281/zenodo.20110976.
* Isensee *et al.* (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods 18.
* Kervadec *et al.* (2019). *Boundary loss for highly unbalanced segmentation*. MIDL. arXiv:1812.07032.
* Kofler *et al.* (2023). *Blob loss for biomedical image segmentation*. IPMI. arXiv:2205.08209.
* Lin *et al.* (2017). *Focal loss for dense object detection*. ICCV.
* Milletari *et al.* (2016). *V-Net: fully convolutional neural networks for volumetric medical image segmentation*. 3DV.
* Perazzi *et al.* (2016). *A benchmark dataset and evaluation methodology for video object segmentation*. CVPR.
* Silversmith *et al.* (2019). *cc3d: connected-components labeling for large 3D arrays* (software). github.com/seung-lab/connected-components-3d.
* Woo *et al.* (2023). *ConvNeXt V2: co-designing and scaling ConvNets with masked autoencoders*. CVPR. arXiv:2301.00808.
* Xiao *et al.* (2018). *Unified perceptual parsing for scene understanding*. ECCV. arXiv:1807.10221.

---

*Manuscript — 2026-09-30. Source code, configs, tables, figures and regeneration scripts: github.com/guillaume-cassez/cityscape-blob-loss-kofler. Author: Guillaume Cassez, independent researcher (ORCID 0009-0007-0987-3931), guillaume-cassez.fr — currently looking for ML / computer vision engineering opportunities.*
