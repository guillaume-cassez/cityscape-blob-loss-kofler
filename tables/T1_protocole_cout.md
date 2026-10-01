# T1 — Protocole et coût (bras G, blob loss seul)

| Poste | Valeur | Source |
|---|---|---|
| Architecture | ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, pleine résolution 1024×2048, 19 classes, BF16 channels_last | `configs/experiment/pilot_fullres_G_blob.yaml` |
| Loss G | CE 0.5 (class_weights ISNS, ignore 255) + Dice 0.5 (smooth 1.0) + **blob Kofler 0.5** (eps 1.0, min_blob_pixels 0, ignore 255) | `configs/loss/ce_dice_blob.yaml` |
| Référence B | MÊME recette, loss ce_dice (CE 0,5 + Dice 0,5) — seule la loss change | `configs/loss/ce_dice.yaml` |
| Entraînement | 160 époques, batch 2×accum 4 (effectif 8), AdamW lr 6e-05 wd 0.01, poly power 1.0, warmup 1 ep, bf16 | `idem` |
| Augmentation | hflip p=0,5 appliqué par le DATASET sur image+label+paquetage d'instances (flip albumentations désactivé — alignement du glob blob) ; même distribution que B | `config G + src/losses/blob_lab.py` |
| Seeds | 42, 123, 456 (3 seeds, résultats moyennés dans chaque réplicat bootstrap) | `résultat/p3_queue` |
| Coût seed 42 | 160/160 époques, 656 s/ep médian [655-666], VRAM 31.5 Go, train_loss final 0.3743, total 29.2 h | `results/p3_queue/pilot_fullres_G_blob_seed42.log` |
| Coût seed 123 | 160/160 époques, 656 s/ep médian [655-669], VRAM 31.5 Go, train_loss final 0.3735, total 29.2 h | `results/p3_queue/pilot_fullres_G_blob_seed123.log` |
| Coût seed 456 | 160/160 époques, 656 s/ep médian [655-669], VRAM 31.5 Go, train_loss final 0.3765, total 29.2 h | `results/p3_queue/pilot_fullres_G_blob_seed456.log` |
| Coût du bras G | ≈ 87 h GPU (3 seeds × 160 époques) | `somme des logs` |
| Pré-calcul blob | paquetages .blob.npz (blob_glob/blob_counts/blob_csr) : +1 à 2 %/époque contre +870 s/époque pour le chemin naïf (CC CPU + 26 syncs/classe) ; cc3d 8-connexité 22,4 ms/image pleine résolution (16 classes, 118 instances) vs 59,7 ms scipy | `src/losses/blob_loss.py docstring, mesures 2026-09-17` |
| Parité | naïf ↔ pré-calculé verrouillée par tests/test_blob_loss.py ; 2975 paquetages régénérés bit-à-bit (vérifié 2026-09-26, commit 132f2e8) ; paquetages absents du disque au 2026-09-30, régénérables par scripts/precompute_blob_lab.py | `DEVLOG + commit 132f2e8` |
| Évaluation | holdout partagé first:500 de Cityscapes val, mIoU dataset-level cityscapesScripts (convention bit-identique, tests/test_official_miou.py), bootstrap apparié B=10 000 seed 20260618 | `src/moe/bootstrap.py` |
