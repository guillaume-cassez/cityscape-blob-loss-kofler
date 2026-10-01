#!/usr/bin/env python3
"""Pré-calcule les paquetages d'instances BlobLoss (.blob.npz) pour Cityscapes train.

Même motif que scripts/precompute_sdt.py : les labels d'instances ne dépendent que du
champ de cibles (l'unique augmentation géométrique full_res est le flip horizontal,
label-préservant — appliqué par le dataset au chargement), donc un pré-calcul unique est
exact. Écriture durable : tmp + rename + fsync fichier et dossier (règle du poste).

Coût mesuré (Tour, 2026-09-17) : ≈ 25-40 ms/image (cc3d 22.4 ms + paquetage) → ≈ 2 min
pour 2975 images en séquentiel, ~12 Go sur SSD (int16 2M px/image).

Usage (Tour, repo root) :
    /home/ser/brats-venv/bin/python scripts/precompute_blob_lab.py [--force] [--limit N]
    # --force : recalcule même si le .npz existe déjà
    # --limit : n premières images (smoke)
Vérification intégrée en fin de run : rechargement de 10 paquetages au hasard +
re-calcul complet de 3 d'entre eux (comparaison octet à octet).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.cityscapes_dataset import map_labels_to_trainids  # noqa: E402
from src.losses.blob_lab import blob_lab_path, compute_blob_lab, save_blob_lab  # noqa: E402

N_CLASSES = 19
IGNORE = 255


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "data/cityscapes"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image

    label_dir = Path(args.root) / "gtFine" / args.split
    paths = sorted(label_dir.glob("*/*_gtFine_labelIds.png"))
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} labels ({args.root}/{args.split})")

    t_start = time.time()
    done = skipped = 0
    max_n = 0
    hist = []
    for i, png in enumerate(paths):
        out = blob_lab_path(png)
        if out.exists() and not args.force:
            skipped += 1
            continue
        lab = np.array(Image.open(png))
        t_np = map_labels_to_trainids(lab).astype(np.int64)
        glob, csr, counts = compute_blob_lab(t_np, N_CLASSES, IGNORE)
        save_blob_lab(out, glob, csr, counts)
        max_n = max(max_n, int(csr[N_CLASSES]))
        hist.append(int(csr[N_CLASSES]))
        done += 1
        if done % 250 == 0:
            el = time.time() - t_start
            print(f"  {done}/{len(paths) - skipped} écrits — {el / done * 1e3:.0f} ms/img — max instances {max_n}")

    el = time.time() - t_start
    print(f"FIN : {done} écrits, {skipped} existants ignorés, {el:.0f} s — max instances/image {max_n}")
    if hist:
        print("instances/image : p50=%d p99=%d max=%d (garde int16 32766, garde counts_pad 8192 => %s)" % (
            int(np.percentile(hist, 50)), int(np.percentile(hist, 99)), max_n,
            "OK" if max_n + 2 <= 8192 else "AUGMENTER blob_counts_pad"))

    # ------------------------------------------------- vérification intégrée
    exists = [p for p in paths if blob_lab_path(p).exists()]
    sample = random.sample(exists, min(10, len(exists)))
    for png in sample:  # rechargement lisible
        with np.load(blob_lab_path(png), allow_pickle=False) as z:
            assert z["glob"].dtype == np.int16 and z["glob"].ndim == 2
            assert z["csr"].shape == (N_CLASSES + 1,) and z["counts"].ndim == 1
    for png in sample[:3]:  # re-calcul bit-à-bit
        lab = np.array(Image.open(png))
        t_np = map_labels_to_trainids(lab).astype(np.int64)
        g2, c2, k2 = compute_blob_lab(t_np, N_CLASSES, IGNORE)
        with np.load(blob_lab_path(png), allow_pickle=False) as z:
            assert np.array_equal(z["glob"], g2), f"glob diverge : {png}"
            assert np.array_equal(z["csr"], c2.astype(np.int64)), f"csr diverge : {png}"
            assert np.array_equal(z["counts"], k2), f"counts diverge : {png}"
    print(f"VÉRIF : {len(sample)} rechargés, 3 re-calculés bit-à-bit — OK")


if __name__ == "__main__":
    main()
