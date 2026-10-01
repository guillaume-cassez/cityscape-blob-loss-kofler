#!/usr/bin/env python3
"""P4.02 — Tables & figures dédiées du paper 4 (blob loss seul, bras G), SANS GPU.

Règle du cadrage (papers/paper4/CADRAGE.md §4) : aucun ✅ de synthèse recopié sans son
Holm — les statistiques métier des paires du papier sont RÉGÉNÉRÉES ici depuis les npz
par image, avec le protocole EXACT du programme (P3.10 → P3.17) :
  · holdout partagé first:500, seeds 42/123/456 moyennés dans chaque réplicat ;
  · bootstrap APPARIÉ par image B=10 000, `bootstrap_seed` 20260618 (mêmes index-set
    pour tous les bras) ; mIoU/IoU dataset-level ré-agrégés depuis les confusions ;
  · masques de finiteur identiques à P3.16 (stack des 12 bras à couverture complète) ;
  · p bilatéral, Holm sur la famille déclarée ICI = les 2 paires du papier
    (G_vs_B appariée 160 époques = primaire ; G_vs_controle = contexte 13 bras) ;
  · le Holm de la famille exploratoire 15 paires de P3.16 est reporté en colonne de
    cross-référence (jamais remplacé en silence).

Sanités OBLIGATOIRES (écrites dans SANITY.md, échec = sortie non nulle) :
  1. G_vs_controle régénéré == table P3.16 (delta/IC/p bit-à-bit, mêmes index bootstrap) ;
  2. mIoU G/B == harness P3.10 `table_Gseul_P310.json` (primaire pré-enregistré) ;
  3. G_vs_controle mIoU == master table P3.14 ;
  4. mIoU par seed == harness ;
  5. points fragments controle/moe == P3.16.

Productions :
  results/moe_v3_cs/paper4_blob/table_metiers_G_pairees.json   (bootstraps bruts)
  results/moe_v3_cs/paper4_blob/table_frag_classes_G_vs_B.json (diagnostic fragmentation)
  results/moe_v3_cs/paper4_blob/frag_classes_*_seed*.npy       (décomposition par classe)
  results/moe_v3_cs/paper4_blob/SANITY.json
  results/moe_v3_cs/metiers_experts/frag_{G,B}_seed*.npy       (fragments manquants)
  papers/paper4/tables/T1..T6 (.md + .csv) + paper4_tables.json + SANITY.md
  papers/paper4/figures/F1_forest_perclass_GvsB.{png,pdf}
  papers/paper4/figures/F2_tradeoff.{png,pdf}
  papers/paper4/figures/F3_polyvalence_G.{png,pdf}
  discord_out/ (copies png des 3 figures)

Usage (Tour, CPU) :
  nohup /home/ser/brats-venv/bin/python scripts/p4_blob_tables_figures.py --workers 6 \
      > results/p3_queue/p4_blob_tables.log 2>&1 &
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import p3_metiers_eval as M  # noqa: E402
from src.moe.bootstrap import (  # noqa: E402
    paired_bootstrap, paired_bootstrap_values, holm,
    DEFAULT_B, DEFAULT_BOOTSTRAP_SEED,
)
from src.postprocessing.consensus import count_fragments  # noqa: E402

SEEDS = (42, 123, 456)
N_POS = 500
PAIRS = [("G", "B"), ("G", "controle")]          # primaire appariée + contexte 13 bras
ARMS_TABLE = ["controle", "B", "C", "Cp", "D", "Dp", "G",
              "fused_CvetoB", "fused_CpvetoB", "fused_DvetoB", "fused_DpvetoB",
              "moe_v3cs"]                        # = P3.16 (masques de finiteur identiques)

MET_DIR = REPO / "results/moe_v3_cs/metiers_experts"
METIERS_DIR = REPO / "results/moe_v3_cs/metiers"
PREDS_DIR = Path("/mnt/data8t/cityscape_p316")
HARNESS = REPO / "results/moe_v3_cs/harness/table_Gseul_P310.json"
P314 = REPO / "results/moe_v3_cs/p314/master_table.json"
P316_MET = REPO / "results/moe_v3_cs/metiers_experts/table_metiers_experts.json"
P316_ATT_B = REPO / "results/moe_v3_cs/p316/attribution_perclass_vs_B.json"
P316_ATT_C = REPO / "results/moe_v3_cs/p316/attribution_perclass_vs_controle.json"
P317 = REPO / "results/moe_v3_cs/p317_polyvalence/table_polyvalence.json"
G_LOGS = {s: REPO / f"results/p3_queue/pilot_fullres_G_blob_seed{s}.log" for s in SEEDS}

OUT_RAW = REPO / "results/moe_v3_cs/paper4_blob"
OUT_TABLES = REPO / "papers/paper4/tables"
OUT_FIGS = REPO / "papers/paper4/figures"
OUT_DISCORD = REPO / "discord_out"

CLASSES_19 = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
              "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
              "truck", "bus", "train", "motorcycle", "bicycle"]
CL_FR = {"road": "route", "sidewalk": "trottoir", "building": "bâtiment", "wall": "mur",
         "fence": "clôture", "pole": "poteau", "traffic light": "feu tricolore",
         "traffic sign": "panneau", "vegetation": "végétation", "terrain": "terrain",
         "sky": "ciel", "person": "piéton", "rider": "cavalier", "car": "voiture",
         "truck": "camion", "bus": "bus", "train": "train", "motorcycle": "moto",
         "bicycle": "vélo"}

LABEL_BRAS = {"B": "B · CE+Dice", "C": "C · CE+Dice+EDT", "Cp": "Cp · CE+Dice+SDT",
              "D": "D · CE+EDT", "Dp": "Dp · CE+SDT", "G": "G · CE+Dice+Blob",
              "fused_CvetoB": "consensus C⊘B", "fused_CpvetoB": "consensus Cp⊘B",
              "fused_DvetoB": "consensus D⊘B", "fused_DpvetoB": "consensus Dp⊘B",
              "moe_v3cs": "MoE-V3-CS (4 experts)", "controle": "contrôle (réf)",
              "A": "A · CE seule"}
LABEL_COURT = {"B": "B", "C": "C", "Cp": "Cp", "D": "D", "Dp": "Dp", "G": "G",
               "fused_CvetoB": "C⊘B", "fused_CpvetoB": "Cp⊘B",
               "fused_DvetoB": "D⊘B", "fused_DpvetoB": "Dp⊘B",
               "moe_v3cs": "MoE-V3-CS", "controle": "contrôle", "A": "A"}

METIERS_ORDER = ["mIoU", "IoU_person", "IoU_rider",
                 "boundary_f1_3px", "boundary_f1_3px_pieds",
                 "rappel_strict_instances", "precision_ped_pixels",
                 "instances_toutes_rappel", "instances_toutes_det05",
                 "instances_individuelles_rappel", "instances_individuelles_det05",
                 "instances_foule_rappel", "instances_foule_det05",
                 "instances_taille_T1_rappel", "instances_taille_T1_det05",
                 "instances_taille_T2_rappel", "instances_taille_T2_det05",
                 "instances_taille_T3_rappel", "instances_taille_T3_det05",
                 "fragments"]

SANITY: list[dict] = []


def log(msg):
    print(f"[p4 {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ecrire_atomique(path: Path, contenu: str | bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    mode = "wb" if isinstance(contenu, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(contenu)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def check(nom: str, ok: bool, detail: str):
    SANITY.append({"check": nom, "ok": bool(ok), "detail": detail})
    log(f"  [sanity {'✅' if ok else '❌'}] {nom} — {detail}")
    if not ok:
        raise SystemExit(f"SANITY ÉCHOUÉE : {nom} — {detail}")


# --------------------------------------------------------------------------- #
# Fragments G/B (manquants) — même code que p316.fragments_task
# --------------------------------------------------------------------------- #

def fragments_task(task):
    arm, seed = task
    dst = MET_DIR / f"frag_{arm}_seed{seed}.npy"
    if dst.exists():
        return str(dst)
    preds = np.load(PREDS_DIR / f"pred_{arm}_seed{seed}.npy", mmap_mode="r")
    fr = np.zeros(N_POS, dtype=np.int64)
    t0 = time.time()
    for i in range(N_POS):
        fr[i] = sum(count_fragments(np.asarray(preds[i])).values())
    tmp = dst.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, fr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    log(f"  [frag] {arm} seed {seed} : mean={fr.mean():.1f} ({time.time()-t0:.0f}s)")
    return str(dst)


def pred_path(arm: str, seed: int) -> Path:
    if arm == "controle":
        return METIERS_DIR / f"pred_{arm}_seed{seed}.npy"
    return PREDS_DIR / f"pred_{arm}_seed{seed}.npy"


def frag_classes_task(task):
    """Décomposition PAR CLASSE du nombre de composantes connexes (diagnostic neuf du
    paper 4 : quelles classes le terme blob compacte / quelles classes il mouchette)."""
    arm, seed = task
    dst = OUT_RAW / f"frag_classes_{arm}_seed{seed}.npy"
    if dst.exists():
        return str(dst)
    preds = np.load(pred_path(arm, seed), mmap_mode="r")
    fc = np.zeros((N_POS, 19), dtype=np.int64)
    t0 = time.time()
    for i in range(N_POS):
        for c, n in count_fragments(np.asarray(preds[i])).items():
            fc[i, c] = n
    tmp = dst.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, fc)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    log(f"  [frag-classes] {arm} seed {seed} : total/image={fc.sum(1).mean():.1f} ({time.time()-t0:.0f}s)")
    return str(dst)


def frag_classes_bootstrap(B: int, rng_seed: int) -> tuple[dict, dict]:
    """Bootstrap apparié G vs B du nombre de fragments PAR CLASSE (protocole identique).
    Sanity bloquante : la somme par classe reproduit bit-à-bit les frag_*.npy totaux."""
    stacks = {}
    for a in ("G", "B", "controle"):
        rows = [np.load(OUT_RAW / f"frag_classes_{a}_seed{s}.npy").astype(np.float64)
                for s in SEEDS]
        stacks[a] = np.stack(rows)                       # (3, 500, 19)
        tot = stacks[a].sum(-1).astype(np.int64)
        ref = np.stack([np.load(MET_DIR / f"frag_{a}_seed{s}.npy").astype(np.int64)
                        for s in SEEDS])
        check(f"frag_classes somme == frag npy ({a})", np.array_equal(tot, ref),
              f"({a}) somme des 19 classes bit-à-bit égale au count total "
              f"(mean {tot.mean():.1f}/image)")
    tables = {}
    for c, nom in enumerate(CLASSES_19):
        vals = {a: stacks[a][..., c] for a in stacks}
        tables[nom] = paired_bootstrap_values(vals, B=B, rng_seed=rng_seed,
                                              pairs=[("G", "B")])
    return tables, {a: stacks[a].mean(axis=(0, 1)) for a in stacks}


# --------------------------------------------------------------------------- #
# Chargement des stacks (identique p316 phase D)
# --------------------------------------------------------------------------- #

def load_met_stack(arm: str, key: str):
    rows = []
    for s in SEEDS:
        d = np.load(MET_DIR / f"metriques_{arm}_seed{s}.npz")
        rows.append(np.asarray(d[key], dtype=np.float64))
    return np.stack(rows)


def bootstraps_paires(B: int, rng_seed: int) -> dict:
    """Toutes les métriques métier, paires G_vs_B + G_vs_controle, protocole P3.16."""
    cms = {a: load_met_stack(a, "cm").astype(np.int64) for a in ("G", "B", "controle")}
    tables = {}
    log("  [boot] mIoU (3 bras, cm dataset-level)")
    tables["mIoU"] = paired_bootstrap(cms, B=B, rng_seed=rng_seed, pairs=PAIRS)
    for classe, nom in ((11, "IoU_person"), (12, "IoU_rider")):
        log(f"  [boot] {nom}")
        tables[nom] = M.boot_iou_classe(cms, classe, B, rng_seed, PAIRS)
    for key, nom in (("bf1", "boundary_f1_3px"), ("bf1_ped", "boundary_f1_3px_pieds")):
        log(f"  [boot] {nom}")
        vals = {a: load_met_stack(a, key) for a in ("G", "B", "controle")}
        tables[nom] = paired_bootstrap_values(vals, B=B, rng_seed=rng_seed, pairs=PAIRS)
    # métriques masquées : stack des 12 bras (masque de finiteur IDENTIQUE à P3.16)
    stacks12 = {}
    for key, nom in (("rec_strict", "rappel_strict_instances"),
                     ("prec_ped", "precision_ped_pixels")):
        log(f"  [boot] {nom} (masque 12 bras)")
        stacks12[nom] = {a: load_met_stack(a, key) for a in ARMS_TABLE}
        tables[nom] = M.boot_values_masque(stacks12[nom], B, rng_seed, PAIRS, nom)
    strat_rec = {st: {a: [] for a in ARMS_TABLE} for st in M.STRATES}
    strat_det = {st: {a: [] for a in ARMS_TABLE} for st in M.STRATES}
    for a in ARMS_TABLE:
        for s in SEEDS:
            d = np.load(MET_DIR / f"metriques_{a}_seed{s}.npz")
            r_, d_ = np.asarray(d["strat_rec"]), np.asarray(d["strat_det"])
            for si, st in enumerate(M.STRATES):
                strat_rec[st][a].append(r_[si])
                strat_det[st][a].append(d_[si])
    for st in M.STRATES:
        for genre, raw in (("rappel", strat_rec), ("det05", strat_det)):
            stack = {a: np.stack(raw[st][a]) for a in ARMS_TABLE}
            finite = np.isfinite(stack["controle"]).all(0)
            if finite.sum() < 10:
                continue
            nom = f"instances_{st}_{genre}"
            log(f"  [boot] {nom} (masque 12 bras)")
            tables[nom] = M.boot_values_masque(stack, B, rng_seed, PAIRS, f"{genre} {st}")
    # fragments (nouveaux pour G/B ; controle existe déjà — P3.16 phase C2)
    frag = {}
    for a in ("G", "B", "controle"):
        rows = [np.load(MET_DIR / f"frag_{a}_seed{s}.npy").astype(np.float64) for s in SEEDS]
        frag[a] = np.stack(rows)
    log("  [boot] fragments")
    tables["fragments"] = paired_bootstrap_values(frag, B=B, rng_seed=rng_seed, pairs=PAIRS)
    tables["fragments"]["point"] = {a: float(frag[a].mean()) for a in frag}
    return tables


# --------------------------------------------------------------------------- #
# Sanités
# --------------------------------------------------------------------------- #

def sanites(tables: dict):
    """Deux familles de checks :
    · BIT-À-BIT vs P3.16 (même npz, mêmes masques, mêmes index bootstrap) — reproductibilité
      exacte de la régénération ;
    · vs harness P3.10 / master P3.14 avec tolérance 1e-4 (0,01 pt) : ces tables proviennent
      d'un AUTRE forward GPU des mêmes checkpoints (non-déterminisme cuDNN/BF16, config
      `deterministic: false`) — écart déjà documenté par la sanity P3.17 (mIoU consolidé =
      P3.14 à 4,1e-3 pt). Le primaire PRÉ-ENREGISTRÉ cité dans T2 reste la table harness ;
      l'écart entre sources est mesuré et reporté, jamais masqué.
    """
    p316 = json.loads(P316_MET.read_text())["tables"]
    harness = json.loads(HARNESS.read_text())
    p314 = json.loads(P314.read_text())
    n_cmp = 0
    for met, t in tables.items():
        if met == "fragments":
            continue
        ref = p316.get(met, {}).get("pairwise", {}).get("G_vs_controle")
        if ref is None:
            continue
        mine = t["pairwise"]["G_vs_controle"]
        for champ, a, b in (("delta", mine["delta"], ref["delta"]),
                            ("ci_lo", mine["ci95"][0], ref["ci95"][0]),
                            ("ci_hi", mine["ci95"][1], ref["ci95"][1]),
                            ("p", mine["p_two_sided"], ref["p_two_sided"])):
            if abs(a - b) > 1e-12 * max(1.0, abs(b)):
                check(f"P3.16 {met} G_vs_controle {champ}", False, f"{a} != {b}")
        n_cmp += 1
    check("P3.16 G_vs_controle reproduit", n_cmp >= 18,
          f"{n_cmp} métriques bit-à-bit (delta/IC/p, mêmes index bootstrap)")
    bm = harness["bootstrap_miou"]
    m = tables["mIoU"]
    ecg = abs(m["point"]["G"] - bm["point"]["G"])
    ecb = abs(m["point"]["B"] - bm["point"]["B"])
    check("harness P3.10 points G/B (tolérance 1e-4, 2 forwards)",
          ecg < 1e-4 and ecb < 1e-4,
          f"npz G={m['point']['G']:.7f} vs harness {bm['point']['G']:.7f} (écart {ecg:.1e}) ; "
          f"B écart {ecb:.1e} — non-déterminisme forward, cf P3.17 sanity 4,1e-3 pt")
    h = bm["pairwise"]["B_vs_G"]
    g = m["pairwise"]["G_vs_B"]
    check("harness P3.10 primaire G−B (tolérance)",
          abs(g["delta"] + h["delta"]) < 1e-4 and abs(g["p_two_sided"] - h["p_two_sided"]) < 0.02,
          f"npz Δ={g['delta']*100:+.4f}pt p={g['p_two_sided']:.4f} vs harness "
          f"Δ={-h['delta']*100:+.4f}pt p={h['p_two_sided']:.4f} — même verdict (ns)")
    r = p314["pairwise"]["G_vs_controle"]
    g14 = m["pairwise"]["G_vs_controle"]
    check("P3.14 master G_vs_controle (tolérance)",
          abs(g14["delta"] - r["delta"]) < 1e-4 and abs(g14["p_two_sided"] - r["p_two_sided"]) < 0.02,
          f"npz Δ={g14['delta']*100:+.4f}pt p={g14['p_two_sided']:.4f} vs P3.14 "
          f"Δ={r['delta']*100:+.4f}pt p={r['p_two_sided']:.4f}")
    from src.moe.bootstrap import miou_from_cm
    for s in SEEDS:
        hs = harness["arms"]["G"].get("miou_dataset_par_seed", {}).get(str(s))
        cm_s = np.load(MET_DIR / f"metriques_G_seed{s}.npz")["cm"].astype(np.float64)
        mi = miou_from_cm(cm_s.sum(0))
        if hs is not None:
            check(f"mIoU seed {s} G vs harness (tolérance)", abs(mi - hs) < 1e-4,
                  f"npz {mi:.7f} vs harness {hs:.7f} (écart {abs(mi-hs):.1e})")
    fp316 = p316.get("fragments", {}).get("point", {})
    if fp316:
        pt = tables["fragments"]["point"]
        check("fragments controle = P3.16 (mêmes npy)",
              abs(pt["controle"] - fp316["controle"]) < 1e-9,
              f"controle={pt['controle']:.2f} vs P3.16 {fp316['controle']:.2f}")


# --------------------------------------------------------------------------- #
# Tables T1-T6
# --------------------------------------------------------------------------- #

def fmt_pt(x: float, nd=3) -> str:
    return f"{x*100:+.{nd}f}" if abs(x) < 10 else f"{x:+.{nd}f}"


def t1_protocole_cout() -> tuple[str, dict]:
    import yaml
    loss_cfg = yaml.safe_load((REPO / "configs/loss/ce_dice_blob.yaml").read_text())
    exp_cfg = yaml.safe_load((REPO / "configs/experiment/pilot_fullres_G_blob.yaml").read_text())
    tr = exp_cfg["training"]
    comps = {c["type"]: c for c in loss_cfg["components"]}
    logseeds = {}
    for s in SEEDS:
        txt = G_LOGS[s].read_text()
        eps = [(int(m[0]), float(m[1]), float(m[2]), int(m[3])) for m in re.findall(
            r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) \| lr=[\d.eE+-]+ \| VRAM=([\d.]+)GB \| time=(\d+)s",
            txt)]
        times = [e[3] for e in eps]
        logseeds[s] = {
            "n_epochs": len(eps), "train_loss_final": eps[-1][1],
            "vram_max_gb": max(e[2] for e in eps),
            "s_epoch_med": float(np.median(times)), "s_epoch_min": min(times),
            "s_epoch_max": max(times), "total_h": sum(times) / 3600.0,
        }
    blob = comps["blob"]
    lignes = [
        ("Architecture", "ConvNeXt-V2-Base (ImageNet-22K) + UPerNet, pleine résolution 1024×2048, 19 classes, BF16 channels_last", "configs/experiment/pilot_fullres_G_blob.yaml"),
        ("Loss G", f"CE {comps['cross_entropy']['weight']} (class_weights ISNS, ignore 255) + Dice {comps['dice']['weight']} (smooth {comps['dice']['smooth']}) + **blob Kofler {blob['weight']}** (eps {blob['eps']}, min_blob_pixels {blob['min_blob_pixels']}, ignore {blob['ignore_index']})", "configs/loss/ce_dice_blob.yaml"),
        ("Référence B", "MÊME recette, loss ce_dice (CE 0,5 + Dice 0,5) — seule la loss change", "configs/loss/ce_dice.yaml"),
        ("Entraînement", f"{tr['epochs']} époques, batch {tr['batch_size']}×accum {tr['gradient_accumulation_steps']} (effectif 8), AdamW lr {tr['optimizer']['lr']:.0e} wd {tr['optimizer']['weight_decay']}, poly power {tr['scheduler']['power']}, warmup {tr['scheduler']['warmup_epochs']} ep, {tr['mixed_precision']}", "idem"),
        ("Augmentation", "hflip p=0,5 appliqué par le DATASET sur image+label+paquetage d'instances (flip albumentations désactivé — alignement du glob blob) ; même distribution que B", "config G + src/losses/blob_lab.py"),
        ("Seeds", "42, 123, 456 (3 seeds, résultats moyennés dans chaque réplicat bootstrap)", "résultat/p3_queue"),
    ]
    for s in SEEDS:
        d = logseeds[s]
        lignes.append((f"Coût seed {s}",
                       f"{d['n_epochs']}/160 époques, {d['s_epoch_med']:.0f} s/ep médian "
                       f"[{d['s_epoch_min']}-{d['s_epoch_max']}], VRAM {d['vram_max_gb']:.1f} Go, "
                       f"train_loss final {d['train_loss_final']:.4f}, total {d['total_h']:.1f} h",
                       f"results/p3_queue/pilot_fullres_G_blob_seed{s}.log"))
    tot = sum(d["total_h"] for d in logseeds.values())
    lignes.append(("Coût du bras G", f"≈ {tot:.0f} h GPU (3 seeds × 160 époques)", "somme des logs"))
    lignes += [
        ("Pré-calcul blob", "paquetages .blob.npz (blob_glob/blob_counts/blob_csr) : +1 à 2 %/époque contre +870 s/époque pour le chemin naïf (CC CPU + 26 syncs/classe) ; cc3d 8-connexité 22,4 ms/image pleine résolution (16 classes, 118 instances) vs 59,7 ms scipy", "src/losses/blob_loss.py docstring, mesures 2026-09-17"),
        ("Parité", "naïf ↔ pré-calculé verrouillée par tests/test_blob_loss.py ; 2975 paquetages régénérés bit-à-bit (vérifié 2026-09-26, commit 132f2e8) ; paquetages absents du disque au 2026-09-30, régénérables par scripts/precompute_blob_lab.py", "DEVLOG + commit 132f2e8"),
        ("Évaluation", "holdout partagé first:500 de Cityscapes val, mIoU dataset-level cityscapesScripts (convention bit-identique, tests/test_official_miou.py), bootstrap apparié B=10 000 seed 20260618", "src/moe/bootstrap.py"),
    ]
    md = ["# T1 — Protocole et coût (bras G, blob loss seul)", "",
          "| Poste | Valeur | Source |", "|---|---|---|"]
    md += [f"| {a} | {b} | `{c}` |" for a, b, c in lignes]
    data = {"lignes": [{"poste": a, "valeur": b, "source": c} for a, b, c in lignes],
            "logs_par_seed": {str(s): logseeds[s] for s in SEEDS}}
    return "\n".join(md) + "\n", data


def familles_holm_miou() -> dict:
    """Tailles et valeurs de Holm des DEUX familles de multiplicité du mIoU.

    ERRATUM v1.1.0 (2026-10-01) : la colonne de contexte 13 bras était intitulée
    « Holm (15 paires) » — en-tête CODÉ EN DUR — alors que ses valeurs venaient de la
    famille **12 paires** de `p314/master_table.json` (mesuré : D 0,0576 = 12×0,0048,
    MoE 0,0726 = 11×0,0066 ; la famille 15 paires de P3.16 donne 0,075 et 0,0924).
    Piège 2 du JALON_P3.18.md : une étiquette écrite à la main finit par mentir.
    Ici les tailles sont CALCULÉES depuis les artefacts, le Holm est RECOMPUTÉ depuis
    les p bruts (sanity bloquante) et les deux familles sont montrées côte à côte.
    """
    p314 = json.loads(P314.read_text())
    p316 = json.loads(P316_MET.read_text())
    poly = json.loads(P317.read_text())
    fam314 = {k: v["p_two_sided"] for k, v in p314["pairwise"].items()}
    pw316 = p316["tables"]["mIoU"]["pairwise"]
    fam316 = {k: v["p_two_sided"] for k, v in pw316.items()}
    h314, h316 = holm(fam314), holm(fam316)
    for k in fam314:
        check(f"Holm P3.14 recomputé == stocké ({k})",
              abs(h314[k] - p314["pairwise"][k]["p_holm"]) < 1e-12,
              f"{h314[k]:.6g} vs {p314['pairwise'][k]['p_holm']:.6g}")
    for k in fam316:
        check(f"Holm P3.16 recomputé == stocké ({k})",
              abs(h316[k] - pw316[k]["p_holm"]) < 1e-12,
              f"{h316[k]:.6g} vs {pw316[k]['p_holm']:.6g}")
    check("famille P3.14 = 12 paires (12 bras vs contrôle)", len(fam314) == 12,
          f"len(pairwise) = {len(fam314)}")
    check("famille P3.16 = 15 paires (12 vs contrôle + 3 fusion-vs-expert)",
          len(fam316) == 15 and len(p316["pairs"]) == 15,
          f"len(pairwise mIoU) = {len(fam316)}, len(pairs) = {len(p316['pairs'])}")
    # La polyvalence (P3.17) doit citer la famille 15 paires, pas une autre.
    for a in p314["arms"]:
        if a == "controle" or f"{a}_vs_controle" not in pw316:
            continue
        check(f"Holm mIoU polyvalence == famille 15 paires ({a})",
              abs(poly["p_holm"][a]["mIoU"] - h316[f"{a}_vs_controle"]) < 1e-9,
              f"{poly['p_holm'][a]['mIoU']:.6g} vs {h316[f'{a}_vs_controle']:.6g}")
    return {"n314": len(fam314), "n316": len(fam316), "h314": h314, "h316": h316,
            "couverts_316": {k.replace("_vs_controle", "") for k in pw316},
            "partiels": poly.get("annexe_couverture_partielle", []),
            "meilleur314": min(h314.values()), "meilleur316": min(h316.values())}


def t2_primaire(tables: dict) -> tuple[str, dict]:
    harness = json.loads(HARNESS.read_text())
    p314 = json.loads(P314.read_text())
    fam = familles_holm_miou()
    m = tables["mIoU"]
    g = m["pairwise"]["G_vs_B"]
    pts = harness["bootstrap_miou"]["point"]
    cis = harness["bootstrap_miou"]["ci95"]
    per_seed = {a: harness["arms"][a].get("miou_dataset_par_seed", {}) for a in ("G", "B")}
    md = ["# T2 — Endpoint primaire pré-enregistré : G vs B apparié (160 époques, seule la loss change)", "",
          "Protocole : mIoU dataset-level (cityscapesScripts, 19 classes), holdout first:500, seeds 42/123/456 moyennés dans chaque réplicat, bootstrap apparié B=10 000 (seed 20260618), p bilatéral, paire unique → Holm = p.", "",
          "| Quantité | G (CE+Dice+0,5·blob) | B (CE+Dice) |", "|---|---|---|",
          f"| mIoU point | **{pts['G']*100:.3f}** | **{pts['B']*100:.3f}** |",
          f"| IC95 | [{cis['G'][0]*100:.3f} ; {cis['G'][1]*100:.3f}] | [{cis['B'][0]*100:.3f} ; {cis['B'][1]*100:.3f}] |"]
    for s in SEEDS:
        md.append(f"| mIoU seed {s} | {per_seed['G'].get(str(s), float('nan'))*100:.3f} | "
                  f"{per_seed['B'].get(str(s), float('nan'))*100:.3f} |")
    md += ["",
           f"**Δ(G−B) = {g['delta']*100:+.3f} pt · IC95 [{g['ci95'][0]*100:+.3f} ; {g['ci95'][1]*100:+.3f}] · "
           f"p = {g['p_two_sided']:.4f} (Holm paire unique = {g['p_holm']:.4f}) → NON SIGNIFICATIF.**", "",
           "Lecture honnête : l'endpoint primaire est **nul**. Le blob loss n'améliore pas le mIoU "
           "en segmentation sémantique exclusive pleine résolution. La contribution du papier est le "
           "diagnostic du trade-off (T3, T4) et la position de polyvalence (T5).", "",
           "## Contexte 13 bras (référence contrôle 80 époques — NON apparié en budget, donné pour le plateau)", ""]
    pw = p314["pairwise"]
    rows = sorted(((a, pw[f"{a}_vs_controle"]["delta"]) for a in p314["arms"] if a != "controle"),
                  key=lambda x: -x[1])
    n314, n316 = fam["n314"], fam["n316"]
    md += [f"| # | Bras | mIoU | Δ vs contrôle | p | Holm (famille {n314} paires, P3.14) | "
           f"Holm (famille {n316} paires, P3.16) |", "|---|---|---|---|---|---|---|"]
    for i, (a, _d) in enumerate(rows, 1):
        e = pw[f"{a}_vs_controle"]
        star = " ← **G (ce papier)**" if a == "G" else ""
        h16 = (f"{fam['h316'][f'{a}_vs_controle']:.3f}" if a in fam["couverts_316"] else "n.c. (a)")
        md.append(f"| {i} | {LABEL_BRAS.get(a, a)}{star} | {p314['point'][a]*100:.2f} | "
                  f"{e['delta']*100:+.2f} [{e['ci95'][0]*100:+.2f}, {e['ci95'][1]*100:+.2f}] | "
                  f"{e['p_two_sided']:.4f} | {e['p_holm']:.3f} | {h16} |")
    md.append(f"| — | contrôle (réf) | {p314['point']['controle']*100:.2f} | — | — | — | — |")
    rang_G = [a for a, _ in rows].index("G") + 1
    # Rang contrôle inclus : le contrôle a Δ = 0 par définition, il se classe donc SOUS
    # tout bras positif — le rang 13 bras se CALCULE en l'insérant, il ne se déduit pas
    # de rang_G (v1.1.0 : la déduction rang_G+1 donnait 8ᵉ au lieu de 7ᵉ).
    rows_13 = sorted(list(rows) + [("controle", 0.0)], key=lambda x: -x[1])
    rang_G_13 = [a for a, _ in rows_13].index("G") + 1
    rang_ctl_13 = [a for a, _ in rows_13].index("controle") + 1
    check("classement 13 bras trié par Δ décroissant",
          all(rows_13[i][1] >= rows_13[i + 1][1] for i in range(len(rows_13) - 1)),
          f"{len(rows_13)} bras, Δ de {rows_13[0][1]*100:+.2f} à {rows_13[-1][1]*100:+.2f} pt")
    check("rang G contrôle inclus == rang G décalé de la position du contrôle",
          rang_G_13 == rang_G + (1 if rang_ctl_13 <= rang_G else 0),
          f"rang_G={rang_G}/{len(rows)}, rang_G_13={rang_G_13}/{len(rows_13)}, "
          f"rang_contrôle={rang_ctl_13}")
    md += ["", f"(a) A est hors de la famille {n316} paires (couverture partielle, "
           f"`annexe_couverture_partielle` de P3.17 = {fam['partiels']}) : son Holm n'y est pas calculable.",
           "", f"**Aucun bras ne passe Holm 0,05 dans AUCUNE des deux familles** (meilleur "
           f"{fam['meilleur314']:.4f} sur {n314} paires, {fam['meilleur316']:.4f} sur {n316} paires) ; "
           f"G est {rang_G}ᵉ/{len(rows)} par amplitude "
           f"({rang_G_13}ᵉ/{len(rows_13)} contrôle inclus — le contrôle, Δ = 0, se classe "
           f"{rang_ctl_13}ᵉ ; table P3.14).",
           "", f"Les deux colonnes sont des familles de multiplicité DIFFÉRENTES et les deux sont "
           f"données : {n314} paires = les 12 bras contre le contrôle (`p314/master_table.json`), "
           f"{n316} paires = la famille exploratoire du programme, qui ajoute les 4 comparaisons "
           f"fusion-contre-expert (`metiers_experts/table_metiers_experts.json`, reprise par P3.17). "
           f"Tailles calculées depuis les artefacts, Holm recomputé depuis les p bruts et comparé aux "
           f"valeurs stockées (sanity bloquante) — jamais recopié d'une étiquette.",
           "", "Provenance : `results/moe_v3_cs/harness/table_Gseul_P310.json` (primaire cité ci-dessus) et "
           "`results/moe_v3_cs/p314/master_table.json` (contexte). La ligne mIoU de T4 est régénérée depuis "
           "les npz P3.16 (autre forward GPU des mêmes checkpoints) : écart ≤ 0,001 pt avec le harness "
           "(non-déterminisme cuDNN/BF16, `deterministic: false` ; même ordre que la sanity P3.17, 4,1e-3 pt) — "
           "mesuré dans SANITY.md, le verdict ns est identique dans les deux sources."]
    data = {"primaire": {"point_G": pts["G"], "point_B": pts["B"], "ci_G": cis["G"], "ci_B": cis["B"],
                         "delta": g["delta"], "ci95": g["ci95"], "p": g["p_two_sided"],
                         "holm": g["p_holm"], "per_seed": per_seed},
            "contexte_13bras": {"classement": [{"bras": a, "delta": d} for a, d in rows],
                                "points": p314["point"],
                                "n_famille_p314": n314, "n_famille_p316": n316,
                                "holm_p314": {k.replace("_vs_controle", ""): v for k, v in fam["h314"].items()},
                                "holm_p316": {k.replace("_vs_controle", ""): v for k, v in fam["h316"].items()},
                                "meilleur_holm_p314": fam["meilleur314"],
                                "meilleur_holm_p316": fam["meilleur316"],
                                "bras_hors_famille_316": sorted(set(p314["arms"]) - {"controle"} - fam["couverts_316"]),
                                "rang_G": rang_G, "rang_G_13": rang_G_13,
                                "rang_controle_13": rang_ctl_13}}
    return "\n".join(md) + "\n", data


def t3_perclass() -> tuple[str, str, dict]:
    attB = json.loads(P316_ATT_B.read_text())
    attC = json.loads(P316_ATT_C.read_text())
    gB = {r["classe"]: r for r in attB["arms"]["G"]["per_class"]}
    gC = {r["classe"]: r for r in attC["arms"]["G"]["per_class"]}
    md = ["# T3 — IoU par classe, bras G (160 époques)", "",
          "À gauche : **G vs B apparié** (primaire — même recette, 160 époques, seule la loss change). "
          "À droite : G vs contrôle 80 époques (contexte 13 bras, NON apparié en budget). "
          "IC95 bootstrap apparié B=10 000. Deux critères, jamais confondus : **gras** = Holm par bras "
          "(famille des 19 classes) < 0,05 ; **†** = IC95 excluant 0 (convention « significant » de P3.16). "
          "Les deux sont reportés ; aucun des deux n'est présenté comme l'autre.", "",
          "| Classe | IoU G | IoU B | Δ(G−B) pt | IC95 | p | Holm | | Δ(G−ctrl) pt | IC95 | p | Holm |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = [["classe", "iou_G", "iou_B", "delta_GvsB_pt", "ci_lo_pt", "ci_hi_pt", "p", "holm",
                 "ci_exclut_0_B", "delta_GvsCtrl_pt", "ci_lo_pt_ctrl", "ci_hi_pt_ctrl", "p_ctrl",
                 "holm_ctrl", "ci_exclut_0_ctrl"]]
    for cl in CLASSES_19:
        b, c = gB[cl], gC[cl]
        sig_b = "**" if b["p_holm_arm"] < 0.05 else ""
        sig_c = "**" if c["p_holm_arm"] < 0.05 else ""
        dag_b = "†" if (b["ci_lo_pt"] > 0 or b["ci_hi_pt"] < 0) else ""
        dag_c = "†" if (c["ci_lo_pt"] > 0 or c["ci_hi_pt"] < 0) else ""
        md.append(f"| {cl} | {b['iou_bras']*100:.2f} | {b['iou_ref']*100:.2f} | "
                  f"{sig_b}{b['delta_pt']:+.2f}{sig_b}{dag_b} | [{b['ci_lo_pt']:+.2f} ; {b['ci_hi_pt']:+.2f}] | "
                  f"{b['p_two_sided']:.4f} | {b['p_holm_arm']:.3f} | | "
                  f"{sig_c}{c['delta_pt']:+.2f}{sig_c}{dag_c} | [{c['ci_lo_pt']:+.2f} ; {c['ci_hi_pt']:+.2f}] | "
                  f"{c['p_two_sided']:.4f} | {c['p_holm_arm']:.3f} |")
        csv_rows.append([cl, f"{b['iou_bras']*100:.4f}", f"{b['iou_ref']*100:.4f}",
                         f"{b['delta_pt']:+.4f}", f"{b['ci_lo_pt']:+.4f}", f"{b['ci_hi_pt']:+.4f}",
                         f"{b['p_two_sided']:.6f}", f"{b['p_holm_arm']:.6f}",
                         "1" if dag_b else "0",
                         f"{c['delta_pt']:+.4f}", f"{c['ci_lo_pt']:+.4f}", f"{c['ci_hi_pt']:+.4f}",
                         f"{c['p_two_sided']:.6f}", f"{c['p_holm_arm']:.6f}",
                         "1" if dag_c else "0"])
    gainsB = [r for r in attB["arms"]["G"]["per_class"]
              if (r["ci_lo_pt"] > 0 or r["ci_hi_pt"] < 0) and r["delta_pt"] > 0]
    pertesB = [r for r in attB["arms"]["G"]["per_class"]
               if (r["ci_lo_pt"] > 0 or r["ci_hi_pt"] < 0) and r["delta_pt"] < 0]
    holmB = [r["classe"] for r in gainsB if r["p_holm_arm"] < 0.05]
    txt_gains = ", ".join("%s %+.2f" % (r["classe"], r["delta_pt"]) for r in gainsB) or "—"
    txt_pertes = ", ".join("%s %+.2f" % (r["classe"], r["delta_pt"]) for r in pertesB) or "—"
    txt_holm = ", ".join(holmB) or "—"
    md += ["", f"Vs **B** (apparié) — IC excluant 0 : gains {txt_gains} ; dégradations {txt_pertes}. "
           f"Après Holm par bras (19 classes), seuls survivent : {txt_holm} "
           f"(truck, person, car, train redeviennent ns : le Holm 19 classes est sévère, c'est écrit tel quel).",
           "Provenance : `results/moe_v3_cs/p316/attribution_perclass_vs_{B,controle}.json` (bootstrap P3.16, régénéré ici à l'identique pour G)."]
    csv_txt = "\n".join(",".join(str(x) for x in r) for r in csv_rows) + "\n"
    data = {"vs_B": {r["classe"]: r for r in attB["arms"]["G"]["per_class"]},
            "vs_controle": {r["classe"]: r for r in attC["arms"]["G"]["per_class"]}}
    return "\n".join(md) + "\n", csv_txt, data


def t4_metiers(tables: dict, p316_holm: dict, frag_cls: dict) -> tuple[str, str, dict]:
    md = ["# T4 — Métriques officielles et métier, paires du papier (RÉGÉNÉRÉ depuis les npz par image)", "",
          "Protocole identique P3.15/P3.16 (mêmes npz, mêmes masques 12 bras, mêmes index bootstrap B=10 000 seed 20260618). "
          "`Holm(2)` = Holm sur la famille de CE papier (les 2 paires G_vs_B et G_vs_controle) ; "
          "`Holm P3.16` = cross-référence du Holm de la famille exploratoire 15 paires (table P3.16, G_vs_controle). "
          "**gras** = Holm(2) < 0,05. Unité : points (×100), sauf `fragments` = composantes connexes par image (count).", "",
          "| Métrique | n | G | B | ctrl | Δ(G−B) [IC95] | p | Holm(2) | Δ(G−ctrl) [IC95] | p | Holm(2) | Holm P3.16 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = [["metrique", "n_images", "point_G", "point_B", "point_controle",
                 "delta_GvsB_pt", "ci_lo", "ci_hi", "p_brut", "holm_2paires",
                 "delta_GvsCtrl_pt", "ci_lo_ctrl", "ci_hi_ctrl", "p_brut_ctrl",
                 "holm_2paires_ctrl", "holm_p316_ctrl"]]
    for met in METIERS_ORDER:
        t = tables.get(met)
        if t is None:
            continue
        gb = t["pairwise"]["G_vs_B"]
        gc = t["pairwise"]["G_vs_controle"]
        h316 = p316_holm.get(met, {}).get("G_vs_controle", {}).get("p_holm")
        h316s = f"{h316:.3f}" if h316 is not None else "—"
        b1 = "**" if gb["p_holm"] < 0.05 else ""
        b2 = "**" if gc["p_holm"] < 0.05 else ""
        pt = t["point"]
        u = 100.0 if met != "fragments" else 1.0     # fragments = counts, pas des points
        nd = 3 if met != "fragments" else 1
        md.append(f"| {met} | {t['n_images']} | {pt['G']*u:.2f} | {pt['B']*u:.2f} | "
                  f"{pt['controle']*u:.2f} | {b1}{gb['delta']*u:+.{nd}f}{b1} "
                  f"[{gb['ci95'][0]*u:+.{nd}f} ; {gb['ci95'][1]*u:+.{nd}f}] | {gb['p_two_sided']:.4g} | "
                  f"{gb['p_holm']:.4g} | {b2}{gc['delta']*u:+.{nd}f}{b2} "
                  f"[{gc['ci95'][0]*u:+.{nd}f} ; {gc['ci95'][1]*u:+.{nd}f}] | {gc['p_two_sided']:.4g} | "
                  f"{gc['p_holm']:.4g} | {h316s} |")
        csv_rows.append([met, t["n_images"], f"{pt['G']*u:.4f}", f"{pt['B']*u:.4f}",
                         f"{pt['controle']*u:.4f}",
                         f"{gb['delta']*u:+.4f}", f"{gb['ci95'][0]*u:+.4f}",
                         f"{gb['ci95'][1]*u:+.4f}", f"{gb['p_two_sided']:.6g}",
                         f"{gb['p_holm']:.6g}",
                         f"{gc['delta']*u:+.4f}", f"{gc['ci95'][0]*u:+.4f}",
                         f"{gc['ci95'][1]*u:+.4f}", f"{gc['p_two_sided']:.6g}",
                         f"{gc['p_holm']:.6g}", h316s])
    # ---- diagnostic NEUF : fragmentation par classe (composantes connexes, G vs B) ----
    rows = sorted(frag_cls.items(), key=lambda kv: -kv[1]["pairwise"]["G_vs_B"]["delta"])
    raw_p = {nom: t["pairwise"]["G_vs_B"]["p_two_sided"] for nom, t in rows}
    h19 = holm(raw_p)
    frag_rows = []
    md += ["", "## Fragmentation par classe — composantes connexes 8-connexes par image (diagnostic neuf, G vs B)", "",
           "Mesure inédite du programme (P3.16 n'avait les fragments que pour contrôle/MoE). "
           "Holm sur la famille des 19 classes. **gras** = Holm < 0,05.", "",
           "| Classe | G | B | ctrl | Δ(G−B) | IC95 | p | Holm(19) |", "|---|---|---|---|---|---|---|---|"]
    for nom, t in rows:
        pw = t["pairwise"]["G_vs_B"]
        b = "**" if h19[nom] < 0.05 else ""
        md.append(f"| {nom} | {t['point']['G']:.1f} | {t['point']['B']:.1f} | "
                  f"{t['point']['controle']:.1f} | {b}{pw['delta']:+.1f}{b} "
                  f"[{pw['ci95'][0]:+.1f} ; {pw['ci95'][1]:+.1f}] | {pw['p_two_sided']:.4g} | "
                  f"{h19[nom]:.4g} |")
        frag_rows.append({"classe": nom, "G": t["point"]["G"], "B": t["point"]["B"],
                          "controle": t["point"]["controle"], "delta": pw["delta"],
                          "ci95": pw["ci95"], "p": pw["p_two_sided"], "holm19": h19[nom]})
        csv_rows.append([f"fragments_{nom}", t["n_images"], f"{t['point']['G']:.4f}",
                         f"{t['point']['B']:.4f}", f"{t['point']['controle']:.4f}",
                         f"{pw['delta']:+.4f}", f"{pw['ci95'][0]:+.4f}", f"{pw['ci95'][1]:+.4f}",
                         f"{pw['p_two_sided']:.6g}", f"{h19[nom]:.6g}", "", "", "", "", "", ""])
    # ---- comptes de classes CALCULÉS depuis l'artefact (2026-10-01) ----
    # La prose ne doit plus jamais porter un compte codé en dur : c'est exactement comme ça que
    # « 18 classes sur 19 » avait survécu dans T4, le manuscrit et la figure F2 alors que
    # l'artefact en donne 16 vs B apparié (voir papers/paper4/SPEC.md, Journal du 2026-10-01).
    n_cls = len(frag_rows)
    n_frag_hausse = sum(1 for r in frag_rows if r["delta"] > 0)
    n_frag_holm = sum(1 for r in frag_rows if r["delta"] > 0 and r["holm19"] < 0.05)
    n_frag_ctrl = sum(1 for r in frag_rows if r["G"] > r["controle"])
    frag_baisses = [r for r in frag_rows if r["delta"] <= 0]
    frag_baisse_materielle = [r["classe"] for r in frag_baisses if r["holm19"] < 0.05]
    frag_baisses_txt = ", ".join(f"{r['classe']} {r['delta']:+.1f}" for r in frag_baisses)
    check(f"frag {n_frag_hausse}/{n_cls} classes en hausse vs B", n_cls == 19 and n_frag_hausse >= 1,
          f"{n_frag_hausse}/{n_cls} hausses vs B, {n_frag_ctrl}/{n_cls} vs contrôle, "
          f"{n_frag_holm} Holm-significatives ; ne montent pas vs B : {frag_baisses_txt}")
    check("frag seule baisse matérielle = terrain", frag_baisse_materielle == ["terrain"],
          f"baisses Holm-significatives : {frag_baisse_materielle}")

    md += ["", "Lecture (thèse du papier) : le terme blob **achète** l'IoU pixel des objets fins "
           "(traffic light, pole, bicycle, person, truck — T3) et un BF1 3px supérieur, et le **paie** "
           "en rappel instance piéton (strict, T1, foule) et en précision pixel piéton. "
           "Les deux paires racontent la même histoire ; la paire appariée en budget (G−B) est celle du primaire.",
           "",
           "Lecture du diagnostic de fragmentation (mesure neuve, 500 images × 3 seeds) : le terme blob "
           "ne **compacte pas** les prédictions — il les **fragmente**. Les composantes connexes augmentent "
           f"dans {n_frag_hausse} classes sur {n_cls} vs B apparié ({n_frag_holm} hausses Holm-significatives) "
           f"et dans {n_frag_ctrl} sur {n_cls} vs le bras contrôle. Classes qui ne montent pas vs B : "
           f"{frag_baisses_txt} — seule {', '.join(frag_baisse_materielle)} est une baisse matérielle "
           "(Holm < 0,001), les autres sont nulles (p ≥ 0,70). Grandes classes mouchetées (car +49,4, pole +45,4, "
           "vegetation +43,8, road +38,6 composantes/image) et masques piétons morcelés (person 24,3 → 53,2, "
           "×2,2), cohérent avec les pixels perdus à l'intérieur des instances GT (rappel strict −4,3 ; "
           "petits T1 −5,6 — un masque troué se coupe en composantes disjointes). Les classes fines gagnent "
           "aussi des composantes (traffic light +0,9, bicycle +2,2, Holm-significatifs) tout en gagnant de "
           "l'IoU pixel : le terme achète de la couverture pixel des objets fins, pas leur intégrité "
           "topologique. Total : 933,9 composantes/image (G) contre 615,1 (B), ×1,52.",
           "", "Note mIoU : cette ligne vient du forward des npz P3.16 ; le primaire pré-enregistré (T2) vient "
           "du forward harness P3.10 — écart ≤ 0,001 pt (non-déterminisme cuDNN/BF16), verdict ns identique, "
           "écart mesuré dans SANITY.md. Toutes les autres lignes sont exactement la source P3.16/P3.17.",
           "", "Provenance : npz `results/moe_v3_cs/metiers_experts/metriques_*_seed*.npz` + "
           "fragments `frag_{G,B,controle}_seed*.npy` et décomposition par classe "
           "`results/moe_v3_cs/paper4_blob/frag_classes_*_seed*.npy` (G/B calculés ce jour depuis les preds "
           "/mnt/data8t/cityscape_p316, contrôle depuis results/moe_v3_cs/metiers) ; bootstraps bruts "
           "`results/moe_v3_cs/paper4_blob/table_metiers_G_pairees.json` + `table_frag_classes_G_vs_B.json`."]
    csv_txt = "\n".join(",".join(str(x) for x in r) for r in csv_rows) + "\n"
    return "\n".join(md) + "\n", csv_txt, {"tables": tables, "frag_classes": frag_rows}


def t5_polyvalence() -> tuple[str, str, dict]:
    p = json.loads(P317.read_text())
    prof = p["profil"]
    ordre = p["classement_polyvalence"]
    md = ["# T5 — Polyvalence (P3.17) : position de G sur 36 endpoints × 13 bras", "",
          "36 endpoints (19 IoU par classe + 17 métriques métier) × 13 bras, holdout first:500, "
          "bootstrap apparié B=10 000 + Holm. `dommage max` = pire Δ d'un endpoint vs contrôle "
          "(plus c'est proche de 0, plus le bras est sans point faible). Critère T0 pré-enregistré : "
          "ΔmIoU significatif ET aucun endpoint >1 pt sous la référence ET aucune classe dégradée >0,5 pt.", "",
          "| # | Bras | percentile moyen | pire rang | #1 | dommage max (pt) | z | pic max (pt) | ΔmIoU (p / Holm) | gains sig (brut/Holm) | pertes sig (brut/Holm) | Pareto |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv_rows = [["rang", "bras", "mean_pct", "pire_rang", "n_rang1", "dommage_max_pt",
                 "endpoint_dommage", "z", "pic_max_pt", "delta_miou_pt", "p_miou", "holm_miou",
                 "gains_sig_brut", "gains_sig_holm", "pertes_sig_brut", "pertes_sig_holm", "pareto"]]
    for i, a in enumerate(ordre, 1):
        q = prof[a]
        star = " ← **G (ce papier)**" if a == "G" else ""
        pareto = "non dominé" if q["pareto_non_dominé"] else "dominé"
        md.append(f"| {i} | {LABEL_BRAS.get(a, a)}{star} | {q['mean_pct']*100:.1f} | {q['pire_rang']} | "
                  f"{q['n_rang1']} | **{q['dommage_max_pt']:.2f}** ({q['endpoint_dommage']}) | "
                  f"{q['dommage_max_z']:+.2f} | {q['pic_max_pt']:+.2f} ({q['endpoint_pic']}) | "
                  f"{q['delta_miou_pt']:+.2f} ({q['p_miou']:.4f} / {q['holm_miou']:.3f}) | "
                  f"{q['n_gains_sig_brut']}/{q['n_gains_sig_holm']} | "
                  f"{q['n_pertes_sig_brut']}/{q['n_pertes_sig_holm']} | {pareto} |")
        csv_rows.append([i, a, f"{q['mean_pct']*100:.2f}", q["pire_rang"], q["n_rang1"],
                         f"{q['dommage_max_pt']:.4f}", q["endpoint_dommage"],
                         f"{q['dommage_max_z']:.3f}", f"{q['pic_max_pt']:.4f}",
                         f"{q['delta_miou_pt']:.4f}", f"{q['p_miou']:.6f}", f"{q['holm_miou']:.6f}",
                         q["n_gains_sig_brut"], q["n_gains_sig_holm"],
                         q["n_pertes_sig_brut"], q["n_pertes_sig_holm"],
                         "non_dominé" if q["pareto_non_dominé"] else "dominé"])
    g = prof["G"]
    md += ["", "## Profil G (bras de ce papier)", "",
           f"- percentile moyen **{g['mean_pct']*100:.1f}** · rang moyen {g['mean_rang']:.1f} · pire rang **{g['pire_rang']}/13** · dernier sur {g['n_dernier']}/36 endpoints",
           f"- **#1 sur {g['n_rang1']}/36 endpoints** (spécialiste : c'est plus que le MoE, 0) — mais dommage maximal **{g['dommage_max_pt']:.2f} pt** ({g['endpoint_dommage']}, z = {g['dommage_max_z']:+.2f} σ)",
           f"- prix consenti par point de mIoU gagné : **{g['prix_pt_par_pt_miou']:.1f} pt** de dommage (contre −1,2 pour le MoE-V3-CS)",
           f"- gains significatifs (brut) : {', '.join(g['gains_sig_brut'])} ; pertes : {len(g['pertes_sig_brut'])} endpoints dont {len(g['pertes_sig_holm'])} survivent à Holm",
           f"- **dominé** au sens de Pareto par : {', '.join(LABEL_COURT.get(x, x) for x in g['domine_par'])}",
           f"- familles : meilleure = {g['meilleure_famille']} ({g['meilleure_famille_delta_moyen_pt']:+.2f} pt moyen), pire = **{g['pire_famille']}** ({g['pire_famille_delta_moyen_pt']:+.2f} pt)",
           "", f"## Verdict T0 (pré-enregistré) : {p['verdict']['T0_verdict']}", "",
           "Lecture pour le papier : G est l'anti-thèse du critère T0 — un spécialiste dominé dont la "
           "spécialité (objets fins) est précisément ce que le MoE-V3-CS absorbe sans hériter des dégâts "
           "(papier 3). Sa place dans le classement est donnée telle quelle, sans re-classement.", "",
           "Provenance : `results/moe_v3_cs/p317_polyvalence/table_polyvalence.json` (P3.17, 2026-09-29)."]
    csv_txt = "\n".join(",".join(str(x) for x in r) for r in csv_rows) + "\n"
    return "\n".join(md) + "\n", csv_txt, {"profil_G": g, "classement": ordre}


def t6_brats() -> tuple[str, dict]:
    md = ["# T6 — Cross-dataset : le même verdict sur BRATS-2023 (chiffres CITÉS, non re-mesurés ici)", "",
          "Le bras G existe aussi côté BRATS (region-based/sigmoid, MedNeXt, recette nnU-Net, "
          "CV 5 folds, n = 1196) — deuxième dataset, deuxième régime de probabilités. "
          "Les chiffres sont cités depuis les artefacts du programme BRATS (vérifiés par grep "
          "le 2026-09-30), pas re-mesurés dans ce dépôt.", "",
          "| Bras BRATS | Résultat | Source exacte |", "|---|---|---|",
          "| G blob loss **seul** (vs baseline nnU-Net) | **−0,00376 Dice** (CV 5 folds ; 0/5 folds gagnées — dernier des 4 experts) | BRATS repo `papers/paper3/versions/v3/NOTE_DECISION_V3.md` ligne 46 + `tables/v3_vs_consensus.md` ligne 38 |",
          "| G comme **expert 3** du MoE-V3 gagnant | gate V3 : **+0,00566 Dice** vs sa propre baseline, **1ᵉʳ des 29 bras** (5/5 folds) | NOTE_DECISION_V3.md lignes 52, 115 ; Zenodo DOI 10.5281/zenodo.22903668 (abstract, API 2026-09-30) |",
          "", "Même conclusion qu'ici : le blob loss seul n'apporte pas de gain global (nul/négatif), "
          "mais il contribue comme expert initialisé d'un mélange. Les deux programmes sont "
          "indépendants (datasets, architectures, métriques, régimes de probabilités différents) — "
          "la convergence des verdicts est l'argument de généralité du papier 4.", "",
          "Papiers BRATS du programme : MoE « The Gate Does Not Choose » DOI 10.5281/zenodo.22903668 "
          "(concept 10.5281/zenodo.22776410) ; kervadec DOI 10.5281/zenodo.22906447 ; "
          "consensus DOI 10.5281/zenodo.22904810 ; distmap DOI 10.5281/zenodo.20110976."]
    data = {"g_seul_dice": -0.00376, "moe_v3_dice": 0.00566, "n_bras_brats": 29,
            "doi_moe_brats": "10.5281/zenodo.22903668",
            "sources": ["BRATS:papers/paper3/versions/v3/NOTE_DECISION_V3.md:46,52,115",
                        "BRATS:papers/paper3/tables/v3_vs_consensus.md:38",
                        "zenodo:22903668:abstract"]}
    return "\n".join(md) + "\n", data


# --------------------------------------------------------------------------- #
# Figures F1-F3
# --------------------------------------------------------------------------- #

def _forest_row(ax, y, d, lo, hi, sig, couleur, alpha_ns=0.5):
    ax.plot([lo, hi], [y, y], color=couleur, lw=1.8, alpha=0.9 if sig else alpha_ns, zorder=2)
    ax.scatter([lo, hi], [y, y], s=14, color=couleur, alpha=0.9 if sig else alpha_ns, zorder=3)
    ax.scatter([d], [y], s=120 if sig else 48, color=couleur, alpha=1.0 if sig else 0.6,
               edgecolor="black" if sig else "none", linewidth=1.2, zorder=4)


def f1_forest_perclass(data_t3: dict) -> dict:
    rows = sorted(data_t3["vs_B"].values(), key=lambda r: -r["delta_pt"])
    fig, ax = plt.subplots(figsize=(9.2, 0.42 * len(rows) + 2.0))
    ymax = 0.0
    for y, r in enumerate(rows):
        ci_sig = r["ci_lo_pt"] > 0 or r["ci_hi_pt"] < 0       # convention P3.16 « significant »
        holm_sig = r["p_holm_arm"] < 0.05                     # Holm par bras (19 classes)
        c = ("#2e7d32" if r["delta_pt"] > 0 else "#c62828") if ci_sig else "#757575"
        _forest_row(ax, y, r["delta_pt"], r["ci_lo_pt"], r["ci_hi_pt"], ci_sig, c)
        ax.text(max(r["ci_hi_pt"], r["delta_pt"]) + 0.18, y,
                f"{r['delta_pt']:+.2f}" + (" *" if holm_sig else ""),
                va="center", fontsize=8.6, color=c if ci_sig else "#555555")
        ymax = max(ymax, abs(r["ci_lo_pt"]), abs(r["ci_hi_pt"]))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['classe']} ({CL_FR[r['classe']]})" for r in rows], fontsize=9)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlim(-(ymax * 1.15 + 0.3), ymax * 1.15 + 1.1)
    ax.invert_yaxis()
    ax.set_xlabel("Δ IoU (points) — G (CE+Dice+0,5·blob Kofler) − B (CE+Dice)", fontsize=10)
    ax.set_title("F1 — IoU par classe, paire appariée G vs B (160 époques, seule la loss change)\n"
                 "bootstrap apparié B=10 000, 500 images, IC95 ; couleur = IC excluant 0 (gris = IC croisant 0) ; "
                 "* = Holm par bras < 0,05", fontsize=10.5, pad=10)
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    rap = {"n_rows": len(rows),
           "n_ci_sig": sum(1 for r in rows if r["ci_lo_pt"] > 0 or r["ci_hi_pt"] < 0),
           "n_sig": sum(1 for r in rows if r["p_holm_arm"] < 0.05),
           "xlim": list(ax.get_xlim()), "classes_ordre": [r["classe"] for r in rows]}
    return rap


def f2_tradeoff(data_t3: dict, tables: dict, frag_rows: list) -> dict:
    # Compte CALCULÉ, jamais codé en dur : le suptitre est gravé dans le PNG/PDF publié, un
    # « 18 sur 19 » écrit à la main y survivrait à toutes les relectures (corrigé le 2026-10-01).
    n_cls = len(frag_rows)
    n_hausse = sum(1 for r in frag_rows if r["delta"] > 0)
    n_ctrl = sum(1 for r in frag_rows if r["G"] > r["controle"])
    achats = ["truck", "traffic light", "pole", "bicycle", "person", "car"]
    rowsA = [next(r for r in data_t3["vs_B"].values() if r["classe"] == c) for c in achats]
    paie = ["rappel_strict_instances", "instances_toutes_rappel",
            "instances_individuelles_rappel", "instances_foule_rappel",
            "instances_taille_T1_rappel", "instances_taille_T2_rappel",
            "instances_taille_T3_rappel"]
    libB = {"rappel_strict_instances": "rappel strict (toutes instances)",
            "instances_toutes_rappel": "rappel toutes",
            "instances_individuelles_rappel": "rappel individuelles",
            "instances_foule_rappel": "rappel foule (cachés)",
            "instances_taille_T1_rappel": "rappel petits T1",
            "instances_taille_T2_rappel": "rappel moyens T2",
            "instances_taille_T3_rappel": "rappel grands T3"}
    top7 = [r["classe"] for r in frag_rows if r["delta"] > 0][:7]
    forcees = ["person", "traffic light", "terrain"]     # lien rappel / paradoxe objets fins / seule baisse
    gardee = set(top7) | set(forcees)
    rowsC = [r for r in frag_rows if r["classe"] in gardee]
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.9, 0.55 * max(len(rowsA), len(paie)) + 2.4))

    def draw(ax, rows, get, titre, xlabel):
        ymax = 0.0
        for y, r in enumerate(rows):
            d, lo, hi, sig, lib, couleur = get(r)
            _forest_row(ax, y, d, lo, hi, sig, couleur)
            ax.text(max(hi, d) + 0.03 * (ymax + 1), y, f"{d:+.2f}" + (" *" if sig else ""),
                    va="center", fontsize=8.8)
            ymax = max(ymax, abs(lo), abs(hi), abs(d))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([get(r)[4] for r in rows], fontsize=9)
        ax.axvline(0, color="black", lw=1.0)
        ax.set_xlim(-(ymax * 1.15 + 0.15), ymax * 1.35 + 0.4)
        ax.invert_yaxis()
        ax.set_title(titre, fontsize=10, pad=8)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(axis="x", alpha=0.25, lw=0.6)

    draw(axA, rowsA,
         lambda r: (r["delta_pt"], r["ci_lo_pt"], r["ci_hi_pt"], r["p_holm_arm"] < 0.05,
                    f"{r['classe']} ({CL_FR[r['classe']]})", "#2e7d32"),
         "Ce que le blob ACHÈTE — IoU pixel des objets fins (vs B)\n(IC excluant 0 ; * = Holm par bras < 0,05)",
         "Δ IoU (points), IC95 — * = Holm < 0,05")
    draw(axB, paie,
         lambda met: (lambda pw: (pw["delta"] * 100, pw["ci95"][0] * 100, pw["ci95"][1] * 100,
                                  pw["p_holm"] < 0.05, libB[met], "#c62828"))(tables[met]["pairwise"]["G_vs_B"]),
         "Ce que le blob PAIE — rappel instance piéton (vs B)",
         "Δ rappel (points), IC95 — * = Holm < 0,05")
    draw(axC, rowsC,
         lambda r: (r["delta"], r["ci95"][0], r["ci95"][1], r["holm19"] < 0.05,
                    f"{r['classe']} ({CL_FR[r['classe']]})",
                    "#ef6c00" if r["delta"] > 0 else "#1565c0"),
         "La facture TOPOLOGIE — composantes connexes par classe (vs B)",
         "Δ nb composantes/image, IC95 — * = Holm(19) < 0,05")
    fig.suptitle(f"F2 — Le trade-off du blob loss seul : IoU pixel des objets fins ↑ contre rappel instance "
                 f"piéton ↓ — et la facture topologie : plus de composantes connexes dans {n_hausse} classes "
                 f"sur {n_cls} vs B apparié ({n_ctrl} sur {n_cls} vs contrôle, piétons ×2,2)\n"
                 "G vs B apparié (160 époques, seule la loss change) ; gauche : attribution per-class P3.16 · "
                 "milieu : métriques instance régénérées · droite : décomposition par classe des composantes "
                 "connexes (mesure inédite, preds P3.16)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    return {"n_rows_gauche": len(rowsA), "n_rows_milieu": len(paie), "n_rows_droite": len(rowsC),
            "classes_panneau_C": [r["classe"] for r in rowsC],
            "frag_n_classes": n_cls, "frag_n_hausse_vs_B": n_hausse,
            "frag_n_hausse_vs_controle": n_ctrl,
            "suptitre_frag": f"{n_hausse}/{n_cls} vs B, {n_ctrl}/{n_cls} vs controle"}


def f3_polyvalence() -> dict:
    p = json.loads(P317.read_text())
    prof, ordre = p["profil"], p["classement_polyvalence"]
    fig, ax = plt.subplots(figsize=(10.4, 7.2))
    xs, ys = [], []
    for a in ordre:
        q = prof[a]
        x, y = q["mean_pct"] * 100, q["dommage_max_pt"]
        xs.append(x); ys.append(y)
        if a == "G":
            ax.scatter([x], [y], s=420, marker="*", color="#c62828", edgecolor="black",
                       linewidth=1.1, zorder=5)
            ax.annotate(f"G · blob loss seul\n(dominé, dommage {y:.2f} pt,\n"
                        f"#1 sur {q['n_rang1']}/36 endpoints)",
                        (x, y), textcoords="offset points", xytext=(14, 18), fontsize=9.5,
                        color="#c62828", fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#c62828", lw=1.0))
        elif a == "moe_v3cs":
            ax.scatter([x], [y], s=190, marker="D", color="#9467bd", edgecolor="black",
                       linewidth=1.1, zorder=5)
            ax.annotate(f"MoE-V3-CS\n(seul bras T0,\ndommage {y:.2f} pt)", (x, y),
                        textcoords="offset points", xytext=(12, -34), fontsize=9.5,
                        color="#6a3d9a",
                        arrowprops=dict(arrowstyle="-", color="#9467bd", lw=1.0))
        else:
            ax.scatter([x], [y], s=88, color="#757575", alpha=0.85, zorder=4)
            ax.annotate(LABEL_COURT.get(a, a), (x, y), textcoords="offset points",
                        xytext=(8, 5), fontsize=8.8, color="#444444")
    ax.axhline(-1.0, color="#2e7d32", ls="--", lw=1.2)
    ax.axvline(50, color="#888888", ls=":", lw=1.0)
    x0, x1 = min(xs) - 4, max(xs) + 4
    ax.set_xlim(x0, x1)
    ax.set_ylim(min(ys) - 2.2, 1.2)
    ax.fill_between([x0, x1], -1.0, 1.2, color="#2e7d32", alpha=0.06, zorder=1)
    ax.text(x1 - 0.5, -0.55, "seuil T0 : aucun endpoint > 1 pt sous la référence", ha="right",
            fontsize=8.8, color="#2e7d32")
    ax.set_xlabel("percentile moyen sur 36 endpoints (%) — « bon partout » →", fontsize=10.5)
    ax.set_ylabel("dommage maximal (pt) — pire Δ d'un endpoint vs contrôle", fontsize=10.5)
    g = prof["G"]
    ax.set_title("F3 — Position de G dans le classement de polyvalence P3.17 (13 bras, 36 endpoints)\n"
                 f"G = spécialiste dominé (percentile moyen {g['mean_pct']*100:.1f} · pire rang "
                 f"{g['pire_rang']}/13 · dommage max {g['dommage_max_pt']:.2f} pt sur {g['endpoint_dommage']}) ;\n"
                 "le MoE-V3-CS est le seul bras dans la zone T0 — la spécialité de G est ce que le MoE "
                 "absorbe sans ses dégâts (papier 3)", fontsize=10.5, pad=10)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    return {"n_points": len(ordre), "x_G": g["mean_pct"] * 100,
            "y_G": g["dommage_max_pt"],
            "x_moe": prof["moe_v3cs"]["mean_pct"] * 100,
            "y_moe": prof["moe_v3cs"]["dommage_max_pt"]}


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-frags", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_DISCORD.mkdir(parents=True, exist_ok=True)

    if not args.skip_frags:
        tasks = [(a, s) for a in ("G", "B") for s in SEEDS
                 if not (MET_DIR / f"frag_{a}_seed{s}.npy").exists()]
        if tasks:
            log(f"[frag] {len(tasks)} tâches fragments (preds → count_fragments)")
            from multiprocessing import Pool
            with Pool(min(args.workers, len(tasks))) as pool:
                pool.map(fragments_task, tasks)
        else:
            log("[frag] déjà calculés")
        tasks_c = [(a, s) for a in ("G", "B", "controle") for s in SEEDS
                   if not (OUT_RAW / f"frag_classes_{a}_seed{s}.npy").exists()]
        if tasks_c:
            log(f"[frag-classes] {len(tasks_c)} tâches décomposition par classe")
            from multiprocessing import Pool
            with Pool(min(args.workers, len(tasks_c))) as pool:
                pool.map(frag_classes_task, tasks_c)
        else:
            log("[frag-classes] déjà calculées")

    log("[boot] régénération des statistiques métier, paires G_vs_B + G_vs_controle")
    tables = bootstraps_paires(args.B, args.bootstrap_seed)
    ecrire_atomique(OUT_RAW / "table_metiers_G_pairees.json", json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocole": f"holdout first:500, seeds {SEEDS}, bootstrap APPARIÉ B={args.B} "
                     f"seed {args.bootstrap_seed}, paires {PAIRS}, Holm sur la famille des 2 paires ; "
                     "masques de finiteur = stack 12 bras (identique P3.16)",
        "pairs": PAIRS, "tables": tables}, indent=1, ensure_ascii=False))
    log(f"→ {OUT_RAW/'table_metiers_G_pairees.json'}")

    log("[boot] fragments par classe (diagnostic neuf, G vs B)")
    frag_cls, frag_means = frag_classes_bootstrap(args.B, args.bootstrap_seed)
    ecrire_atomique(OUT_RAW / "table_frag_classes_G_vs_B.json", json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocole": "composantes connexes 8-connexes par classe et par image "
                     "(count_fragments, src/postprocessing/consensus.py), bootstrap apparié "
                     f"B={args.B} seed {args.bootstrap_seed}, paire G_vs_B, Holm famille 19 classes",
        "means_par_classe": {"G": frag_means["G"].tolist(), "B": frag_means["B"].tolist(),
                             "controle": frag_means["controle"].tolist()},
        "tables": frag_cls}, indent=1, ensure_ascii=False))

    log("[sanity] comparaison P3.16 / P3.10 / P3.14 / harness")
    sanites(tables)
    ecrire_atomique(OUT_RAW / "SANITY.json", json.dumps(
        {"date": datetime.now().isoformat(timespec="seconds"), "checks": SANITY},
        indent=1, ensure_ascii=False))

    p316_holm = {met: t.get("pairwise", {}) for met, t in
                 json.loads(P316_MET.read_text())["tables"].items()}

    log("[tables] T1-T6")
    t1_md, t1_d = t1_protocole_cout()
    t2_md, t2_d = t2_primaire(tables)
    t3_md, t3_csv, t3_d = t3_perclass()
    t4_md, t4_csv, t4_d = t4_metiers(tables, p316_holm, frag_cls)
    t5_md, t5_csv, t5_d = t5_polyvalence()
    t6_md, t6_d = t6_brats()
    for nom, md, csvtxt in (("T1_protocole_cout", t1_md, None), ("T2_primaire", t2_md, None),
                            ("T3_perclass", t3_md, t3_csv), ("T4_metiers", t4_md, t4_csv),
                            ("T5_polyvalence", t5_md, t5_csv), ("T6_brats_croisee", t6_md, None)):
        ecrire_atomique(OUT_TABLES / f"{nom}.md", md)
        if csvtxt:
            ecrire_atomique(OUT_TABLES / f"{nom}.csv", csvtxt)
        log(f"→ {OUT_TABLES/f'{nom}.md'}")

    sanity_md = ["# SANITY — P4.02 (régénération des statistiques du paper 4)", "",
                 f"Date : {datetime.now().isoformat(timespec='seconds')} · "
                 f"B={args.B} · bootstrap_seed={args.bootstrap_seed}", "",
                 "| Check | Résultat | Détail |", "|---|---|---|"]
    sanity_md += [f"| {c['check']} | {'✅' if c['ok'] else '❌'} | {c['detail']} |" for c in SANITY]
    sanity_md += ["", "Tous les checks sont bloquants : le script sort en erreur au premier échec, "
                  "aucune table/figure n'est écrite sur une sanité rouge.", "",
                  "Deux familles de tolérance, déclarées : (1) **bit-à-bit** vs P3.16 — mêmes npz, "
                  "mêmes masques 12 bras, mêmes index bootstrap → la régénération est une reproduction "
                  "exacte ; (2) **tolérance 1e-4 (0,01 pt)** vs harness P3.10 / master P3.14 — ces tables "
                  "proviennent d'un AUTRE forward GPU des mêmes checkpoints (non-déterminisme cuDNN/BF16, "
                  "`deterministic: false`), écart du même ordre que la sanity P3.17 déjà documentée "
                  "(mIoU consolidé = P3.14 à 4,1e-3 pt). Le primaire pré-enregistré cité dans T2 est la "
                  "table harness ; les deux sources donnent le même verdict (non significatif)."]
    ecrire_atomique(OUT_TABLES / "SANITY.md", "\n".join(sanity_md) + "\n")

    log("[figures] F1-F3")

    def save(stem: str, rap: dict) -> dict:
        """Une figure vient d'être dessinée (gcf) : png+pdf + copie discord_out + autocontrôle."""
        fig = plt.gcf()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_FIGS / f"{stem}.{ext}", dpi=170, bbox_inches="tight")
        p = OUT_FIGS / f"{stem}.png"
        rap.update({"chemin": str(p.relative_to(REPO)),
                    "octets": p.stat().st_size,
                    "png_px": list(plt.imread(p).shape[:2])})
        ecrire_atomique(OUT_DISCORD / f"{stem}.png", p.read_bytes())
        plt.close("all")
        return rap

    figs = {}
    rap = f1_forest_perclass(t3_d)
    figs["F1_forest_perclass_GvsB"] = save("F1_forest_perclass_GvsB", rap)
    rap = f2_tradeoff(t3_d, tables, t4_d["frag_classes"])
    figs["F2_tradeoff"] = save("F2_tradeoff", rap)
    rap = f3_polyvalence()
    figs["F3_polyvalence_G"] = save("F3_polyvalence_G", rap)

    for k, r in figs.items():
        log(f"→ {r['chemin']} ({r['octets']/1e6:.1f} Mo, {r['png_px'][1]}×{r['png_px'][0]} px)")
    check("F1 19 classes", figs["F1_forest_perclass_GvsB"]["n_rows"] == 19,
          f"n_rows={figs['F1_forest_perclass_GvsB']['n_rows']}")
    check("F2 panneaux 6+7+10",
          figs["F2_tradeoff"]["n_rows_gauche"] == 6 and
          figs["F2_tradeoff"]["n_rows_milieu"] == 7 and
          figs["F2_tradeoff"]["n_rows_droite"] == 10,
          f"{figs['F2_tradeoff']['n_rows_gauche']}+{figs['F2_tradeoff']['n_rows_milieu']}"
          f"+{figs['F2_tradeoff']['n_rows_droite']}")
    check("F3 11 bras", figs["F3_polyvalence_G"]["n_points"] == 11,
          f"n_points={figs['F3_polyvalence_G']['n_points']}")

    ecrire_atomique(OUT_TABLES / "paper4_tables.json", json.dumps({
        "date": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/p4_blob_tables_figures.py",
        "B": args.B, "bootstrap_seed": args.bootstrap_seed,
        "provenance": {
            "T1": ["configs/loss/ce_dice_blob.yaml", "configs/experiment/pilot_fullres_G_blob.yaml",
                   *[f"results/p3_queue/pilot_fullres_G_blob_seed{s}.log" for s in SEEDS]],
            "T2": ["results/moe_v3_cs/harness/table_Gseul_P310.json",
                   "results/moe_v3_cs/p314/master_table.json"],
            "T3": ["results/moe_v3_cs/p316/attribution_perclass_vs_B.json",
                   "results/moe_v3_cs/p316/attribution_perclass_vs_controle.json"],
            "T4": [f"results/moe_v3_cs/metiers_experts/metriques_{a}_seed{s}.npz"
                   for a in ARMS_TABLE for s in SEEDS] +
                  ["results/moe_v3_cs/metiers_experts/frag_{G,B,controle}_seed*.npy"],
            "T5": ["results/moe_v3_cs/p317_polyvalence/table_polyvalence.json"],
            "T6": ["BRATS:papers/paper3/versions/v3/NOTE_DECISION_V3.md", "zenodo:22903668"]},
        "T1": t1_d, "T2": t2_d, "T4_frag_classes": t4_d["frag_classes"],
        "T5": t5_d, "T6": t6_d,
        "figures_autocontrole": figs,
        "sanity": SANITY}, indent=1, ensure_ascii=False))
    log(f"[fin] {time.time()-t0:.0f}s — T1-T6 + F1-F3 écrits, sanités {sum(c['ok'] for c in SANITY)}/{len(SANITY)} ✅")


if __name__ == "__main__":
    main()
