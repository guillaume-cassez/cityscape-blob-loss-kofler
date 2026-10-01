#!/usr/bin/env python3
"""p4_check_numbers.py — cohérence manuscrit paper4 ↔ artefacts (gate SPEC ## Tests).

Vérifie que les chiffres clés cités dans papers/paper4/publish/repo/paper.md (EN) et
paper_fr.md (FR) sont EXACTEMENT ceux des artefacts sources (harness P3.10, tables
P4.02, T3/T4/T5 csv, décomposition fragments). Toute divergence fait échouer le script
(exit 1) — le manuscrit ne peut pas dériver des tables en silence.

Usage : python3 scripts/p4_check_numbers.py   (depuis la racine du dépôt)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "papers" / "paper4" / "publish" / "repo"
MINUS = "\u2212"  # − U+2212 utilisé dans les manuscrits (pas le trait d'ASCII)

FAILS, N = [], 0


def load(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def norm_fr(txt):
    """FR : décimales à virgule -> point, pour comparaison numérique."""
    import re
    return re.sub(r"(\d),(\d)", r"\1.\2", txt)


EN = load(BUNDLE / "paper.md")
FR = norm_fr(load(BUNDLE / "paper_fr.md"))


def fmt(v, nd, signed=False, pct=False):
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    return s.replace("-", MINUS)


def check(label, en_str, fr_str=None):
    """en_str doit apparaître dans paper.md ; fr_str (défaut = en_str avec
    décimales normalisées) dans paper_fr.md (après normalisation virgules)."""
    global N
    N += 1
    fr_str = fr_str if fr_str is not None else en_str
    if en_str not in EN:
        FAILS.append(f"[EN] {label}: attendu « {en_str} » absent de paper.md")
    if fr_str not in FR:
        FAILS.append(f"[FR] {label}: attendu « {fr_str} » absent de paper_fr.md")


def check_fr_only(label, fr_str):
    global N
    N += 1
    if fr_str not in FR:
        FAILS.append(f"[FR] {label}: attendu « {fr_str} » absent de paper_fr.md")


# --------------------------------------------------------------------------- 1. PRIMAIRE (harness P3.10)
h = json.load(open(ROOT / "results/moe_v3_cs/harness/table_Gseul_P310.json"))
bm = h["bootstrap_miou"]
pw = bm["pairwise"]["B_vs_G"]
g_pt, b_pt = bm["point"]["G"] * 100, bm["point"]["B"] * 100
d_pt = -pw["delta"] * 100              # B_vs_G -> G−B
lo, hi = -pw["ci95"][1] * 100, -pw["ci95"][0] * 100
pval = pw["p_two_sided"]
check("primaire point G", fmt(g_pt, 3))
check("primaire point B", fmt(b_pt, 3))
check("primaire delta", f"{fmt(d_pt, 3, signed=True)} pt")
check("primaire CI lo", f"[{fmt(lo, 3, signed=True)} ; {fmt(hi, 3, signed=True)}]")
check("primaire p", f"p = {fmt(pval, 4)}", f"p = {fmt(pval, 4)}")
seeds_g = h["arms"]["G"]["miou_dataset_par_seed"]
seeds_b = h["arms"]["B"]["miou_dataset_par_seed"]
for s in ("42", "123", "456"):
    check(f"seed G {s}", fmt(seeds_g[s] * 100, 3))
    check(f"seed B {s}", fmt(seeds_b[s] * 100, 3))

# --------------------------------------------------------------------------- 2. RÉGÉNÉRATION (paper4_tables.json)
pt = json.load(open(ROOT / "papers/paper4/tables/paper4_tables.json"))
pr = pt["T2"]["primaire"]
check("regen delta", f"{fmt(pr['delta'] * 100, 3, signed=True)}")
check("regen p", f"{pr['p']:.4f}")

# --------------------------------------------------------------------------- 3. PER-CLASSE (T3 csv)
t3 = {r["classe"]: r for r in csv.DictReader(open(ROOT / "papers/paper4/tables/T3_perclass.csv"))}
for cl, nd in [("traffic light", 2), ("pole", 2), ("bicycle", 2), ("truck", 2), ("train", 2)]:
    d = float(t3[cl]["delta_GvsB_pt"])
    check(f"perclass {cl}", fmt(d, nd, signed=True))

# --------------------------------------------------------------------------- 4. MÉTIERS (T4 csv)
t4 = {r["metrique"]: r for r in csv.DictReader(open(ROOT / "papers/paper4/tables/T4_metiers.csv"))}
pairs = [("rappel_strict_instances", 3), ("instances_taille_T1_rappel", 3),
         ("instances_foule_rappel", 3), ("precision_ped_pixels", 3),
         ("boundary_f1_3px", 3), ("IoU_person", 3)]
for m, nd in pairs:
    d = float(t4[m]["delta_GvsB_pt"])
    check(f"metier {m}", fmt(d, nd, signed=True))
frag = t4["fragments"]
check("fragments G", fmt(float(frag["point_G"]), 2))
check("fragments B", fmt(float(frag["point_B"]), 2))
ratio = float(frag["point_G"]) / float(frag["point_B"])
check("fragments ratio", f"×{ratio:.2f}")

# --------------------------------------------------------------------------- 5. POLYVALENCE (T5 csv)
t5 = {r["bras"]: r for r in csv.DictReader(open(ROOT / "papers/paper4/tables/T5_polyvalence.csv"))}
g5 = t5["G"]
check("T5 mean_pct", f"{float(g5['mean_pct']):.1f}")
check("T5 dommage", f"{fmt(float(g5['dommage_max_pt']), 2)}")
check("T5 z", f"{fmt(float(g5['z']), 2, signed=True)}")
check("T5 pic", f"{fmt(float(g5['pic_max_pt']), 2, signed=True)}")
check("T5 pire rang", f"{g5['pire_rang']}/13")
prix = float(g5["dommage_max_pt"]) / float(g5["delta_miou_pt"])
check("T5 prix/mIoU", f"{fmt(prix, 1)}")
moe = t5["moe_v3cs"]
check("T5 moe dommage", f"{fmt(float(moe['dommage_max_pt']), 2)}")
check("T5 moe dmiou", f"{fmt(float(moe['delta_miou_pt']), 2, signed=True)}")
check("T5 moe p", f"{float(moe['p_miou']):.4f}")
prix_moe = float(moe["dommage_max_pt"]) / float(moe["delta_miou_pt"])
check("T5 moe prix/mIoU", f"{fmt(prix_moe, 1)}")

# --------------------------------------------------------------------------- 6. FRAGMENTS PAR CLASSE (json dédié)
fc = json.load(open(ROOT / "results/moe_v3_cs/paper4_blob/table_frag_classes_G_vs_B.json"))
car = fc["tables"]["car"]["pairwise"]["G_vs_B"]["delta"]
per_g = fc["tables"]["person"]["point"]["G"]
per_b = fc["tables"]["person"]["point"]["B"]
ter = fc["tables"]["terrain"]["pairwise"]["G_vs_B"]["delta"]
check("frag car", f"{fmt(car, 1, signed=True)}")
check("frag person G", fmt(per_g, 1))
check("frag person B", fmt(per_b, 1))
check("frag person ratio", f"×{per_g / per_b:.1f}")
check("frag terrain", f"{fmt(ter, 1, signed=True)}")

# --- 6bis. COMPTES de classes (ajouté le 2026-10-01 : c'est l'absence de ces checks qui a
# laissé passer « 18 of 19 classes » alors que l'artefact en donne 16. Les comptes sont
# CALCULÉS depuis l'artefact, puis on exige que le manuscrit porte exactement ce rendu.) ---
means = fc["means_par_classe"]
classes = list(fc["tables"])
deltas = {c: fc["tables"][c]["pairwise"]["G_vs_B"] for c in classes}
holms = {c: fc["tables"][c]["pairwise"]["G_vs_B"]["p_holm"] for c in classes}

n_hausse_B = sum(1 for c in classes if deltas[c]["delta"] > 0)
n_hausse_holm = sum(1 for c in classes if deltas[c]["delta"] > 0 and holms[c] < 0.05)
n_baisse = sum(1 for c in classes if deltas[c]["delta"] <= 0)
n_hausse_ctrl = sum(1 for c in classes if means["G"][classes.index(c)] > means["controle"][classes.index(c)])
somme_G = sum(means["G"])
somme_B = sum(means["B"])
ratio = somme_G / somme_B
baisses_materielles = [c for c in classes if deltas[c]["delta"] <= 0 and holms[c] < 0.05]

# sanity des comptes eux-mêmes (sinon un artefact vide ferait « passer » le manuscrit)
assert len(classes) == 19, f"artefact fragments : {len(classes)} classes, 19 attendues"
assert baisses_materielles == ["terrain"], \
    f"baisses Holm-significatives attendues ['terrain'], mesuré {baisses_materielles}"

check("frag nb classes en hausse vs B", f"{n_hausse_B} of 19", f"{n_hausse_B} classes sur 19")
check("frag nb hausses Holm-significatives",
      f"{n_hausse_holm} Holm-significant rises", f"{n_hausse_holm} hausses Holm-significatives")
check("frag nb classes en hausse vs controle", f"{n_hausse_ctrl} of 19", f"{n_hausse_ctrl} sur 19")
check("frag somme G", fmt(somme_G, 1))
check("frag somme B", fmt(somme_B, 1))
check("frag ratio", f"×{ratio:.2f}")
check("frag seule baisse materielle", f"{fmt(deltas['terrain']['delta'], 1, signed=True)}")


def absent(label, *motifs):
    """Garde anti-régression : ces formulations NE doivent plus apparaître."""
    global N
    N += 1
    trouves = []
    for m in motifs:
        if m in EN:
            trouves.append(f"EN:{m!r}")
        if m in FR:
            trouves.append(f"FR:{m!r}")
    if trouves:
        FAILS.append(f"[REGRESSION] {label}: {', '.join(trouves)}")


absent("compte de classes erroné (l'artefact donne "
       f"{n_hausse_B}/{len(classes)} vs B et {n_hausse_ctrl}/{len(classes)} vs contrôle)",
       "18 of 19 classes", "18 classes of 19", "18 classes sur 19", "18 sur 19", "18/19",
       "only terrain decreases", "seul terrain baisse")


# --------------------------------------------------------------------------- 7. COÛTS (T1 / logs) — constantes vérifiées par P4.02
for en, fr in [("656 s/epoch", "656 s/époque"), ("+870 s/epoch", "+870 s/époque"),
               ("22.4 ms/image", "22,4 ms/image"), ("31.5 GB", "31,5 Go"),
               ("29.2 h", "29,2 h"), ("0.3743 / 0.3735 / 0.3765", "0,3743 / 0,3735 / 0,3765")]:
    check(f"cout {en}", en, fr.replace(",", "."))

# --------------------------------------------------------------------------- 8. BRATS (citations T6)
check("brats seul", f"{MINUS}0.00376", f"{MINUS}0.00376")
check("brats expert", "+0.00566", "+0.00566")

# --------------------------------------------------------------------------- rapport
if FAILS:
    print(f"ÉCHEC — {len(FAILS)} divergence(s) sur {N} checks :")
    for f in FAILS:
        print("  " + f)
    sys.exit(1)
print(f"OK — {N} checks : manuscrits EN/FR alignés sur les artefacts.")
sys.exit(0)
