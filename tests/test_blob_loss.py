"""Tests non-régression BlobLoss (Kofler IPMI 2023) — port multi-classe du G BRATS.

Le test REINE est `testAlgebraEgaleBoucleDeReference` : l'astuce scatter_add en une passe
(miroir de tests/test_blob_loss_regions.py côté BRATS) doit être ÉGALE au pixel près à la
boucle naïve instance-par-instance de l'eq. 1 — c'est une réécriture algébrique, pas une
approximation. Les autres tests verrouillent : ignore_index hors du fond, instance-poids-égal
(LE but du terme), batch vide différentiable, gradients, eps, min_blob_pixels, bf16, builder.

Exécution (Tour, brats-venv — cc3d y est) :
    /home/ser/brats-venv/bin/python -m pytest tests/test_blob_loss.py -q
"""
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.losses.blob_loss import BlobLoss
from src.losses.builder import build_loss


def make_case(b=2, c=4, h=17, w=23, seed=0, ignore_frac=0.03, max_inst=3):
    """Logits aléatoires + target multi-classe avec instances multiples ET pixels ignorés."""
    g = np.random.default_rng(seed)
    target = g.integers(0, c, size=(b, h, w)).astype(np.int64)
    # Recouvrir de « blobs » compacts pour créer de vraies instances disjointes.
    for _ in range(b * c * max_inst):
        bi, ci = g.integers(0, b), g.integers(0, c)
        yy, xx = g.integers(0, h - 3), g.integers(0, w - 3)
        target[bi, yy : yy + 3, xx : xx + 3] = ci
    mask_ig = g.random((b, h, w)) < ignore_frac
    target[mask_ig] = 255
    logits = torch.from_numpy(g.normal(size=(b, c, h, w)).astype(np.float32))
    return logits, torch.from_numpy(target)


def reference_blob(logits, target, eps=1.0, ignore_index=255, min_blob_pixels=0):
    """Boucle naïve de l'eq. 1 de Kofler : une passe explicite par instance (O(voisins×instances)).

    Moyenne à DEUX NIVEAUX comme l'implémentation (fidèle à RegionBlobLoss BRATS) :
    instances → classe → image → batch.
    """
    from scipy.ndimage import label as _cc

    probs = torch.softmax(logits.float(), dim=1)
    struct = np.ones((3, 3), dtype=np.uint8)
    per_element = []
    for b in range(probs.shape[0]):
        t_b = target[b].numpy()
        valid = t_b != ignore_index
        per_class = []
        for c in range(probs.shape[1]):
            gt = (t_b == c).astype(np.uint8)
            if not gt.any():
                continue
            lab, n = _cc(gt, structure=struct)
            inst_terms = []
            for nn in range(1, n + 1):
                inst = lab == nn
                c_n = float(inst.sum())
                if min_blob_pixels > 0 and c_n < min_blob_pixels:
                    continue
                omega = valid & ((lab == 0) | inst)  # eq. 1 : tout SAUF autres instances de c
                p = probs[b, c][torch.from_numpy(omega)]
                t = torch.from_numpy(inst[omega].astype(np.float32))
                inter = (p * t).sum()
                card = (p + t).sum()
                inst_terms.append(1.0 - (2.0 * inter + eps) / (card + eps))
            if inst_terms:
                per_class.append(torch.stack(inst_terms).mean())
        if per_class:
            per_element.append(torch.stack(per_class).mean())
    if not per_element:
        return probs.sum() * 0.0
    return torch.stack(per_element).mean()


class TestAlgebre:
    def testAlgebraEgaleBoucleDeReference(self):
        for seed in range(5):
            logits, target = make_case(seed=seed)
            got = BlobLoss(eps=1.0)(logits, target)
            exp = reference_blob(logits, target, eps=1.0)
            assert torch.allclose(got, exp, atol=1e-5), f"seed {seed}: {got} != {exp}"

    def testEgalAvecEpsZero(self):
        logits, target = make_case(seed=7)
        got = BlobLoss(eps=0.0)(logits, target)
        exp = reference_blob(logits, target, eps=0.0)
        assert torch.allclose(got, exp, atol=1e-5)

    def testMinBlobPixelsFiltre(self):
        logits, target = make_case(seed=3)
        got = BlobLoss(eps=1.0, min_blob_pixels=12)(logits, target)
        exp = reference_blob(logits, target, eps=1.0, min_blob_pixels=12)
        assert torch.allclose(got, exp, atol=1e-5)

    def testFormesInvalides(self):
        with pytest.raises(ValueError):
            BlobLoss()(torch.zeros(2, 4, 8, 8), torch.zeros(2, 8, 9, dtype=torch.long))
        with pytest.raises(ValueError):
            BlobLoss()(torch.zeros(2, 4, 8, 8), torch.zeros(2, 4, 8, 8))


class TestSemantique:
    def testInstancesPoidsEgalQuelQueSoitVolume(self):
        """LE but du terme : une instance à 4 px pèse autant qu'une à 2500 px.

        Classes : 0 = fond (grande mer), 1 = grand bloc 50×50 (PARFAITEMENT prédit),
        2 = petit bloc 2×2 (totalement raté — prédit comme fond). eps=0 ⇒
        dice(classe 0) ≈ 0, dice(classe 1) ≈ 0, dice(classe 2) ≈ 1, moyenne des
        classes = 1/3. Une loss pondérée par le VOLUME donnerait ≈ 4/(4+2500+mer) ≈ 1e-3 :
        l'instance ratée y serait noyée. C'est exactement l'écart que le terme blob achète.
        """
        h = w = 64
        target = torch.zeros(h, w, dtype=torch.long)          # fond = classe 0
        target[8:58, 8:58] = 1                                # 2500 px
        target[2:4, 2:4] = 2                                  # 4 px
        logits = torch.full((3, h, w), -20.0)
        # Fond et grand bloc parfaits (canal 0 = +20 partout SAUF le grand bloc).
        logits[0] = 20.0
        logits[0, 8:58, 8:58] = -20.0
        logits[1, 8:58, 8:58] = 20.0
        # Petite instance ratée : prédite comme fond (logits[2] reste à −20 partout,
        # et le canal 0 y est déjà à +20, y compris SUR les pixels du petit bloc).
        loss = BlobLoss(eps=0.0)(logits.unsqueeze(0), target.unsqueeze(0))
        assert abs(float(loss) - 1.0 / 3.0) < 1e-3, \
            f"attendu ≈1/3 (moyenne par INSTANCE, 3 classes dont 1 ratée), reçu {loss}"

    def testIgnoreExcluDuFondEtDesInstances(self):
        """Un pixel ignoré ne compte ni comme vrai négatif ni dans S0 — sémantique `valid` BRATS."""
        logits, target = make_case(seed=11, ignore_frac=0.0)
        base = BlobLoss(eps=1.0)(logits, target)
        target2 = target.clone()
        # Ignorer des pixels de FOND (aucune instance touchée) : la perte DOIT changer
        # (S0 diminue) — mais rester finite ; et tout ignorer ⇒ 0 différentiable.
        ys, xs = (target2[0] == 0).nonzero(as_tuple=True)
        target2[0, ys[:50], xs[:50]] = 255
        mod = BlobLoss(eps=1.0)(logits, target2)
        exp = reference_blob(logits, target2, eps=1.0)
        assert torch.allclose(mod, exp, atol=1e-5)
        assert torch.isfinite(base) and torch.isfinite(mod)

    def testBatchVideZeroDifferentiable(self):
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = torch.full((2, 8, 8), 255, dtype=torch.long)
        loss = BlobLoss()(logits, target)
        assert float(loss.detach()) == 0.0
        assert loss.grad_fn is not None, "zéro non différentiable — casserait l'accumulateur"
        loss.backward()
        assert logits.grad is not None

    def testGradientFlotte(self):
        logits = torch.randn(1, 3, 16, 16, requires_grad=True)
        target = torch.zeros(1, 16, 16, dtype=torch.long)
        target[0, 3:8, 3:8] = 1
        BlobLoss()(logits, target).backward()
        assert logits.grad.abs().sum() > 0

    def testPredictionParfaiteFaible(self):
        _, target = make_case(b=1, c=4, h=32, w=32, seed=5)
        logits = torch.full((1, 4, 32, 32), -20.0)
        t = target.clone()
        t[t == 255] = 0  # position ignorée : prédire le fond, le blob l'exclut de toute façon
        logits.scatter_(1, t.unsqueeze(1).clamp(max=3), 20.0)
        loss = BlobLoss(eps=1e-6)(logits, target)
        assert float(loss.detach()) < 0.05, f"prédiction parfaite devrait ≈0, reçu {loss}"

    def testBf16EnEntree(self):
        logits, target = make_case(seed=13)
        f32 = BlobLoss()(logits, target)
        bf16 = BlobLoss()(logits.to(torch.bfloat16), target)
        assert torch.isfinite(bf16)
        assert abs(float(f32) - float(bf16)) < 5e-2  # softmax interne en float32 ⇒ dérive faible


class TestCheminRapide:
    """Le chemin pré-calculé (.blob.npz via data loader) doit être IDENTIQUE au fallback."""

    @staticmethod
    def _pack(target, n_cls, pad=8192):
        from src.losses.blob_lab import compute_blob_lab
        globs, csrs, countss = [], [], []
        for b in range(target.shape[0]):
            g, csr, c = compute_blob_lab(target[b].numpy(), n_cls, 255)
            pc = np.zeros(pad, dtype=np.int64)
            pc[: c.shape[0]] = c
            globs.append(g)
            csrs.append(csr)
            countss.append(pc)
        return {
            "glob": torch.from_numpy(np.stack(globs)),
            "csr": torch.from_numpy(np.stack(csrs)),
            "counts": torch.from_numpy(np.stack(countss)),
        }

    def testCheminRapideEgalFallback(self):
        for seed in range(5):
            logits, target = make_case(seed=seed)
            slow = BlobLoss(eps=1.0)(logits, target)
            fast = BlobLoss(eps=1.0)(logits, target, blob=self._pack(target, logits.shape[1]))
            assert torch.allclose(slow, fast, atol=1e-6), f"seed {seed}: {slow} != {fast}"

    def testCheminRapideEgalReferenceScipy(self):
        logits, target = make_case(seed=21)
        fast = BlobLoss(eps=1.0)(logits, target, blob=self._pack(target, logits.shape[1]))
        exp = reference_blob(logits, target, eps=1.0)
        assert torch.allclose(fast, exp, atol=1e-5)

    def testMinBlobPixelsCheminRapide(self):
        logits, target = make_case(seed=23)
        slow = BlobLoss(min_blob_pixels=12)(logits, target)
        fast = BlobLoss(min_blob_pixels=12)(logits, target, blob=self._pack(target, logits.shape[1]))
        assert torch.allclose(slow, fast, atol=1e-6)

    def testPaquetageManuel(self):
        """csr/counts/glob vérifiés à la main sur une cible 4×4 explicite (8-connexité)."""
        from src.losses.blob_lab import compute_blob_lab
        t = np.array([
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 2, 2],
            [255, 3, 3, 255],
        ], dtype=np.int64)
        # classe 0 : 1 instance (7 px, tout est 8-connexe) ; classe 1 : 2 instances (2 px + 1 px) ;
        # classe 2 : 1 instance (2 px) ; classe 3 : 1 instance (2 px) → N = 5
        glob, csr, counts = compute_blob_lab(t, 4, 255)
        assert csr.tolist() == [0, 1, 3, 4, 5]
        # seau 0 = fond hors instances (vide ici) ; ids : cl0→1, cl1→2,3, cl2→4, cl3→5 ; poubelle 6
        assert counts.tolist() == [0, 7, 2, 1, 2, 2, 2]
        n_total = 5
        assert (glob[t == 255] == n_total + 1).all()
        assert sorted(set(glob[t == 1].tolist())) == [2, 3]     # deux instances distinctes
        assert set(glob[t == 0].tolist()) == {1}                # classe 0 mono-instance
        assert (glob[t == 0] != 0).all() and (glob[t != 255] != 6).all()

    def testFlipEquivariance(self):
        """Le flip horizontal commute avec la perte (pré-requis du flip dataset-side)."""
        logits, target = make_case(seed=29)
        base = BlobLoss()(logits, target)
        flipped_slow = BlobLoss()(torch.flip(logits, [-1]), torch.flip(target, [-1]))
        blob = self._pack(target, logits.shape[1])
        blob_f = {
            "glob": torch.flip(blob["glob"], [-1]),  # flip = label-préservant (mêmes seaux)
            "csr": blob["csr"], "counts": blob["counts"],
        }
        flipped_fast = BlobLoss()(torch.flip(logits, [-1]), torch.flip(target, [-1]), blob=blob_f)
        assert torch.allclose(base, flipped_slow, atol=1e-6)
        assert torch.allclose(base, flipped_fast, atol=1e-6)


class TestIntegration:
    def testBuilderComposite(self):
        cfg = OmegaConf.create({
            "loss": {"type": "composite", "components": [
                {"type": "cross_entropy", "weight": 0.5, "class_weights": None, "ignore_index": 255},
                {"type": "dice", "weight": 0.5, "smooth": 1.0, "ignore_index": 255},
                {"type": "blob", "weight": 0.5, "eps": 1.0, "ignore_index": 255, "min_blob_pixels": 0},
            ]}
        })
        crit = build_loss(cfg, class_frequencies=[100.0] * 19)
        types = [c.__class__.__name__ for c in crit.components]
        assert types == ["CrossEntropyLoss", "DiceLoss", "BlobLoss"]
        assert list(crit.weights) == [0.5, 0.5, 0.5]
        # Le forward composite (pred, target) passe — le blob ne réclame pas de sdt.
        logits, target = make_case(b=1, c=19, h=16, w=16, seed=17)
        total, d = crit(logits, target)
        assert "BlobLoss" in d and torch.isfinite(total)

    def testConfigYamlChargee(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg = OmegaConf.load(os.path.join(root, "configs/loss/ce_dice_blob.yaml"))
        types = [c.type for c in cfg.components]
        assert types == ["cross_entropy", "dice", "blob"]
        blob = [c for c in cfg.components if c.type == "blob"][0]
        assert float(blob.weight) == 0.5 and float(blob.eps) == 1.0  # β BRATS, eps nnU-Net
