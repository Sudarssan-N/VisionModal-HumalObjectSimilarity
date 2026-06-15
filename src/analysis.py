"""Phase 5 — error analysis by semantic category + RSA vs the human embedding.

Two analyses:
  1. Per-category odd-one-out accuracy (pre vs post alignment). A test triplet is
     assigned to the category of its odd-one-out item (via the THINGS 27-category
     manual matrix); categories where alignment helps least are the residual gap.
  2. RSA: Spearman correlation between each model's representational similarity
     matrix (1854x1854) and the human SPoSE similarity matrix, pre vs post.

Uses scipy only (rsatoolbox not required).

Run:  python src/analysis.py
Writes results/analysis.json (+ results/rsa.png).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.align import standardize  # noqa: E402
from src.data import load_split  # noqa: E402


def _transformed(name: str):
    """Return standardized features and aligned features (if a W exists)."""
    X = np.load(C.FEATURES_DIR / f"{name}.npy").astype(np.float32)
    Xs, _, _ = standardize(X)
    W_path = C.RESULTS_DIR / f"W_{name}.npy"
    Z = Xs @ np.load(W_path) if W_path.exists() else None
    return Xs, Z


def _odd_one_out_correct(Z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Boolean per-triplet correctness (vectorized, dot-product similarity)."""
    a, b, o = t[:, 0], t[:, 1], t[:, 2]
    s_ab = np.einsum("ij,ij->i", Z[a], Z[b])
    s_ao = np.einsum("ij,ij->i", Z[a], Z[o])
    s_bo = np.einsum("ij,ij->i", Z[b], Z[o])
    return (s_ab >= s_ao) & (s_ab >= s_bo)


def category_analysis(test_t: np.ndarray, names: list[str]) -> dict:
    cat = np.load(C.DATA_DIR / "category_mat.npy")  # (1854, 27) uint8
    # Primary category per concept = first membership (0 if none).
    has = cat.sum(1) > 0
    primary = np.where(has, cat.argmax(1), -1)
    odd_cat = primary[test_t[:, 2]]  # category of the odd-one-out

    out = {}
    for name in names:
        Xs, Z = _transformed(name)
        rec = {}
        for label, feats in (("baseline", Xs), ("aligned", Z)):
            if feats is None:
                continue
            correct = _odd_one_out_correct(feats, test_t)
            per_cat = {}
            for c in range(cat.shape[1]):
                m = odd_cat == c
                if m.sum() >= 30:
                    cname = C.CATEGORY_NAMES[c] if c < len(C.CATEGORY_NAMES) else str(c)
                    per_cat[cname] = float(correct[m].mean())
            rec[label] = {"overall": float(correct.mean()), "per_category": per_cat}
        out[name] = rec
    return out


def _upper(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def rsa_analysis(names: list[str]) -> dict:
    from scipy.stats import spearmanr

    spose = np.load(C.DATA_DIR / "spose_human.npy")  # (1854, 66), non-negative
    human_sim = spose @ spose.T
    human_v = _upper(human_sim)

    out = {}
    for name in names:
        Xs, Z = _transformed(name)
        rec = {}
        for label, feats in (("baseline", Xs), ("aligned", Z)):
            if feats is None:
                continue
            f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
            sim = f @ f.T
            rho, _ = spearmanr(_upper(sim), human_v)
            rec[label] = float(rho)
        out[name] = rec
    return out


def main():
    test_t = load_split("test")
    names = [f.stem for f in sorted(C.FEATURES_DIR.glob("*.npy"))]
    if not names:
        print("No features found. Run extract_features.py first.")
        return

    print("category error analysis...")
    cat_res = category_analysis(test_t, names)
    print("RSA vs human SPoSE embedding...")
    rsa_res = rsa_analysis(names)

    print(f"\n{'model':18s} {'RSA base':>9s} {'RSA align':>10s}")
    print("-" * 40)
    for n in names:
        b = rsa_res[n].get("baseline", float("nan"))
        a = rsa_res[n].get("aligned", float("nan"))
        print(f"{n:18s} {b:9.4f} {a:10.4f}")

    results = {"category": cat_res, "rsa": rsa_res}
    (C.RESULTS_DIR / "analysis.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {C.RESULTS_DIR / 'analysis.json'}")
    _plot_rsa(rsa_res, names)


def _plot_rsa(rsa_res: dict, names: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skipping rsa plot ({e})")
        return
    base = [rsa_res[n].get("baseline", np.nan) for n in names]
    align = [rsa_res[n].get("aligned", np.nan) for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 4))
    ax.bar(x - 0.2, base, 0.4, label="baseline")
    ax.bar(x + 0.2, align, 0.4, label="aligned")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Spearman RSA vs human")
    ax.set_title("Representational similarity to human (SPoSE)")
    ax.legend()
    fig.tight_layout()
    out = C.RESULTS_DIR / "rsa.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
