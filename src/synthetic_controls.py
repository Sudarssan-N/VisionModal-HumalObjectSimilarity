"""Synthetic controls for the cross-model transfer / Procrustes recovery claim.

The headline result (Sec. 4.2) is that a transform learned on one backbone
recovers ~88% of another's alignment gain *after* an orthogonal-Procrustes basis
alignment, vs. ~19% before. Two synthetic checks confirm that recovery is real
structure, not a free-parameter artifact of the rotation:

  (1) Shared-latent POSITIVE control. We build M synthetic "models" that are each
      a *different* noisy linear read-out of one shared latent (the SPoSE human
      embedding), so by construction the human-relevant structure is fully shared
      and differs across models only by an (unknown) linear basis. Naive transfer
      in independently-fit PCA spaces should look model-specific (low recovery);
      the orthogonal-Procrustes basis alignment should recover most of it.

  (2) Shuffled-correspondence NEGATIVE control. We refit the Procrustes rotation
      on a *permuted* concept correspondence (wrong row matching) and re-transfer.
      An orthogonal rotation with d^2 free parameters could in principle fake the
      recovery; if recovery instead collapses back toward the raw level, it is
      coming from genuine cross-model concept correspondence, not the rotation.

Generative protocol. Let Z in R^{N x k} be the standardized SPoSE embedding
(N=1854, k=66) -- the shared latent. For model m we draw a random read-out
A_m in R^{k x d_m} (i.i.d. normal) and set X_m = signal * (Z A_m) + noise_m * eps,
eps ~ N(0, I). Each model thus has the same latent under a different linear basis
plus per-model noise. Supervision/eval are the *real* THINGS triplets (the human
choices SPoSE was fit to), so "alignment" means recovering the shared latent's
geometry, exactly as in the real experiment. We then run the real Phase-4
pipeline (independent PCA -> per-model W -> raw / Procrustes / shuffled transfer).

Runs on CPU in a couple of minutes; needs only data/spose_human.npy + triplets.

Run:  python src/synthetic_controls.py
      SHARED_DIM=128 EPOCHS=10 SIGNAL=0.25 python src/synthetic_controls.py
Writes results/synthetic_controls.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.align import (accuracy, pca_reduce, procrustes_rotation,  # noqa: E402
                       standardize, train_aligner)
from src.data import load_split  # noqa: E402
from src.transfer import _recovery_fractions  # noqa: E402

# Heterogeneous output dims so the synthetic zoo mirrors the real one's mix.
SYNTH_DIMS = [128, 160, 192, 224, 256]


def make_shared_latent_features(signal: float, seed: int):
    """M models, each a different noisy linear read-out of the SPoSE latent."""
    Z = np.load(C.DATA_DIR / "spose_human.npy").astype(np.float32)   # (N, 66)
    Z, _, _ = standardize(Z)                                         # shared latent
    rng = np.random.default_rng(seed)
    feats = {}
    for i, d in enumerate(SYNTH_DIMS):
        A = rng.standard_normal((Z.shape[1], d)).astype(np.float32)  # per-model basis
        noise = 0.5 + 0.3 * (i / len(SYNTH_DIMS))                    # per-model noise
        X = signal * (Z @ A) + noise * rng.standard_normal(
            (Z.shape[0], d)).astype(np.float32)
        feats[f"synth{i}_d{d}"] = X
    return feats


def main():
    device = C.get_device()
    shared_dim = int(os.environ.get("SHARED_DIM", 128))
    epochs = int(os.environ.get("EPOCHS", 10))
    signal = float(os.environ.get("SIGNAL", 0.25))
    max_train = int(os.environ.get("MAX_TRAIN", 300_000))
    seed = int(os.environ.get("SEED", C.SEED))
    print(f"device: {device}  shared_dim: {shared_dim}  signal: {signal}  "
          f"epochs: {epochs}")

    train_t, val_t = load_split("train"), load_split("val")
    test_pt = torch.tensor(load_split("test"), dtype=torch.long, device=device)

    raw_feats = make_shared_latent_features(signal, seed)
    names = list(raw_feats)

    # Independent PCA + per-model W in the shared space (same as src/transfer.py).
    shared: dict[str, torch.Tensor] = {}
    W: dict[str, torch.Tensor] = {}
    for name in names:
        Z = pca_reduce(raw_feats[name], shared_dim, seed=seed)
        Zs, _, _ = standardize(Z)
        shared[name] = torch.tensor(Zs, dtype=torch.float32, device=device)
        out = train_aligner(Zs, train_t, val_t, epochs=epochs, max_train=max_train,
                            device=device, seed=seed, verbose=False)
        W[name] = torch.tensor(out["W"], dtype=torch.float32, device=device)
        print(f"  {name:14s} own-model val acc {out['best_val']:.4f} "
              f"(baseline {out['baseline_val']:.4f})")

    p = len(names)
    N = shared[names[0]].shape[0]
    base = {n: accuracy(torch.eye(shared_dim, device=device), shared[n], test_pt)
            for n in names}

    # One fixed wrong correspondence for the negative control.
    perm = torch.tensor(np.random.default_rng(seed).permutation(N), device=device)

    M_raw = np.zeros((p, p))
    M_aligned = np.zeros((p, p))
    M_shuffled = np.zeros((p, p))
    for a, src in enumerate(names):
        for b, tgt in enumerate(names):
            M_raw[a, b] = accuracy(W[src], shared[tgt], test_pt)
            if a == b:
                M_aligned[a, b] = M_shuffled[a, b] = M_raw[a, b]
                continue
            R = procrustes_rotation(shared[tgt], shared[src])          # correct rows
            M_aligned[a, b] = accuracy(W[src], shared[tgt] @ R, test_pt)
            R_sh = procrustes_rotation(shared[tgt][perm], shared[src])  # wrong rows
            M_shuffled[a, b] = accuracy(W[src], shared[tgt] @ R_sh, test_pt)

    rho_raw = _recovery_fractions(M_raw, base, names)
    rho_aligned = _recovery_fractions(M_aligned, base, names)
    rho_shuffled = _recovery_fractions(M_shuffled, base, names)
    mean = {
        "raw": float(np.nanmean(list(rho_raw.values()))),
        "procrustes": float(np.nanmean(list(rho_aligned.values()))),
        "procrustes_shuffled": float(np.nanmean(list(rho_shuffled.values()))),
    }

    print("\nmean recovery fraction (share of own-model alignment gain):")
    print(f"  raw (independent PCA)          {mean['raw']:+.3f}")
    print(f"  Procrustes (true correspond.)  {mean['procrustes']:+.3f}   <- positive control")
    print(f"  Procrustes (shuffled rows)     {mean['procrustes_shuffled']:+.3f}   <- negative control")
    print("\nexpected: raw LOW, Procrustes HIGH, shuffled collapses back toward raw.")

    results = {
        "names": names, "shared_dim": shared_dim, "signal": signal,
        "n_models": p, "dims": SYNTH_DIMS,
        "mean_recovery": mean,
        "recovery_fraction": {
            "raw": rho_raw, "procrustes": rho_aligned,
            "procrustes_shuffled": rho_shuffled,
        },
        "baseline": base,
        "transfer_matrix": M_raw.tolist(),
        "transfer_matrix_procrustes": M_aligned.tolist(),
        "transfer_matrix_procrustes_shuffled": M_shuffled.tolist(),
    }
    out = C.RESULTS_DIR / "synthetic_controls.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
