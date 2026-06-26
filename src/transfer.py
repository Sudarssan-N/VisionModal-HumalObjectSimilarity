"""Phase 4 — cross-model transfer of alignment transforms.

To compare transforms across architectures with different dimensionalities, all
models are first standardized and PCA-reduced to a shared dimensionality, and a
linear transform W is learned per model in that shared space. The transfer matrix
reports test odd-one-out accuracy when applying W_source to model_target's shared
features. The diagonal is each model's own aligned accuracy.

Confound control (basis alignment). Each model's PCA basis is fit *independently*,
so the shared-d axes of two models are not in a common coordinate frame. Applying
W_src to a target's features therefore pays a penalty for basis misalignment on
top of any genuinely model-specific structure, which could inflate apparent
specificity. We add an orthogonal-Procrustes control: for each (src, tgt) pair we
rotate the target's PCA basis into the source's frame -- fit on the shared concept
correspondence, using no human judgments -- before applying W_src. If transfer
stays weak after this basis alignment, weak transfer reflects model-specific
semantic structure rather than an artifact of independently-fit PCA axes.

Run:  python src/transfer.py
      SHARED_DIM=256 EPOCHS=15 MAX_TRAIN=500000 python src/transfer.py
Writes results/transfer.json and results/transfer_heatmap.png.
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


def _recovery_fractions(M: np.ndarray, base: dict, names: list[str]) -> dict:
    """Per-target recovery fraction rho_t.

    Share of a target's own-model alignment gain (over the W=I baseline) captured
    by the *average* transferred transform applied to it. Column-wise: fix target
    t, average over the source transforms W_src (s != t).
        rho_t = (mean_off_diag_t - baseline_t) / (own_model_t - baseline_t)
    """
    p = len(names)
    rho = {}
    for b, tgt in enumerate(names):
        off = float(np.mean([M[a, b] for a in range(p) if a != b]))
        denom = M[b, b] - base[tgt]
        rho[tgt] = float((off - base[tgt]) / denom) if abs(denom) > 1e-9 else float("nan")
    return rho


def main():
    device = C.get_device()
    shared_dim = int(os.environ.get("SHARED_DIM", 256))
    epochs = int(os.environ.get("EPOCHS", 30))
    lr = float(os.environ.get("LR", 1e-3))
    wd = float(os.environ.get("WEIGHT_DECAY", 1e-3))
    max_train = os.environ.get("MAX_TRAIN")
    max_train = int(max_train) if max_train else None
    print(f"device: {device}  shared_dim: {shared_dim}")

    train_t = load_split("train")
    val_t = load_split("val")
    test_t = load_split("test")
    test_pt = torch.tensor(test_t, dtype=torch.long, device=device)

    feat_files = sorted(C.FEATURES_DIR.glob("*.npy"))
    names = [f.stem for f in feat_files]
    if len(names) < 2:
        print("Need >=2 models with features for transfer.")
        return

    # Shared-space features (standardized PCA) + per-model W in that space.
    shared: dict[str, torch.Tensor] = {}
    W: dict[str, torch.Tensor] = {}
    for name in names:
        X = np.load(C.FEATURES_DIR / f"{name}.npy").astype(np.float32)
        Z = pca_reduce(X, shared_dim, seed=C.SEED)            # (N, p)
        Zs, _, _ = standardize(Z)                             # re-standardize PCA dims
        shared[name] = torch.tensor(Zs, dtype=torch.float32, device=device)
        print(f"\n=== learning W for {name} (shared {Zs.shape[1]}d) ===")
        out = train_aligner(Zs, train_t, val_t, lr=lr, weight_decay=wd,
                            epochs=epochs, max_train=max_train, device=device,
                            seed=C.SEED, verbose=True)
        W[name] = torch.tensor(out["W"], dtype=torch.float32, device=device)

    p = len(names)
    identity = {n: torch.eye(shared[n].shape[1], device=device) for n in names}
    base = {n: accuracy(identity[n], shared[n], test_pt) for n in names}

    # Two transfer matrices: rows = source transform W, cols = target features.
    #   M_raw     : apply W_src directly to target's (independently PCA'd) features.
    #   M_aligned : first rotate target into source's PCA frame (orthogonal
    #               Procrustes over the shared concepts), then apply W_src.
    M_raw = np.zeros((p, p))
    M_aligned = np.zeros((p, p))
    for a, src in enumerate(names):
        for b, tgt in enumerate(names):
            M_raw[a, b] = accuracy(W[src], shared[tgt], test_pt)
            if a == b:
                M_aligned[a, b] = M_raw[a, b]      # R == I for a model with itself
            else:
                R = procrustes_rotation(shared[tgt], shared[src])  # tgt -> src frame
                M_aligned[a, b] = accuracy(W[src], shared[tgt] @ R, test_pt)

    rho_raw = _recovery_fractions(M_raw, base, names)
    rho_aligned = _recovery_fractions(M_aligned, base, names)
    mean_raw = float(np.nanmean(list(rho_raw.values())))
    mean_aligned = float(np.nanmean(list(rho_aligned.values())))

    # Report
    def _print_matrix(title, M):
        print(f"\n{title} (test acc); rows=source W, cols=target features")
        print("            " + "".join(f"{n[:10]:>11s}" for n in names))
        for a, src in enumerate(names):
            print(f"{src[:11]:11s} " + "".join(f"{M[a, b]:11.4f}" for b in range(p)))

    _print_matrix("Raw transfer (independent PCA)", M_raw)
    _print_matrix("Basis-aligned transfer (Procrustes control)", M_aligned)
    print("\nbaseline (W=I) per model:")
    for n in names:
        print(f"  {n:18s} {base[n]:.4f}")
    print("\nrecovery fraction rho_t  (raw  ->  basis-aligned):")
    for n in names:
        print(f"  {n:18s} {rho_raw[n]:+.3f}  ->  {rho_aligned[n]:+.3f}")
    print(f"  {'MEAN':18s} {mean_raw:+.3f}  ->  {mean_aligned:+.3f}")

    results = {
        "names": names, "shared_dim": shared_dim,
        "transfer_matrix": M_raw.tolist(),
        "transfer_matrix_basis_aligned": M_aligned.tolist(),
        "baseline": base,
        "diagonal_aligned": {names[i]: M_raw[i, i] for i in range(p)},
        "recovery_fraction": rho_raw,
        "recovery_fraction_basis_aligned": rho_aligned,
        "mean_recovery": mean_raw,
        "mean_recovery_basis_aligned": mean_aligned,
    }
    (C.RESULTS_DIR / "transfer.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {C.RESULTS_DIR / 'transfer.json'}")

    _plot_heatmaps(M_raw, M_aligned, names)


def _plot_heatmaps(M_raw: np.ndarray, M_aligned: np.ndarray, names: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skipping heatmap ({e})")
        return
    fig, axes = plt.subplots(
        1, 2, figsize=(2.6 * len(names) + 2.0, 1.4 * len(names) + 1.4))
    panels = [(axes[0], M_raw, "Independent PCA (raw)"),
              (axes[1], M_aligned, "Procrustes-aligned PCA")]
    im = None
    for ax, M, title in panels:
        im = ax.imshow(M, cmap="viridis", vmin=C.CHANCE, vmax=C.HUMAN_NOISE_CEILING)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("target features")
        ax.set_ylabel("source transform W")
        ax.set_title(title)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w", fontsize=7)
    fig.suptitle("Cross-model transfer of alignment transforms (odd-one-out test acc)")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    out = C.RESULTS_DIR / "transfer_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
