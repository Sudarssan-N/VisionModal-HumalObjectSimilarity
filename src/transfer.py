"""Phase 4 — cross-model transfer of alignment transforms.

To compare transforms across architectures with different dimensionalities, all
models are first standardized and PCA-reduced to a shared dimensionality. A
linear transform W is learned per model in that shared space. The transfer matrix
reports test odd-one-out accuracy when applying W_source to model_target's shared
features. The diagonal is each model's own aligned accuracy.

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
from src.align import accuracy, pca_reduce, standardize, train_aligner  # noqa: E402
from src.data import load_split  # noqa: E402


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

    # Transfer matrix: rows = source transform, cols = target features.
    p = len(names)
    M = np.zeros((p, p))
    identity = {n: torch.eye(shared[n].shape[1], device=device) for n in names}
    base = {n: accuracy(identity[n], shared[n], test_pt) for n in names}
    for a, src in enumerate(names):
        for b, tgt in enumerate(names):
            M[a, b] = accuracy(W[src], shared[tgt], test_pt)

    # Report
    print(f"\nTransfer matrix (test acc); rows=source W, cols=target features")
    print("            " + "".join(f"{n[:10]:>11s}" for n in names))
    for a, src in enumerate(names):
        print(f"{src[:11]:11s} " + "".join(f"{M[a, b]:11.4f}" for b in range(p)))
    print("\nbaseline (W=I) per model:")
    for n in names:
        print(f"  {n:18s} {base[n]:.4f}")

    results = {
        "names": names, "shared_dim": shared_dim,
        "transfer_matrix": M.tolist(),
        "baseline": base,
        "diagonal_aligned": {names[i]: M[i, i] for i in range(p)},
    }
    (C.RESULTS_DIR / "transfer.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {C.RESULTS_DIR / 'transfer.json'}")

    _plot_heatmap(M, names)


def _plot_heatmap(M: np.ndarray, names: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] skipping heatmap ({e})")
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 1, 1.4 * len(names) + 1))
    im = ax.imshow(M, cmap="viridis", vmin=C.CHANCE, vmax=C.HUMAN_NOISE_CEILING)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("target features")
    ax.set_ylabel("source transform W")
    ax.set_title("Cross-model transfer (odd-one-out test acc)")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = C.RESULTS_DIR / "transfer_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
