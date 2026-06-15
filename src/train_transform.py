"""Phase 3 — learn a cheap linear transform per backbone to align it to humans.

For each model: standardize its embeddings, learn a square linear transform W on
the train triplets (early-stopped on val), and report odd-one-out accuracy on the
held-out test set before vs after alignment.

Run:  python src/train_transform.py                 # all models with features
      python src/train_transform.py clip_vitb16     # one model
      MAX_TRAIN=500000 EPOCHS=15 python src/train_transform.py   # quick pass
Writes results/aligned.json and results/W_{model}.npy.
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
from src.align import accuracy, standardize, train_aligner  # noqa: E402
from src.data import load_split  # noqa: E402


def evaluate_test(W: np.ndarray, X: np.ndarray, mean, std, test_t, device) -> float:
    Xs, _, _ = standardize(X, mean, std)
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    W_t = torch.tensor(W, dtype=torch.float32, device=device)
    tt = torch.tensor(test_t, dtype=torch.long, device=device)
    return accuracy(W_t, Xs_t, tt)


def main(names: list[str]):
    device = C.get_device()
    print(f"device: {device}")
    lr = float(os.environ.get("LR", 1e-3))
    wd = float(os.environ.get("WEIGHT_DECAY", 1e-3))
    epochs = int(os.environ.get("EPOCHS", 30))
    max_train = os.environ.get("MAX_TRAIN")
    max_train = int(max_train) if max_train else None

    train_t = load_split("train")
    val_t = load_split("val")
    test_t = load_split("test")

    results = {}
    for name in names:
        fp = C.FEATURES_DIR / f"{name}.npy"
        if not fp.exists():
            print(f"[skip] {name}: no features at {fp}")
            continue
        print(f"\n=== {name} ===")
        X = np.load(fp).astype(np.float32)
        out = train_aligner(
            X, train_t, val_t, lr=lr, weight_decay=wd, epochs=epochs,
            max_train=max_train, device=device, seed=C.SEED,
        )
        # Identity (standardized) baseline on test for an apples-to-apples delta.
        d = X.shape[1]
        base_test = evaluate_test(np.eye(d, dtype=np.float32), X,
                                  out["mean"], out["std"], test_t, device)
        aligned_test = evaluate_test(out["W"], X, out["mean"], out["std"],
                                     test_t, device)
        np.save(C.RESULTS_DIR / f"W_{name}.npy", out["W"])
        results[name] = {
            "dim": int(d),
            "baseline_test": base_test,
            "aligned_test": aligned_test,
            "delta": aligned_test - base_test,
            "best_val": out["best_val"],
            "pct_ceiling": 100 * aligned_test / C.HUMAN_NOISE_CEILING,
        }
        print(f"  test: baseline {base_test:.4f} -> aligned {aligned_test:.4f} "
              f"(Δ {aligned_test - base_test:+.4f})")

    if results:
        print(f"\n{'model':18s} {'base':>7s} {'aligned':>8s} {'Δ':>7s} {'%ceil':>7s}")
        print("-" * 52)
        for k, v in results.items():
            print(f"{k:18s} {v['baseline_test']:7.4f} {v['aligned_test']:8.4f} "
                  f"{v['delta']:+7.4f} {v['pct_ceiling']:6.1f}%")
        out_path = C.RESULTS_DIR / "aligned.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    requested = sys.argv[1:] or list(C.MODEL_ZOO.keys())
    main(requested)
