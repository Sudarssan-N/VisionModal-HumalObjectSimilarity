"""Phase 3 — learn a cheap linear transform per backbone to align it to humans.

For each model: standardize its embeddings, learn a square linear transform W on
the train triplets (early-stopped on val), and report odd-one-out accuracy on the
held-out test set before vs after alignment.

Run:  python src/train_transform.py                 # all models with features
      python src/train_transform.py clip_vitb16     # one model
      MAX_TRAIN=500000 EPOCHS=15 python src/train_transform.py   # quick pass
      SEEDS=0,1,2 python src/train_transform.py      # full-data CIs (mean±std)
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
    seeds_env = os.environ.get("SEEDS")
    seeds = [int(s) for s in seeds_env.split(",")] if seeds_env else [C.SEED]
    if len(seeds) > 1:
        print(f"seeds: {seeds} (reporting full-data mean±std)")

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
        d = X.shape[1]

        per_seed, first = [], None
        for s in seeds:
            out = train_aligner(
                X, train_t, val_t, lr=lr, weight_decay=wd, epochs=epochs,
                max_train=max_train, device=device, seed=s,
                verbose=(len(seeds) == 1),
            )
            acc_s = evaluate_test(out["W"], X, out["mean"], out["std"],
                                  test_t, device)
            per_seed.append(acc_s)
            first = first or out
            if len(seeds) > 1:
                print(f"  seed {s}: aligned test {acc_s:.4f}")

        # Identity (standardized) baseline on test for an apples-to-apples delta;
        # standardization stats are seed-independent, so compute once.
        base_test = evaluate_test(np.eye(d, dtype=np.float32), X,
                                  first["mean"], first["std"], test_t, device)
        aligned_mean = float(np.mean(per_seed))
        aligned_std = float(np.std(per_seed))
        np.save(C.RESULTS_DIR / f"W_{name}.npy", first["W"])   # W from first seed
        results[name] = {
            "dim": int(d),
            "baseline_test": base_test,
            "aligned_test": aligned_mean,            # mean over seeds (== value if 1 seed)
            "aligned_test_std": aligned_std,
            "aligned_test_seeds": per_seed,
            "n_seeds": len(seeds),
            "delta": aligned_mean - base_test,
            "best_val": first["best_val"],
            "pct_ceiling": 100 * aligned_mean / C.HUMAN_NOISE_CEILING,
        }
        pm = f" ± {aligned_std:.4f}" if len(seeds) > 1 else ""
        print(f"  test: baseline {base_test:.4f} -> aligned {aligned_mean:.4f}{pm} "
              f"(Δ {aligned_mean - base_test:+.4f})")

    if results:
        print(f"\n{'model':18s} {'base':>7s} {'aligned':>8s} {'std':>7s} "
              f"{'Δ':>7s} {'%ceil':>7s}")
        print("-" * 60)
        for k, v in results.items():
            print(f"{k:18s} {v['baseline_test']:7.4f} {v['aligned_test']:8.4f} "
                  f"{v.get('aligned_test_std', 0.0):7.4f} {v['delta']:+7.4f} "
                  f"{v['pct_ceiling']:6.1f}%")
        out_path = C.RESULTS_DIR / "aligned.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    requested = sys.argv[1:] or list(C.MODEL_ZOO.keys())
    main(requested)
