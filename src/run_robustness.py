"""Robustness checks for the alignment results.

Three checks per backbone (whichever have features):
  1. Multi-seed: re-train W with several seeds -> mean +/- std aligned test acc.
  2. lambda sweep: vary L2 weight decay -> best by val, report test.
  3. Image-level split: re-split triplets so train/val/test share NO images
     (stricter than the default triplet-level split) -> confirms results are not
     a leakage artifact.

Run:  python src/run_robustness.py
      SEEDS=0,1,2 LAMBDAS=1e-4,1e-3,1e-2 MAX_TRAIN=1000000 EPOCHS=20 \
          python src/run_robustness.py
Writes results/robustness.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.align import train_aligner  # noqa: E402
from src.data import load_split, split_triplets_by_image  # noqa: E402
from src.train_transform import evaluate_test  # noqa: E402


def _env_floats(name, default):
    return [float(x) for x in os.environ.get(name, default).split(",")]


def _env_ints(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",")]


def main():
    device = C.get_device()
    seeds = _env_ints("SEEDS", "0,1,2")
    lambdas = _env_floats("LAMBDAS", "1e-4,1e-3,1e-2")
    epochs = int(os.environ.get("EPOCHS", 20))
    lr = float(os.environ.get("LR", 1e-3))
    max_train = int(os.environ.get("MAX_TRAIN", 1_000_000))
    print(f"device {device} | seeds {seeds} | lambdas {lambdas} | "
          f"epochs {epochs} | max_train {max_train}")

    train_t, val_t, test_t = (load_split(s) for s in ("train", "val", "test"))

    names = [f.stem for f in sorted(C.FEATURES_DIR.glob("*.npy"))]
    if not names:
        print("No features. Run extract_features.py first.")
        return

    only_split = os.environ.get("ONLY_SPLIT", "0") == "1"

    # Image-disjoint split (shared across models). For an apples-to-apples
    # leakage estimate we also build a size-matched RANDOM triplet-level split of
    # the SAME pool, so image-split vs triplet-split differ ONLY in whether
    # train/test share images (same distribution, same sizes).
    img = split_triplets_by_image(train_t, seed=0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train_t))
    n_tr, n_va, n_te = (len(img["train"]), len(img["val"]), len(img["test"]))
    tri = {"train": train_t[perm[:n_tr]],
           "val": train_t[perm[n_tr:n_tr + n_va]],
           "test": train_t[perm[n_tr + n_va:n_tr + n_va + n_te]]}

    results = {}
    for name in names:
        X = np.load(C.FEATURES_DIR / f"{name}.npy").astype(np.float32)
        print(f"\n=== {name} (d={X.shape[1]}) ===")
        rec = {"dim": int(X.shape[1])}

        if not only_split:
            # 1. multi-seed
            seed_accs = []
            for s in seeds:
                out = train_aligner(X, train_t, val_t, lr=lr, weight_decay=1e-3,
                                    epochs=epochs, max_train=max_train,
                                    device=device, seed=s, verbose=False)
                acc = evaluate_test(out["W"], X, out["mean"], out["std"],
                                    test_t, device)
                seed_accs.append(acc)
                print(f"  seed {s}: aligned test {acc:.4f}")
            rec["seeds"] = {"accs": seed_accs,
                            "mean": float(np.mean(seed_accs)),
                            "std": float(np.std(seed_accs))}

            # 2. lambda sweep
            sweep = {}
            best = (None, -1, -1)  # lambda, val, test
            for wd in lambdas:
                out = train_aligner(X, train_t, val_t, lr=lr, weight_decay=wd,
                                    epochs=epochs, max_train=max_train,
                                    device=device, seed=0, verbose=False)
                test = evaluate_test(out["W"], X, out["mean"], out["std"],
                                     test_t, device)
                sweep[f"{wd:g}"] = {"val": out["best_val"], "test": test}
                print(f"  lambda {wd:g}: val {out['best_val']:.4f}  test {test:.4f}")
                if out["best_val"] > best[1]:
                    best = (f"{wd:g}", out["best_val"], test)
            rec["lambda_sweep"] = {"sweep": sweep,
                                   "best_lambda": best[0], "best_test": best[2]}

        # 3. image-disjoint vs size-matched triplet-level split (isolates leakage)
        out_i = train_aligner(X, img["train"], img["val"], lr=lr, weight_decay=1e-3,
                              epochs=epochs, max_train=max_train, device=device,
                              seed=0, verbose=False)
        img_test = evaluate_test(out_i["W"], X, out_i["mean"], out_i["std"],
                                 img["test"], device)
        out_t = train_aligner(X, tri["train"], tri["val"], lr=lr, weight_decay=1e-3,
                              epochs=epochs, max_train=max_train, device=device,
                              seed=0, verbose=False)
        tri_test = evaluate_test(out_t["W"], X, out_t["mean"], out_t["std"],
                                 tri["test"], device)
        rec["image_split"] = {
            "aligned_test": img_test,
            "triplet_matched_test": tri_test,
            "leakage_gap": tri_test - img_test,
            "n_test": int(len(img["test"])),
        }
        print(f"  image-disjoint: {img_test:.4f}  vs  matched triplet-level: "
              f"{tri_test:.4f}  (leakage gap {tri_test - img_test:+.4f})")

        results[name] = rec

    # Summary
    print(f"\n{'model':16s} {'seed mean±std':>16s} {'best λ':>8s} "
          f"{'img-disjoint':>12s} {'matched-trip':>12s} {'gap':>8s}")
    print("-" * 78)
    for n, r in results.items():
        s = r.get("seeds")
        seedcell = f"{s['mean']:.4f}±{s['std']:.4f}" if s else "   (skipped)  "
        bl = r.get("lambda_sweep", {}).get("best_lambda", "—")
        im = r["image_split"]
        print(f"{n:16s} {seedcell:>16s} {bl:>8} "
              f"{im['aligned_test']:12.4f} {im['triplet_matched_test']:12.4f} "
              f"{im['leakage_gap']:+8.4f}")

    (C.RESULTS_DIR / "robustness.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {C.RESULTS_DIR / 'robustness.json'}")


if __name__ == "__main__":
    main()
