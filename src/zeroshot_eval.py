"""Phase 2 — zero-shot odd-one-out accuracy per backbone.

For each triplet (i, j, k) with k = human odd-one-out, the model predicts the
odd one out as the item with the *lowest summed similarity* to the other two
(equivalently: the pair with the highest similarity are the "two similar" ones).
We score how often the model's odd-one-out matches the human's.

Run:  python src/zeroshot_eval.py
Writes results/zeroshot.json and prints a table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data import load_split  # noqa: E402


def cosine_sim_matrix(feats: np.ndarray) -> np.ndarray:
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    return f @ f.T


def odd_one_out_accuracy(sim: np.ndarray, triplets: np.ndarray) -> float:
    """Vectorized over all triplets. Predicted odd-one-out = argmin of each
    item's summed similarity to the other two; correct if it equals column 2."""
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    s_ij = sim[i, j]
    s_ik = sim[i, k]
    s_jk = sim[j, k]
    # summed similarity to the other two, per position
    sum_i = s_ij + s_ik
    sum_j = s_ij + s_jk
    sum_k = s_ik + s_jk
    sums = np.stack([sum_i, sum_j, sum_k], axis=1)  # (M, 3)
    pred_pos = np.argmin(sums, axis=1)              # 0->i, 1->j, 2->k odd
    # human odd-one-out is always column 2 (position index 2)
    return float(np.mean(pred_pos == 2))


def bootstrap_ci(sim, triplets, n_boot=200, seed=C.SEED):
    rng = np.random.default_rng(seed)
    accs = []
    m = len(triplets)
    for _ in range(n_boot):
        idx = rng.integers(0, m, m)
        accs.append(odd_one_out_accuracy(sim, triplets[idx]))
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    triplets = load_split("test")
    print(f"test triplets: {triplets.shape}")
    results = {}
    feat_files = sorted(C.FEATURES_DIR.glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError("No features. Run extract_features.py first.")

    print(f"\n{'model':22s} {'acc':>7s}  {'95% CI':>16s}  {'% ceiling':>9s}")
    print("-" * 60)
    for fp in feat_files:
        name = fp.stem
        feats = np.load(fp)
        sim = cosine_sim_matrix(feats)
        acc = odd_one_out_accuracy(sim, triplets)
        lo, hi = bootstrap_ci(sim, triplets)
        pct = 100 * acc / C.HUMAN_NOISE_CEILING
        results[name] = {"acc": acc, "ci": [lo, hi], "pct_ceiling": pct,
                         "dim": int(feats.shape[1])}
        print(f"{name:22s} {acc:7.4f}  [{lo:.4f}, {hi:.4f}]  {pct:8.1f}%")

    print("-" * 60)
    print(f"{'human ceiling':22s} {C.HUMAN_NOISE_CEILING:7.4f}")
    print(f"{'chance':22s} {C.CHANCE:7.4f}")

    out = C.RESULTS_DIR / "zeroshot.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
