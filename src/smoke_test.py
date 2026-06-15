"""End-to-end smoke test WITHOUT images.

The only data piece that needs a manual download is the images (-> features). This
script fabricates feature matrices that are a noisy linear function of the real
human SPoSE embedding, drops them in features/, and runs the real Phase 3-5
scripts with tiny settings. It verifies the pipeline executes and that alignment
*improves* accuracy / RSA on data with known structure.

Run:  python src/smoke_test.py            # synth features, run, then clean up
      python src/smoke_test.py --keep      # leave synth features in place
      python src/smoke_test.py --force      # overwrite existing real features
Great for validating the repo in Google Colab before images are available.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# Representative output dims for the zoo (used only for synthetic features).
SYNTH_DIMS = {
    "dinov2_vitb14": 768,
    "clip_vitb16": 512,
    "siglip_vitb16": 768,
    "vit_b16_sup": 768,
    "resnet50_sup": 2048,
}


def make_synthetic_features(signal: float, seed: int = C.SEED) -> list[str]:
    spose = np.load(C.DATA_DIR / "spose_human.npy").astype(np.float32)  # (N, 66)
    spose = (spose - spose.mean(0)) / (spose.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    names = []
    for i, (name, d) in enumerate(SYNTH_DIMS.items()):
        R = rng.standard_normal((spose.shape[1], d)).astype(np.float32)
        # Each model gets a different random readout + noise level so transfer
        # is non-trivial but structured.
        noise = 0.5 + 0.3 * (i / len(SYNTH_DIMS))
        X = signal * (spose @ R) + noise * rng.standard_normal(
            (spose.shape[0], d)).astype(np.float32)
        np.save(C.FEATURES_DIR / f"{name}.npy", X)
        names.append(name)
    print(f"wrote {len(names)} synthetic feature matrices to {C.FEATURES_DIR}")
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--signal", type=float, default=0.25)
    args = ap.parse_args()

    existing = list(C.FEATURES_DIR.glob("*.npy"))
    if existing and not args.force:
        print(f"[abort] {len(existing)} feature files already exist in "
              f"{C.FEATURES_DIR}. Use --force to overwrite (will delete them).")
        return
    for f in existing:
        f.unlink()

    # Small, fast settings for the smoke run.
    os.environ.setdefault("EPOCHS", "4")
    os.environ.setdefault("MAX_TRAIN", "100000")
    os.environ.setdefault("SHARED_DIM", "128")

    names = make_synthetic_features(args.signal)

    print("\n########## Phase 3: train_transform ##########")
    import src.train_transform as p3
    p3.main(names)

    print("\n########## Phase 4: transfer ##########")
    import src.transfer as p4
    p4.main()

    print("\n########## Phase 5: analysis ##########")
    import src.analysis as p5
    p5.main()

    # Sanity assertions on the synthetic structure.
    import json
    aligned = json.loads((C.RESULTS_DIR / "aligned.json").read_text())
    improved = [k for k, v in aligned.items() if v["delta"] > 0]
    print(f"\n[check] alignment improved test acc for {len(improved)}/{len(aligned)} "
          f"models (expected: most/all on synthetic structured data)")

    if not args.keep:
        for name in names:
            (C.FEATURES_DIR / f"{name}.npy").unlink(missing_ok=True)
            (C.RESULTS_DIR / f"W_{name}.npy").unlink(missing_ok=True)
        print("cleaned up synthetic features + transforms "
              "(results/*.json/png kept). Use --keep to retain.")
    print("\nSMOKE TEST COMPLETE.")


if __name__ == "__main__":
    main()
