"""Phase 0 (cont.) — convert the unpacked THINGS release into fast, canonical files.

Reads the unzipped Figshare archive under data/raw/extracted and writes:
  data/concepts.txt          1,854 concept names (THINGS order)
  data/triplets_train.npy    (M,3) int, odd-one-out in last column, 0-based
  data/triplets_val.npy
  data/triplets_test.npy     (noise-ceiling set, testset1)

The release already stores triplets 0-based and reordered as
[chosen_pair_a, chosen_pair_b, odd_one_out], so no reordering is needed.

Run:  python src/prepare_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

RAW = C.DATA_DIR / "raw" / "extracted"
TRIPLET_DIR = RAW / "triplets" / "triplet_dataset"
CONCEPTS_SRC = RAW / "variables" / "unique_id.txt"

SPLITS = {
    "train": TRIPLET_DIR / "trainset.txt",
    "val": TRIPLET_DIR / "validationset.txt",
    "test": TRIPLET_DIR / "testset1.txt",
}


def convert_split(name: str, src: Path) -> np.ndarray:
    if not src.exists():
        raise FileNotFoundError(f"{src} not found — unzip full_triplet_dataset.zip first")
    t = np.loadtxt(src, dtype=np.int64)
    assert t.ndim == 2 and t.shape[1] == 3, f"{name}: bad shape {t.shape}"
    lo, hi = int(t.min()), int(t.max())
    assert lo == 0 and hi == C.N_IMAGES - 1, (
        f"{name}: index range [{lo},{hi}] != [0,{C.N_IMAGES - 1}] "
        f"(expected 0-based)"
    )
    out = C.DATA_DIR / f"triplets_{name}.npy"
    np.save(out, t)
    print(f"  {name:5s}: {t.shape[0]:>9,} triplets  range[{lo},{hi}]  -> {out.name}")
    return t


def main():
    # Concepts
    names = [ln.strip() for ln in CONCEPTS_SRC.read_text().splitlines() if ln.strip()]
    assert len(names) == C.N_IMAGES, f"{len(names)} concepts, expected {C.N_IMAGES}"
    (C.DATA_DIR / "concepts.txt").write_text("\n".join(names) + "\n")
    print(f"concepts: {len(names)} -> data/concepts.txt "
          f"(first: {names[:3]}, last: {names[-1]})")

    # Triplets
    print("triplets:")
    for name, src in SPLITS.items():
        convert_split(name, src)

    print("\nPhase 0 data prep complete. Next: place THINGS images in data/images/, "
          "then run src/extract_features.py")


if __name__ == "__main__":
    main()
