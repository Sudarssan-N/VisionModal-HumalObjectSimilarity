"""Data loading: concept index, odd-one-out triplets, and leakage-free splits.

Conventions
-----------
* A triplet is stored as a row ``(i, j, k)`` of image indices where ``k`` is the
  human-chosen *odd one out* (i and j are judged more similar to each other).
  The THINGS-data release stores the two "similar" items first and the odd one
  out last; if your source uses another convention, fix it once in
  ``load_triplets`` and everything downstream stays correct.
* Splits are made at the **image level**: a set of held-out images is chosen and
  any triplet that touches a held-out image goes to test (val likewise). This
  prevents the linear transform from seeing test images during training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


# --------------------------------------------------------------------------- #
# Concept / image index
# --------------------------------------------------------------------------- #
def load_concepts() -> pd.DataFrame:
    """Return the 1,854-row concept table.

    Auto-detects the image-filename column from a THINGSplus-style TSV. The
    returned frame is ordered to match the triplet image indices (row order ==
    index used in the behavioral data).
    """
    if not C.CONCEPTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {C.CONCEPTS_FILE}. See DATA_SETUP.md for the download step."
        )
    df = pd.read_csv(C.CONCEPTS_FILE, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    if len(df) != C.N_IMAGES:
        print(f"[warn] concepts file has {len(df)} rows, expected {C.N_IMAGES}")
    return df


def image_paths(df: pd.DataFrame | None = None) -> list[Path]:
    """Resolve the on-disk path for each concept's reference image.

    Tries common THINGS filename columns, else falls back to a sorted listing
    of ``data/images``.
    """
    if df is None:
        df = load_concepts()
    candidate_cols = [
        "image_path", "filename", "Image", "image_name", "uniqueID", "Word",
    ]
    col = next((c for c in candidate_cols if c in df.columns), None)
    if col is not None:
        paths = []
        for name in df[col].astype(str):
            p = C.IMAGE_DIR / name
            if not p.suffix:
                p = p.with_suffix(".jpg")
            paths.append(p)
        if all(p.exists() for p in paths):
            return paths
        print(f"[warn] not all images resolved from column '{col}'; "
              f"falling back to sorted directory listing")
    files = sorted(
        p for p in C.IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(files) != C.N_IMAGES:
        print(f"[warn] found {len(files)} images in {C.IMAGE_DIR}, "
              f"expected {C.N_IMAGES}")
    return files


# --------------------------------------------------------------------------- #
# Triplets
# --------------------------------------------------------------------------- #
def load_triplets() -> np.ndarray:
    """Return all triplets as an ``(M, 3)`` int array, odd-one-out in column 2.

    Accepts ``.npy`` or whitespace/comma-delimited text. Adjust here if your
    source orders the odd-one-out differently.
    """
    if C.TRIPLETS_FILE.exists():
        if C.TRIPLETS_FILE.suffix == ".npy":
            t = np.load(C.TRIPLETS_FILE)
        else:
            t = np.loadtxt(C.TRIPLETS_FILE, dtype=int)
    else:
        # Try a text file with the same stem.
        txt = C.TRIPLETS_FILE.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(
                f"Missing {C.TRIPLETS_FILE}. See DATA_SETUP.md."
            )
        t = np.loadtxt(txt, dtype=int)
    t = np.asarray(t, dtype=np.int64)
    assert t.ndim == 2 and t.shape[1] == 3, f"bad triplet shape {t.shape}"
    return t


def split_triplets(
    triplets: np.ndarray,
    n_images: int = C.N_IMAGES,
    val_frac: float = C.VAL_IMAGE_FRAC,
    test_frac: float = C.HELDOUT_IMAGE_FRAC,
    seed: int = C.SEED,
) -> dict[str, np.ndarray]:
    """Image-level split. A triplet belongs to the most-held-out partition any
    of its three images belongs to (test > val > train)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_images)
    n_test = int(round(test_frac * n_images))
    n_val = int(round(val_frac * n_images))
    test_imgs = set(perm[:n_test].tolist())
    val_imgs = set(perm[n_test:n_test + n_val].tolist())

    def partition(row: np.ndarray) -> str:
        s = set(row.tolist())
        if s & test_imgs:
            return "test"
        if s & val_imgs:
            return "val"
        return "train"

    labels = np.array([partition(r) for r in triplets])
    out = {k: triplets[labels == k] for k in ("train", "val", "test")}
    for k, v in out.items():
        print(f"  {k}: {len(v):,} triplets")
    return out


if __name__ == "__main__":
    # Smoke test once data is in place.
    df = load_concepts()
    print(f"concepts: {len(df)} rows, columns={list(df.columns)[:8]}")
    t = load_triplets()
    print(f"triplets: {t.shape}, index range [{t.min()}, {t.max()}]")
    split_triplets(t)
