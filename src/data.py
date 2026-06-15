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
    """Return the 1,854-row concept table (one concept name per line).

    Row order == the image index used in the behavioral triplets.
    """
    if not C.CONCEPTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {C.CONCEPTS_FILE}. Run src/prepare_data.py (see DATA_SETUP.md)."
        )
    names = [ln.strip() for ln in C.CONCEPTS_FILE.read_text().splitlines() if ln.strip()]
    if len(names) != C.N_IMAGES:
        print(f"[warn] concepts file has {len(names)} rows, expected {C.N_IMAGES}")
    return pd.DataFrame({"index": range(len(names)), "concept": names})


def image_paths(df: pd.DataFrame | None = None) -> list[Path]:
    """Resolve the on-disk reference image for each concept, in index order.

    Looks for ``data/images/{concept}.{jpg,jpeg,png}`` (one representative image
    per concept). Falls back to a sorted directory listing if that layout is not
    present. The order MUST match the concept/triplet index order.
    """
    if df is None:
        df = load_concepts()
    exts = (".jpg", ".jpeg", ".png")
    paths: list[Path] = []
    resolved = True
    for name in df["concept"]:
        hit = next((C.IMAGE_DIR / f"{name}{e}" for e in exts
                    if (C.IMAGE_DIR / f"{name}{e}").exists()), None)
        if hit is None:
            resolved = False
            break
        paths.append(hit)
    if resolved and len(paths) == len(df):
        return paths

    print("[warn] could not resolve all images by concept name; "
          "falling back to sorted directory listing")
    files = sorted(
        p for p in C.IMAGE_DIR.iterdir() if p.suffix.lower() in exts
    )
    if len(files) != C.N_IMAGES:
        print(f"[warn] found {len(files)} images in {C.IMAGE_DIR}, "
              f"expected {C.N_IMAGES}")
    return files


# --------------------------------------------------------------------------- #
# Triplets
# --------------------------------------------------------------------------- #
_SPLIT_FILES = {
    "train": C.TRIPLETS_TRAIN_FILE,
    "val": C.TRIPLETS_VAL_FILE,
    "test": C.TRIPLETS_TEST_FILE,
}


def load_split(name: str) -> np.ndarray:
    """Load a pre-split triplet set ('train'|'val'|'test') as (M,3) int array.

    Triplets are 0-based with the odd-one-out in the last column.
    """
    fp = _SPLIT_FILES[name]
    if not fp.exists():
        raise FileNotFoundError(f"Missing {fp}. Run src/prepare_data.py first.")
    t = np.asarray(np.load(fp), dtype=np.int64)
    assert t.ndim == 2 and t.shape[1] == 3, f"bad triplet shape {t.shape}"
    return t


def load_all_splits() -> dict[str, np.ndarray]:
    return {k: load_split(k) for k in _SPLIT_FILES}


def split_triplets_by_image(
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
    print(f"concepts: {len(df)} rows (e.g. {df['concept'].iloc[:3].tolist()})")
    for name, t in load_all_splits().items():
        print(f"{name:5s}: {t.shape}, index range [{t.min()}, {t.max()}]")
