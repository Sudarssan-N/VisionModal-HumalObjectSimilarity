"""Phase 0 (cont.) — pick one representative image per concept.

Robust to the common THINGS / THINGSplus layouts:
  * flat CC0 set:   <raw>/aardvark.jpg
  * folder set:     <raw>/aardvark/aardvark_01b.jpg
  * prefixed names: <raw>/.../aardvark_01b.jpg
It recursively scans <raw>, matches each of the 1,854 concepts to an image, and
copies it to data/images/{concept}.jpg in pipeline order (concepts.txt).

Run:  python src/organize_images.py /path/to/raw_images
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data import load_concepts  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png")


def _index_images(raw_root: Path):
    """Map candidate keys (stem, parent-dir, prefix) -> list of image paths."""
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_parent: dict[str, list[Path]] = defaultdict(list)
    by_prefix: dict[str, list[Path]] = defaultdict(list)
    for p in raw_root.rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        stem = p.stem.lower()
        by_stem[stem].append(p)
        by_parent[p.parent.name.lower()].append(p)
        # prefix before a trailing _NN / _NNx token (e.g. aardvark_01b -> aardvark)
        prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        by_prefix[prefix].append(p)
    return by_stem, by_parent, by_prefix


def main(raw_root: Path):
    if not raw_root.is_dir():
        print(f"[error] {raw_root} is not a directory")
        raise SystemExit(1)
    concepts = load_concepts()["concept"].tolist()
    by_stem, by_parent, by_prefix = _index_images(raw_root)
    total_imgs = sum(len(v) for v in by_stem.values())
    print(f"scanned {raw_root}: {total_imgs} images found")

    missing = []
    for name in concepts:
        key = name.lower()
        hit = (by_stem.get(key) or by_parent.get(key) or by_prefix.get(key))
        if not hit:
            missing.append(name)
            continue
        shutil.copy(sorted(hit)[0], C.IMAGE_DIR / f"{name}.jpg")

    n_done = len(concepts) - len(missing)
    print(f"copied {n_done}/{len(concepts)} concept images -> {C.IMAGE_DIR}")
    if missing:
        print(f"[warn] {len(missing)} concepts unresolved, e.g. {missing[:10]}")
        print("Share an `ls` of the raw folder if many are missing — the layout "
              "may need a tweak.")
    else:
        print("All 1,854 concept images resolved. Ready for src/extract_features.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/organize_images.py /path/to/raw_images")
        raise SystemExit(1)
    main(Path(sys.argv[1]).expanduser())
