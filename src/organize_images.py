"""Phase 0 (cont.) — pick one representative image per concept.

The THINGS image database is organized as one folder per concept, e.g.
    <raw>/aardvark/aardvark_01b.jpg
    <raw>/abacus/abacus_03n.jpg
This copies the first image of each of the 1,854 concepts to
    data/images/{concept}.jpg
in the order expected by the pipeline (concepts.txt).

Run:  python src/organize_images.py /path/to/THINGS/object_images
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data import load_concepts  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png")


def main(raw_root: Path):
    concepts = load_concepts()["concept"].tolist()
    missing = []
    for name in concepts:
        folder = raw_root / name
        if not folder.is_dir():
            missing.append(name)
            continue
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
        if not imgs:
            missing.append(name)
            continue
        dst = C.IMAGE_DIR / f"{name}.jpg"
        shutil.copy(imgs[0], dst)

    n_done = len(concepts) - len(missing)
    print(f"copied {n_done}/{len(concepts)} concept images -> {C.IMAGE_DIR}")
    if missing:
        print(f"[warn] {len(missing)} concepts had no image, e.g. {missing[:10]}")
        print("Check the raw folder layout (expected one subfolder per concept).")
    else:
        print("All 1,854 concept images resolved. Ready for src/extract_features.py")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python src/organize_images.py /path/to/THINGS/object_images")
        raise SystemExit(1)
    main(Path(sys.argv[1]).expanduser())
