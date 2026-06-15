"""Phase 0 — download the THINGS behavioral odd-one-out data.

The behavioral triplet data + code is mirrored on Figshare+ with a public REST
API, so we can pull the file list and download programmatically (no login). The
*images* are licensed separately and need a one-time manual click-through — see
DATA_SETUP.md.

Run:  python src/download_data.py            # list + download behavioral files
      python src/download_data.py --list     # just show available files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

# Figshare+ article: "THINGS-data: Behavioral odd-one-out data and code"
FIGSHARE_ARTICLE_ID = 20552784
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

RAW_DIR = C.DATA_DIR / "raw"


def list_files() -> list[dict]:
    r = requests.get(FIGSHARE_API, timeout=30)
    r.raise_for_status()
    return r.json().get("files", [])


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list files only")
    args = ap.parse_args()

    try:
        files = list_files()
    except Exception as e:
        print(f"[error] could not reach Figshare API: {e}")
        print("Fallback: download manually from")
        print("  https://plus.figshare.com/articles/dataset/"
              "THINGS-data_Behavioral_odd-one-out_data_and_code/20552784")
        print(f"and place files under {RAW_DIR}")
        return

    print(f"Found {len(files)} files in Figshare article {FIGSHARE_ARTICLE_ID}:")
    for f in files:
        print(f"  - {f['name']}  ({f['size'] / 1e6:.1f} MB)")

    if args.list:
        return

    for f in files:
        dest = RAW_DIR / f["name"]
        if dest.exists() and dest.stat().st_size == f["size"]:
            print(f"[skip] {f['name']} already downloaded")
            continue
        download(f["download_url"], dest)

    print(f"\nDone -> {RAW_DIR}")
    print("Next: unpack archives there, then point config.TRIPLETS_FILE /"
          " CONCEPTS_FILE at the triplet + concept files. See DATA_SETUP.md.")


if __name__ == "__main__":
    main()
