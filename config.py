"""Central configuration: paths, seeds, and the model zoo.

Keep this the single source of truth for filenames and model specs so every
script (download -> extract -> eval -> align) agrees on layout.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"          # the 1,854 reference object images
FEATURES_DIR = ROOT / "features"         # cached (N, d) embedding matrices
RESULTS_DIR = ROOT / "results"           # tables, figures, metrics json

for _d in (DATA_DIR, IMAGE_DIR, FEATURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Data files (produced by src/prepare_data.py from the THINGS release).
# Triplets are 0-based with the odd-one-out in the LAST column.
CONCEPTS_FILE = DATA_DIR / "concepts.txt"          # 1,854 concept names, THINGS order
TRIPLETS_TRAIN_FILE = DATA_DIR / "triplets_train.npy"   # 4.12M
TRIPLETS_VAL_FILE = DATA_DIR / "triplets_val.npy"       # 0.45M
TRIPLETS_TEST_FILE = DATA_DIR / "triplets_test.npy"     # noise-ceiling set (testset1)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 0
N_IMAGES = 1854
HUMAN_NOISE_CEILING = 0.6731  # Hebart et al. 2023, odd-one-out
CHANCE = 1.0 / 3.0

# Image-level held-out fraction for leakage-free triplet splitting.
HELDOUT_IMAGE_FRAC = 0.10
VAL_IMAGE_FRAC = 0.10

# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
def get_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
# Each entry declares which backend loads it and the HF/timm identifier.
# backend: "hf_dinov2" | "hf_clip" | "hf_siglip" | "timm"
MODEL_ZOO: dict[str, dict] = {
    "dinov2_vitb14": {
        "backend": "hf_dinov2",
        "id": "facebook/dinov2-base",
        "family": "ssl",
    },
    "clip_vitb16": {
        "backend": "hf_clip",
        "id": "openai/clip-vit-base-patch16",
        "family": "contrastive",
    },
    "siglip_vitb16": {
        "backend": "hf_siglip",
        "id": "google/siglip-base-patch16-224",
        "family": "contrastive",
    },
    "vit_b16_sup": {
        "backend": "timm",
        "id": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "family": "supervised",
    },
    "resnet50_sup": {
        "backend": "timm",
        "id": "resnet50.a1_in1k",
        "family": "supervised",
    },
}

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
