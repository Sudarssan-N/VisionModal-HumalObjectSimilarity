"""Phase 1 — extract one embedding per image for each backbone in the zoo.

Run:  python src/extract_features.py            # all models
      python src/extract_features.py clip_vitb16 dinov2_vitb14

Saves features/{model}.npy of shape (N_IMAGES, d) in dataset image order.
Frozen, no grad. Each model uses its own native preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data import image_paths  # noqa: E402


# --------------------------------------------------------------------------- #
# Per-backend loaders. Each returns (preprocess_fn, embed_fn).
#   preprocess_fn: PIL.Image -> tensor (C,H,W)
#   embed_fn: batched tensor (B,C,H,W) -> tensor (B, d)
# --------------------------------------------------------------------------- #
def _load_timm(model_id: str, device: str):
    import timm
    from timm.data import resolve_data_config, create_transform

    model = timm.create_model(model_id, pretrained=True, num_classes=0).eval().to(device)
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg)

    @torch.no_grad()
    def embed(batch):
        return model(batch.to(device))  # num_classes=0 -> pooled features

    return tf, embed


def _load_hf_dinov2(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    def tf(img: Image.Image):
        return proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(batch):
        out = model(pixel_values=batch.to(device))
        return out.pooler_output if out.pooler_output is not None \
            else out.last_hidden_state[:, 0]

    return tf, embed


def _load_hf_clip(model_id: str, device: str):
    from transformers import CLIPModel, CLIPImageProcessor

    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).eval().to(device)

    def tf(img: Image.Image):
        return proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(batch):
        return model.get_image_features(pixel_values=batch.to(device))

    return tf, embed


def _load_hf_siglip(model_id: str, device: str):
    from transformers import SiglipModel, SiglipImageProcessor

    proc = SiglipImageProcessor.from_pretrained(model_id)
    model = SiglipModel.from_pretrained(model_id).eval().to(device)

    def tf(img: Image.Image):
        return proc(images=img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def embed(batch):
        return model.get_image_features(pixel_values=batch.to(device))

    return tf, embed


_BACKENDS = {
    "timm": _load_timm,
    "hf_dinov2": _load_hf_dinov2,
    "hf_clip": _load_hf_clip,
    "hf_siglip": _load_hf_siglip,
}


def _as_tensor(out) -> torch.Tensor:
    """Coerce a model output to a (B, d) tensor across transformers versions.

    Recent transformers can return a ModelOutput from get_image_features instead
    of a bare tensor; handle both."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0]
    raise TypeError(f"cannot extract embedding tensor from {type(out)}")


def extract_one(name: str, spec: dict, device: str) -> np.ndarray:
    out_path = C.FEATURES_DIR / f"{name}.npy"
    if out_path.exists():
        print(f"[skip] {name}: {out_path} exists")
        return np.load(out_path)

    tf, embed = _BACKENDS[spec["backend"]](spec["id"], device)
    paths = image_paths()
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush():
        if not batch:
            return
        x = torch.stack(batch)
        z = _as_tensor(embed(x)).float().cpu().numpy()
        feats.append(z)
        batch.clear()

    for p in tqdm(paths, desc=name):
        img = Image.open(p).convert("RGB")
        batch.append(tf(img))
        if len(batch) >= C.BATCH_SIZE:
            flush()
    flush()

    feats = np.concatenate(feats, axis=0)
    assert feats.shape[0] == len(paths), (feats.shape, len(paths))
    np.save(out_path, feats)
    print(f"[ok] {name}: {feats.shape} -> {out_path} "
          f"(mean L2 norm {np.linalg.norm(feats, axis=1).mean():.2f})")
    return feats


def main(names: list[str]):
    device = C.get_device()
    print(f"device: {device}")
    for name in names:
        spec = C.MODEL_ZOO[name]
        try:
            extract_one(name, spec, device)
        except Exception as e:  # MPS op gaps -> retry on CPU
            print(f"[warn] {name} failed on {device} ({e}); retrying on cpu")
            extract_one(name, spec, "cpu")


if __name__ == "__main__":
    requested = sys.argv[1:] or list(C.MODEL_ZOO.keys())
    main(requested)
