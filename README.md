# Cheap Linear Transforms for Human–Model Visual Alignment

How far can a single cheap linear transform close the human–model visual
similarity gap on the **THINGS odd-one-out** benchmark — and does that transform
transfer across architectures?

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the 7-day plan and
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for live progress.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# Phase 0 — data (see DATA_SETUP.md for the image click-through)
python src/download_data.py
python src/data.py                 # verify

# Phase 1 — extract embeddings for the model zoo (frozen, MPS)
python src/extract_features.py

# Phase 2 — zero-shot odd-one-out baseline
python src/zeroshot_eval.py

# Phase 3+ — linear transforms, transfer, analysis (in progress)
```

## Model zoo

| Name | Backend | Family |
|------|---------|--------|
| `dinov2_vitb14` | HF | self-supervised |
| `clip_vitb16` | HF | contrastive |
| `siglip_vitb16` | HF | contrastive |
| `vit_b16_sup` | timm | supervised |
| `resnet50_sup` | timm | supervised |

## Layout

```
config.py            paths, seeds, model zoo
src/download_data.py Phase 0 — behavioral data (Figshare API)
src/data.py          concept index, triplets, leakage-free splits
src/extract_features.py  Phase 1 — embeddings per backbone
src/zeroshot_eval.py Phase 2 — baseline odd-one-out accuracy
data/ features/ results/   (gitignored)
```
