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
python src/download_data.py                       # behavioral archive (412 MB)
#   unzip data/raw/osfstorage-archive.zip + full_triplet_dataset.zip
python src/prepare_data.py                        # -> concepts.txt, triplets_{train,val,test}.npy
python src/organize_images.py /path/to/THINGS     # one image per concept (after manual image DL)
python src/data.py                                # verify

# Phase 1 — extract embeddings for the model zoo (frozen, MPS/CUDA)
python src/extract_features.py

# Phase 2 — zero-shot odd-one-out baseline
python src/zeroshot_eval.py

# Phase 3 — learn cheap linear transforms (per backbone)
python src/train_transform.py

# Phase 4 — cross-model transfer matrix + heatmap
python src/transfer.py

# Phase 5 — category error analysis + RSA vs human
python src/analysis.py
```

### Validate the pipeline without images

The images need a manual license click-through. To test Phases 3–5 end-to-end
*before* getting images, run the synthetic smoke test (fabricates features from
the real human embedding, runs the real scripts, then cleans up):

```bash
python src/smoke_test.py            # expects alignment to improve acc 5/5
```

A ready-to-run Colab notebook is at [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb).

> **Apple Silicon note:** the MPS backend can intermittently crash (exit 133) on
> some ops. If that happens, force CPU: `DEVICE=cpu python src/...`. Colab (CUDA)
> is unaffected. Env knobs: `DEVICE`, `EPOCHS`, `LR`, `WEIGHT_DECAY`,
> `MAX_TRAIN`, `SHARED_DIM`, `BATCH_SIZE`.

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
config.py             paths, seeds, model zoo, device selection
src/download_data.py  Phase 0 — behavioral data (Figshare API)
src/prepare_data.py   Phase 0 — convert release to .npy + concept index
src/organize_images.py Phase 0 — one representative image per concept
src/data.py           concept index, triplet splits
src/extract_features.py  Phase 1 — embeddings per backbone
src/zeroshot_eval.py  Phase 2 — baseline odd-one-out accuracy
src/align.py          core: cosine triplet loss + linear transform training
src/train_transform.py Phase 3 — per-model alignment
src/transfer.py       Phase 4 — cross-model transfer (PCA shared space)
src/analysis.py       Phase 5 — category errors + RSA vs human SPoSE
src/smoke_test.py     end-to-end validation without images
notebooks/colab_run.ipynb   Colab runner
data/ features/ results/    (gitignored)
```
