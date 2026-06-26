# How Far Can Cheap Linear Transforms Close the Human–Model Visual Similarity Gap?

Pretrained vision models judge object similarity differently from humans. This
project asks how much of that gap is just a **simple linear distortion** of an
otherwise human-like space — and whether the fix is **shared across
architectures** — using the **THINGS odd-one-out** benchmark (4.70M human triplet
judgments over 1,854 objects).

We freeze five backbones (DINOv2, CLIP, SigLIP, ViT, ResNet-50), learn a single
L2-regularized **linear transform** per model against human triplets, and measure
how far it closes the gap, whether it transfers between models, and what it leaves
behind.

> 📄 Full write-up: [`paper/main.tex`](paper/) (NeurIPS-format preprint) ·
> 📊 Auto-generated results: `results/REPORT.md` ·
> 🧪 One-click run: [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb)

---

## TL;DR — three findings

1. **Universal near-ceiling alignability.** A cheap linear transform lifts *every*
   backbone to **88–92% of the human noise ceiling** (0.59–0.62 accuracy), erasing
   large initial differences between training paradigms. The biggest gains go to
   the weakest starting points (DINOv2 **+0.20**).
2. **The correction is largely *shared* across architectures** (the main result).
   Naive cross-model transfer looks model-specific (**~19%** recovery), but that's a
   PCA-basis artifact: after a rigid, label-free orthogonal-Procrustes alignment into
   a common frame, a transform learned on one model recovers **~88% (84–92%)** of
   another's gain → an approximately *universal* linear "human-alignment direction,"
   not a per-model patch.
3. **A shared semantic ceiling.** The same abstract/functional categories (office
   supplies, furniture, musical instruments, medical equipment) resist alignment
   across all five models — a residual that linear maps can't fix.

---

## Results at a glance

**Zero-shot vs. aligned odd-one-out accuracy** (human ceiling = 0.673, chance = 0.333):

| Model | Family | Zero-shot | Aligned | Δ | % ceiling |
|---|---|---|---|---|---|
| SigLIP ViT-B/16 | contrastive | 0.468 | **0.616 ± 0.001** | +0.148 | 91.5% |
| ResNet-50 | supervised | 0.433 | 0.612 ± 0.002 | +0.179 | 91.0% |
| DINOv2 ViT-B/14 | self-sup. | 0.408 | 0.612 ± 0.003 | **+0.203** | 90.9% |
| CLIP ViT-B/16 | contrastive | 0.474 | 0.604 ± 0.001 | +0.130 | 89.7% |
| ViT-B/16 | supervised | 0.436 | 0.592 ± 0.002 | +0.155 | 87.9% |

_Aligned = mean±std over seeds {0,1,2} on full data._

**Cross-model transfer (the main result).** Mean recovery of another model's
alignment gain — naive vs. after the orthogonal-Procrustes basis-alignment control:

| Transfer setting | Mean recovery | Range |
|---|---|---|
| Raw (independent PCA) | 0.194 | 0.129–0.269 |
| **Basis-aligned (Procrustes)** | **0.881** | 0.844–0.916 |

The 19%→88% jump shows the apparent model-specificity is a PCA-basis artifact — the
human-alignment correction is largely shared across architectures.

**RSA to the human embedding** converges after alignment (Spearman):

| | CLIP | DINOv2 | ResNet | SigLIP | ViT |
|---|---|---|---|---|---|
| baseline | 0.42 | 0.14 | 0.29 | 0.32 | 0.15 |
| aligned | 0.81 | 0.79 | 0.83 | 0.83 | 0.78 |

**Robustness.** Stable across seeds (std ≤ 0.003) and L2 strength; an image-disjoint
analysis shows only a small (1–6 pt) image-leakage component, smallest for the
contrastive models.

---

## Quickstart

### Option A — Colab (recommended, fully scripted)

Open [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb) and run top to bottom.
It clones the repo, installs deps, downloads all public data **including the CC0
images** (no login/agreement), runs the full pipeline, and builds the paper PDF.

### Option B — local

```bash
pip install -r requirements.txt

# Phase 0 — data (all public, all scripted)
python src/download_data.py                         # behavioral triplets (412 MB)
unzip -o data/raw/osfstorage-archive.zip -d data/raw/extracted
unzip -o data/raw/extracted/full_triplet_dataset.zip -d data/raw/extracted/triplets
python src/prepare_data.py
python src/download_data.py --images                # CC0 images (1.1 GB, no login)
python src/organize_images.py data/raw/images_cc0
python src/data.py                                  # verify

# Phases 1–5
python src/extract_features.py     # embeddings (frozen, GPU/MPS)
python src/zeroshot_eval.py        # zero-shot baseline
python src/train_transform.py      # learn linear transforms
python src/transfer.py             # cross-model transfer + heatmap
python src/analysis.py             # category errors + RSA

# Report + robustness
python src/make_report.py                              # -> results/REPORT.md
python src/run_robustness.py && python src/make_report.py
```

### Validate without downloading images

```bash
python src/smoke_test.py    # fabricates features from the human embedding,
                            # runs the real Phase 3–5 scripts, expects 5/5 to improve
```

> **Apple Silicon:** the MPS backend can intermittently crash (exit 133); force CPU
> with `DEVICE=cpu python src/...`. Colab (CUDA) is unaffected. Env knobs:
> `DEVICE`, `EPOCHS`, `LR`, `WEIGHT_DECAY`, `MAX_TRAIN`, `SHARED_DIM`, `BATCH_SIZE`.

---

## Method (one paragraph)

For each frozen backbone we extract one embedding per image, standardize it, and
learn a square transform `W` (initialized to identity, with a learnable cosine
temperature) using the SPoSE/VICE-style triplet objective: a 3-way softmax over
candidate similar-pairs trained by cross-entropy to prefer the human choice, with
L2 regularization. We evaluate odd-one-out accuracy on a held-out test set, build
a cross-model transfer matrix in a shared PCA space, and compare representational
similarity matrices to the human SPoSE embedding before/after alignment. See the
[paper](paper/) for equations and details.

## Model zoo

| Name | Backend | Family |
|------|---------|--------|
| `dinov2_vitb14` | HF transformers | self-supervised |
| `clip_vitb16` | HF transformers | contrastive |
| `siglip_vitb16` | HF transformers | contrastive |
| `vit_b16_sup` | timm | supervised |
| `resnet50_sup` | timm | supervised |

## Data (all public)

- **Behavioral triplets** — THINGS odd-one-out, via the Figshare API ([article](https://plus.figshare.com/articles/dataset/THINGS-data_Behavioral_odd-one-out_data_and_code/20552784)).
- **Images** — THINGSplus CC0 set, one image per concept ([OSF `jum2f`](https://osf.io/jum2f/)), downloaded by `download_data.py --images`.
- **Human embedding** — SPoSE 66-d, shipped in the behavioral release.

See [`DATA_SETUP.md`](DATA_SETUP.md).

## Repository layout

```
config.py              paths, seeds, model zoo, device selection, category names
src/download_data.py   Phase 0 — behavioral data (Figshare) + CC0 images (OSF)
src/prepare_data.py    Phase 0 — convert release to .npy + concept/category/SPoSE assets
src/organize_images.py Phase 0 — one representative image per concept (robust matching)
src/data.py            concept index, triplet splits (incl. image-disjoint)
src/extract_features.py  Phase 1 — embeddings per backbone (HF/timm)
src/zeroshot_eval.py   Phase 2 — baseline odd-one-out accuracy + bootstrap CIs
src/align.py           core — cosine triplet loss + linear transform training
src/train_transform.py Phase 3 — per-model alignment
src/transfer.py        Phase 4 — cross-model transfer (PCA shared space) + heatmap
src/analysis.py        Phase 5 — category error analysis + RSA vs human SPoSE
src/run_robustness.py  seeds + λ sweep + image-disjoint leakage isolation
src/make_report.py     assemble results/*.json -> results/REPORT.md
src/smoke_test.py      end-to-end validation without images
notebooks/colab_run.ipynb   one-click Colab runner (full pipeline + paper)
paper/                 NeurIPS-format LaTeX preprint
data/ features/ results/    (gitignored)
```

## Project docs

- [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — phased plan
- [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) — running log + canonical results tables
- [`DATA_SETUP.md`](DATA_SETUP.md) — data acquisition
- [`paper/`](paper/) — the write-up

## Acknowledgements & references

Built on the THINGS initiative (Hebart et al., 2019/2023; Stoinski et al., 2024)
and the human-alignment line of Muttenthaler et al. (2023/2024). Backbones:
DINOv2, CLIP, SigLIP, ViT, ResNet. Full citations in [`paper/references.bib`](paper/references.bib).
