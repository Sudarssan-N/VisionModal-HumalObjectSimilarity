# Project Plan — Cheap Linear Transforms for Human–Model Visual Alignment

**One-line goal:** Quantify the zero-shot human–model similarity gap on THINGS odd-one-out across a model zoo, learn cheap linear transforms to close it, and test whether those transforms transfer across architectures — all in **7 days**.

> Source design docs: `Representational Alignment Between Vision Models and Human Object Similarity (THINGS odd-one-out).md` and `representational_alignment_analysis.md`. This file is the compressed, execution-focused plan. Day-by-day progress is tracked in `EXPERIMENT_LOG.md`.

---

## Scope decisions for the 1-week sprint

To fit a week, we cut the original 6–8 week scope down to the **minimum publishable core** and defer everything optional.

**In scope (must-have):**
- THINGS images (1,854) + odd-one-out triplets (train/val/test split).
- Feature extraction for a **focused zoo of 5 backbones** (not the full menagerie):
  - `DINOv2 ViT-B/14` (self-supervised)
  - `CLIP ViT-B/16` (multimodal contrastive)
  - `SigLIP ViT-B/16` (multimodal contrastive, newer)
  - `ViT-B/16` ImageNet supervised (timm)
  - `ResNet-50` ImageNet supervised (timm)
- Zero-shot odd-one-out accuracy per backbone vs. human noise ceiling (~0.67).
- Learned L2-regularized linear transform per backbone; report aligned accuracy.
- Cross-model transfer matrix (apply W_A to model B).
- Lightweight semantic error analysis using THINGS categories + one RSA figure.

**Deferred (explicitly out of scope this week):**
- THINGS-fMRI / THINGS-MEG neural alignment.
- Hyperbolic embeddings.
- Diffusion-model backbones.
- Large model variants (ViT-L/14, etc.) — add only if Day 1–2 has slack.

**Success criterion for the week:** a reproducible pipeline + a results table (zero-shot vs aligned per model), a transfer heatmap, and one error-analysis figure — enough to draft a workshop short paper.

---

## Phases (mapped to days)

### Phase 0 — Setup & data acquisition  *(Day 1, morning)*
- Create repo structure, `requirements.txt`, fix random seeds.
- Download THINGS images (THINGSplus CC0 set, 1,854 images).
- Download THINGS odd-one-out triplet data (integer-indexed triplets).
- Build a canonical `image_index → filename → category` table.
- Define train/val/test split **at the image level** to avoid triplet leakage (hold out a set of images; any triplet touching a held-out image goes to test).
- **Exit criteria:** all data on disk, split saved as `.npy`/`.parquet`, sanity counts logged.

### Phase 1 — Feature extraction  *(Day 1 afternoon → Day 2 morning)*
- Single `extract_features.py` that loads each backbone, runs the 1,854 images through the frozen model on MPS, saves an `(1854, d)` matrix per model to `features/{model}.npy`.
- Standardize preprocessing per model (use each model's own transform/processor).
- Cache embeddings — this is a one-off.
- **Exit criteria:** one feature matrix per backbone, shapes + norms logged.

### Phase 2 — Zero-shot odd-one-out baseline  *(Day 2 afternoon)*
- Precompute the `1854×1854` cosine-similarity matrix per model (cheap).
- For each triplet (i, j, k), predict odd-one-out = the item with the lowest summed similarity to the other two; score against human majority choice.
- Report per-model zero-shot accuracy + 95% CI, plotted against the human noise ceiling.
- **Exit criteria:** baseline accuracy table committed to `EXPERIMENT_LOG.md`.

### Phase 3 — Learn linear transforms  *(Day 3 → Day 4 morning)*
- Implement transform `X' = XW` trained with a triplet objective (hinge/margin or softmax over the 3 candidate odd-one-out choices) + L2 regularization.
- Train per backbone on the train split, early-stop on val, evaluate on held-out test.
- Sweep `λ` and learning rate minimally (small grid).
- **Exit criteria:** aligned test accuracy per model; delta vs zero-shot logged.

### Phase 4 — Cross-model transfer  *(Day 4 afternoon → Day 5)*
- Handle dimensionality mismatch: either (a) PCA all models to a shared `d=512`, or (b) fit an image-correspondence linear map M projecting model B → reference space, then apply reference W. Pick one and document it.
- Build the N×N transfer matrix: rows = transform source, cols = target model, cells = test accuracy.
- **Exit criteria:** transfer heatmap figure + interpretation note.

### Phase 5 — Error analysis & RSA  *(Day 5 → Day 6)*
- Group residual errors by THINGS semantic category; compare pre/post-alignment per category.
- Build RDMs (human + each model, pre/post) with `rsatoolbox`; report RDM correlations + one RSA figure.
- **Exit criteria:** error-by-category table + RSA figure.

### Phase 6 — Writeup & figures  *(Day 7)*
- Assemble results table, transfer heatmap, RSA/error figures.
- Draft a 4-page workshop-style writeup (intro, method, results, discussion).
- Clean repo, write `README.md` with repro instructions.
- **Exit criteria:** draft + reproducible repo.

---

## Repo structure (target)

```
.
├── PROJECT_PLAN.md            # this file
├── EXPERIMENT_LOG.md          # living progress tracker
├── README.md                  # repro instructions (Phase 6)
├── requirements.txt
├── config.py                  # paths, seeds, model list
├── data/                      # THINGS images + triplets (gitignored)
├── features/                  # cached embeddings (gitignored)
├── results/                   # tables, figures, metrics json
└── src/
    ├── download_data.py
    ├── extract_features.py
    ├── zeroshot_eval.py
    ├── train_transform.py
    ├── transfer.py
    └── analysis.py
```

---

## Key risks & mitigations
- **Data download / access friction** → start Phase 0 first thing Day 1; have a fallback to precomputed THINGS embeddings if image download stalls.
- **MPS dtype/op quirks** → keep feature extraction in float32, fall back to CPU per-model if a backbone errors on MPS.
- **Triplet leakage inflating accuracy** → split at the image level, not the triplet level.
- **Transfer dimensionality mismatch** → decide PCA-to-512 vs correspondence-map on Day 4 and stick with it.
- **Scope creep** → all "Deferred" items stay deferred unless Days 1–2 finish early.

---

## Reference anchors
- Hebart et al. 2023 (eLife 82580) — THINGS behavioral odd-one-out, human noise ceiling ≈ 0.67.
- Muttenthaler et al. 2023/2024 — linear alignment of model reps to human similarity (`human_alignment` / gLocal codebases) — adapt their probe + RDM tooling.
