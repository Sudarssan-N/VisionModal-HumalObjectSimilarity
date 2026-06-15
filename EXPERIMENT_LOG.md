# Experiment Log — Human–Model Visual Alignment (THINGS odd-one-out)

Living progress tracker. Append newest entries at the top of the **Daily log**. Keep the **Status board** and **Results** tables current. See `PROJECT_PLAN.md` for the phase definitions.

---

## Status board

| Phase | Description | Status | Day target | Notes |
|-------|-------------|--------|------------|-------|
| 0 | Setup & data acquisition | 🟡 In progress | Day 1 | Behavioral data done + verified; only THINGS images (manual DL) pending |
| 1 | Feature extraction (5 backbones) | ⬜ Not started | Day 1–2 | Script ready; blocked on images |
| 2 | Zero-shot odd-one-out baseline | ⬜ Not started | Day 2 | |
| 3 | Learn linear transforms | 🟡 Code done | Day 3–4 | `train_transform.py` validated on synthetic; needs real features |
| 4 | Cross-model transfer | 🟡 Code done | Day 4–5 | `transfer.py` validated on synthetic; needs real features |
| 5 | Error analysis & RSA | 🟡 Code done | Day 5–6 | `analysis.py` validated on synthetic; needs real features |
| 6 | Writeup & figures | ⬜ Not started | Day 7 | |

Legend: ⬜ Not started · 🟡 In progress · ✅ Done · ⚠️ Blocked

---

## Key numbers to beat / reference
- Human noise ceiling (odd-one-out): **≈ 0.673**
- Chance accuracy (3-way): **0.333**

---

## Results

### Table 1 — Zero-shot vs aligned odd-one-out accuracy (test split)
| Model | Family | d | Zero-shot acc | Aligned acc | Δ | % of ceiling (aligned) |
|-------|--------|---|---------------|-------------|---|------------------------|
| DINOv2 ViT-B/14 | SSL | — | — | — | — | — |
| CLIP ViT-B/16 | Contrastive | — | — | — | — | — |
| SigLIP ViT-B/16 | Contrastive | — | — | — | — | — |
| ViT-B/16 (IN sup.) | Supervised | — | — | — | — | — |
| ResNet-50 (IN sup.) | Supervised | — | — | — | — | — |

### Table 2 — Cross-model transfer (test acc; row = transform source, col = target)
| W \ target | DINOv2 | CLIP | SigLIP | ViT | ResNet |
|-----------|--------|------|--------|-----|--------|
| DINOv2 | — | — | — | — | — |
| CLIP | — | — | — | — | — |
| SigLIP | — | — | — | — | — |
| ViT | — | — | — | — | — |
| ResNet | — | — | — | — | — |

### Error analysis / RSA
- RDM correlation (human vs model, pre/post): _TBD_
- Hardest semantic categories post-alignment: _TBD_

---

## Decisions & deviations
_Record any scope changes, hyperparameter choices, or assumptions here so results stay reproducible._
- _(none yet)_

---

## Open questions / blockers
- _(none yet)_

---

## Daily log

### Day 1 — First REAL result (DINOv2) + full automation
- **Images turned out to be fully scriptable**: THINGSplus CC0 set (1 image/concept) downloads directly from OSF (`osf.io/download/wb36u/`), no login/agreement. Added `download_data.py --images` + robust `organize_images.py`. Confirmed working (DINOv2 got real features).
- **Real DINOv2 numbers** (test set, noise ceiling 0.673):
  - Zero-shot **0.408** (CI 0.401–0.417) = 60.7% of ceiling — matches published DINOv2 results.
  - Aligned **0.609** = **+0.20**, 90% of ceiling. Cheap linear transform closes most of the gap; consistent with Muttenthaler et al. gLocal (~0.60–0.64).
- Added verified THINGS **27 category names** (alphabetical, reconciled vs metadata 50057/50058 cells) → `config.CATEGORY_NAMES`; `analysis.py` now labels categories by name.
- Added `make_report.py` → `results/REPORT.md` (Table 1, transfer matrix, RSA, hardest categories). Wired into Colab notebook.
- **Next:** finish extracting the other 4 backbones, then rerun zeroshot/train_transform/transfer/analysis/make_report for the full 5-model comparison. (`transfer.json` still stale-synthetic until all 5 features exist.)

### Day 1 — Phases 3–5 built + validated (no images yet)
- Wrote `src/align.py` (core): cosine-similarity triplet loss (SPoSE/VICE-style 3-way softmax over pairs), linear transform with **learnable temperature**, early-stopped training.
  - First attempt used raw dot products → softmax saturated, gradients blew up, val acc *dropped*. Fixed by switching to L2-normalized cosine + learnable `logit_scale` (CLIP-style). Now stable and consistent with the Phase 2 cosine baseline.
- `train_transform.py` (Phase 3), `transfer.py` (Phase 4, PCA→shared 256d + heatmap), `analysis.py` (Phase 5, category errors + RSA vs SPoSE using scipy, no rsatoolbox).
- `smoke_test.py`: fabricates features as a noisy linear function of the real human embedding, runs the real Phase 3–5 scripts, asserts alignment improves. **Validated end-to-end (CPU): alignment improved test acc 5/5 (+0.034–0.059), RSA 0.69→0.85, transfer matrix + plots produced.**
- Added `notebooks/colab_run.ipynb` for GPU runs.
- **macOS MPS instability:** intermittent exit-133 native crash on MPS; added `DEVICE` override (use `DEVICE=cpu` locally). Colab/CUDA unaffected — this is the intended run target.
- **Next:** get THINGS images → `extract_features.py` → run Phases 2–5 on real features.

### Day 1 — Environment + behavioral data
- **Requirements:** verified torch 2.3.1 / transformers 4.39.3 / sklearn 1.4 / etc. already present. Installed `timm` 1.0.27; removed a pre-existing broken `wandb` (protobuf mismatch) that timm imports. `rsatoolbox` deferred (needs sklearn≥1.6) — documented; Phase 5 only.
- **Download:** pulled THINGS behavioral archive via Figshare API (412 MB), unzipped, ran `src/prepare_data.py`.
  - Confirmed triplet convention: **0-based, odd-one-out in last column** (matches pipeline).
  - Splits: train 4,120,663 / val 453,642 / test (noise-ceiling) 15,640. Index range validated [0,1853].
  - Concept index: 1,854 names from `unique_id.txt` → `data/concepts.txt`.
- Wired `config.py`/`data.py` to real files; `python src/data.py` smoke test passes; all module imports OK.
- Added `src/organize_images.py` to pick one image/concept once THINGS images are downloaded.
- **Blocker:** THINGS images need a manual license click-through (can't be scripted). Everything else is ready.
- **Next:** user downloads THINGSplus images → `organize_images.py` → `extract_features.py` (Phase 1).

### Day 1 — Scaffold
- Initialized git repo, pushed to `github.com/Sudarssan-N/VisionModal-HumalObjectSimilarity` (`main`).
- Built runnable Phase 0–2 foundation:
  - `config.py` (paths, seeds, 5-model zoo), `src/data.py` (concept index, triplets, **image-level** leakage-free split).
  - `src/download_data.py` (behavioral data via public Figshare API).
  - `src/extract_features.py` (per-backend adapters for DINOv2/CLIP/SigLIP via HF, ViT/ResNet via timm; MPS with CPU fallback).
  - `src/zeroshot_eval.py` (vectorized odd-one-out accuracy + bootstrap CI).
  - `DATA_SETUP.md`, `README.md`.
- **Next:** run `python src/download_data.py`, complete the THINGSplus image click-through, point `config.py` at the files, then `python src/data.py` to verify.

### Day 0 — Planning
- Compressed the original 1.5–2.5 month design into a 7-day sprint.
- Locked the 5-backbone zoo; deferred neural/hyperbolic/diffusion extensions.
- Created `PROJECT_PLAN.md` and this log.
