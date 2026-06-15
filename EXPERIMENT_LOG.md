# Experiment Log — Human–Model Visual Alignment (THINGS odd-one-out)

Living progress tracker. Append newest entries at the top of the **Daily log**. Keep the **Status board** and **Results** tables current. See `PROJECT_PLAN.md` for the phase definitions.

---

## Status board

| Phase | Description | Status | Day target | Notes |
|-------|-------------|--------|------------|-------|
| 0 | Setup & data acquisition | 🟡 In progress | Day 1 | Behavioral data done + verified; only THINGS images (manual DL) pending |
| 1 | Feature extraction (5 backbones) | ⬜ Not started | Day 1–2 | Script ready; blocked on images |
| 2 | Zero-shot odd-one-out baseline | ⬜ Not started | Day 2 | |
| 3 | Learn linear transforms | ⬜ Not started | Day 3–4 | |
| 4 | Cross-model transfer | ⬜ Not started | Day 4–5 | |
| 5 | Error analysis & RSA | ⬜ Not started | Day 5–6 | |
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
