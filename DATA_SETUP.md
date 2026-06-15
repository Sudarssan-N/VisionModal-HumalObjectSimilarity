# Data Setup (Phase 0)

All data is publicly available. Two pieces: the **behavioral triplets** (scriptable)
and the **images** (one-time manual click-through due to licensing).

## 1. Behavioral odd-one-out triplets — scripted

```bash
python src/download_data.py --list      # see what's available
python src/download_data.py             # download into data/raw/
```

This pulls the "THINGS-data: Behavioral odd-one-out data and code" release from
the public Figshare+ API (no login). After it finishes:

- Unpack any archives under `data/raw/`.
- Locate the triplet file (an `(M, 3)` integer array; ~4.7M rows) and the
  concept list (1,854 rows). Point `config.py` at them:
  - `TRIPLETS_FILE` → the triplet `.npy`/`.txt`
  - `CONCEPTS_FILE` → `things_concepts.tsv` (or equivalent concept list)

> Convention check: triplets must be stored with the **odd-one-out in the last
> column**. If the source differs, fix the reordering once in
> `src/data.py::load_triplets`.

Sources:
- Figshare+: https://plus.figshare.com/articles/dataset/THINGS-data_Behavioral_odd-one-out_data_and_code/20552784
- OSF: https://osf.io/f5rn6/
- Code: https://github.com/ViCCo-Group/THINGS-data

## 2. Images — manual (one-time agreement)

We only need the **1,854 reference images** (one per concept), not the full 26k set.

**Preferred — THINGSplus CC0 images** (safe to reproduce in a paper):
- Reachable via https://things-initiative.org/ → THINGSplus.
- Download the license-free image set and place the 1,854 images in `data/images/`.

**Alternative — original THINGS images** (research use only, requires agreeing to terms):
- https://things-initiative.org/ → THINGS object images database.

After placing images:

```bash
python src/data.py        # smoke test: prints concept count, triplet shape, split sizes
```

`src/data.py::image_paths` resolves each concept to its file via a filename
column in the concept TSV, falling back to a sorted directory listing. Make sure
the image order matches the concept/triplet index order.

## 3. Verify

```bash
python src/data.py
# concepts: 1854 rows ...
# triplets: (4707..., 3) ...
#   train / val / test counts printed
```

Once this passes, Phase 0 is done → run `src/extract_features.py`.
