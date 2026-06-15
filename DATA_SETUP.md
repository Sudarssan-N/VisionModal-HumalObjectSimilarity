# Data Setup (Phase 0)

All data is publicly available. Two pieces: the **behavioral triplets** (scriptable)
and the **images** (one-time manual click-through due to licensing).

## 1. Behavioral odd-one-out triplets — scripted ✅ DONE

```bash
python src/download_data.py             # downloads osfstorage-archive.zip (412 MB) -> data/raw/
cd data/raw && unzip -o osfstorage-archive.zip -d extracted
cd extracted && unzip -o full_triplet_dataset.zip -d triplets
cd ../../.. && python src/prepare_data.py   # -> data/concepts.txt, triplets_{train,val,test}.npy
```

What the release contains (confirmed):
- `triplet_dataset/trainset.txt` — 4,120,663 triplets (90% of regular data)
- `triplet_dataset/validationset.txt` — 453,642 triplets (10%)
- `triplet_dataset/testset1.txt` — 15,640 noise-ceiling triplets (used as our test set)
- `variables/unique_id.txt` — the 1,854 concept names in THINGS order

> Convention (confirmed from `dataset_description.txt`): triplets are **0-based**
> and reordered as `[chosen_pair_a, chosen_pair_b, odd_one_out]`, i.e. the
> **odd-one-out is the last column** — exactly what the pipeline expects.

`prepare_data.py` validates the index range is `[0, 1853]` and writes fast `.npy`
files; `config.py` already points at them.

Sources:
- Figshare+: https://plus.figshare.com/articles/dataset/THINGS-data_Behavioral_odd-one-out_data_and_code/20552784
- OSF: https://osf.io/f5rn6/
- Code: https://github.com/ViCCo-Group/THINGS-data

## 2. Images — manual (one-time agreement)

We only need the **1,854 reference images** (one per concept), not the full 26k set.

**Preferred — THINGSplus CC0 images** (safe to reproduce in a paper):
- Reachable via https://things-initiative.org/ → THINGSplus.

**Alternative — original THINGS images** (research use only, requires agreeing to terms):
- https://things-initiative.org/ → THINGS object images database.

The download is organized as one folder per concept
(`object_images/aardvark/aardvark_01b.jpg`, ...). Use the helper to pick one
representative image per concept and name it correctly:

```bash
python src/organize_images.py /path/to/THINGS/object_images
# copies one image per concept -> data/images/{concept}.jpg
```

The pipeline needs exactly **one image per concept**, named `{concept}.jpg`, so
the image order matches the concept/triplet index order. `src/data.py::image_paths`
resolves `data/images/{concept}.jpg`, falling back to a sorted directory listing.

## 3. Verify

```bash
python src/data.py
# concepts: 1854 rows ...
# train: (4120663, 3) ... / val / test printed
```

Behavioral data already passes this. Once images are in `data/images/`,
Phase 0 is fully done → run `src/extract_features.py`.
