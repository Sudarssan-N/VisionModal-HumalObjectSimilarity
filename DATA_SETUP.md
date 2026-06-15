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

## 2. Images — scripted ✅ (CC0, no login)

We only need the **1,854 reference images** (one per concept), not the full 26k set.
The **THINGSplus CC0 set** (one copyright-free image per concept) is on the
THINGSplus OSF (`osf.io/jum2f`) and is **directly downloadable — no agreement,
no password** (the password only applies to the copyrighted full image set, which
we do not use). It is also safe to reproduce in a paper.

```bash
python src/download_data.py --images        # downloads + unzips images_THINGSplus-CC0.zip (1.1 GB)
python src/organize_images.py data/raw/images_cc0   # one image per concept -> data/images/{concept}.jpg
python src/data.py                          # verify
```

`organize_images.py` recursively scans the unzipped folder and matches each
concept to its image (handles flat `{concept}.jpg`, per-concept folders, or
`{concept}_NNx.jpg` naming). The pipeline needs exactly **one image per concept**,
named `{concept}.jpg`, so the order matches the concept/triplet index.

> Alternative (only if you specifically need the original photographs): the full
> `images_THINGS.zip` (4.8 GB) on the same OSF requires agreeing to terms and a
> password — not needed for this project.

## 3. Verify

```bash
python src/data.py
# concepts: 1854 rows ...
# train: (4120663, 3) ... / val / test printed
```

Behavioral data already passes this. Once images are in `data/images/`,
Phase 0 is fully done → run `src/extract_features.py`.
