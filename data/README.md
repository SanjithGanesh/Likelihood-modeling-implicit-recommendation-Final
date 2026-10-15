# Data setup

This repository does not redistribute MovieLens data.

## MovieLens 1M

Download MovieLens 1M directly from GroupLens Research:

https://grouplens.org/datasets/movielens/1m/

Place the archive and extracted directory at:

```text
data/raw/ml-1m.zip
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Expected archive integrity:

```text
size:   5,917,549 bytes
MD5:    c4d9eecfca2ab87c1945afe126590906
SHA256: a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20
```

Then run:

```bash
python scripts/prepare_data.py --config config/publication.yaml --version v1
```

This generates the processed dataset and canonical user-wise train/validation/test split under `data/processed/` and `data/splits/`.

The expected publication split is:

```text
train:       805,443
validation:   97,383
test:         97,383
users:         6,040
items:         3,706
```

The authoritative processed-artifact and split hashes are stored in `config/final_evaluation_v1.yaml`.

## Development dataset

MovieLens latest-small was used only for development and pipeline/model-contract validation. It is not the primary publication benchmark. Its manifest is retained in `data/manifests/ml-latest-small.yaml` for provenance.
