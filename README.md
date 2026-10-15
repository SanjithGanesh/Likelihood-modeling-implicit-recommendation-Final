# Likelihood Modeling for Implicit Recommendation

Reproducibility code and compact publication artifacts for:

**Likelihood Modeling for Implicit Recommendation: A Unified Full-Catalog Evaluation of Bernoulli, Poisson, and Gaussian Objectives**

Author: **Sanjith Ganesh**

## Overview

This repository contains the implementation used to compare 13 recommendation approaches under one controlled MovieLens 1M full-catalog ranking protocol. The comparison spans Bernoulli, Poisson, Gaussian, hierarchical Bayesian, decomposition, and exponential-family embedding formulations.

The primary task treats every observed MovieLens rating as a positive interaction. Hyperparameters are selected on validation NDCG@10 only. Final models are refit from scratch on train + validation and evaluated on the held-out test set using seeds 2026, 2027, and 2028.

Primary ranking metrics are reported at K = 5, 10, and 20:

- Precision
- Recall
- Hit Rate
- NDCG
- MRR
- MAP

The repository also includes:

- a preference-aware sensitivity analysis using rating >= 4 as relevance;
- a validation-only Gaussian EF-EMB context-normalization stability analysis;
- compact authoritative result summaries;
- publication figure data and PDF figures;
- tests for model semantics, leakage controls, reproducibility, evaluation, and runners.

## Models

The 13 evaluated approaches are:

- Random
- Popularity
- PCA/SVD
- BernoulliMF
- PoissonMF
- GaussianMF
- LogisticPCA
- Hierarchical Poisson Factorization (HPF)
- Bernoulli EF-EMB (`b-emb`)
- Poisson EF-EMB (`p-emb`)
- Additive Poisson EF-EMB (`ap-emb`)
- Gaussian EF-EMB (`g-emb`)
- Nonnegative Gaussian EF-EMB (`ng-emb`)

Poor-performing models are intentionally retained; no method was removed based on final-test performance.

## Repository layout

```text
config/     Publication, tuning, final-evaluation, sensitivity, and figure configs
data/       Data placeholders and MovieLens manifests; raw data are not committed
docs/       Model specification, ground-truth protocol, and provenance notes
results/    Compact authoritative publication outputs and figures
scripts/    Data preparation, tuning, final evaluation, secondary analyses, aggregation, figures
src/        Shared model, sampling, split, evaluation, and reproducibility implementation
tests/      Scientific and pipeline tests
release/    Release manifest / checksums
```

## Environment

The completed study used:

- Python 3.11.9
- PyTorch 2.7.0 (CPU)

Install the pinned dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

The primary benchmark is **MovieLens 1M** from GroupLens Research.

The dataset is intentionally **not redistributed in this repository**. Download it from the official GroupLens MovieLens 1M page and place both the archive and extracted directory under `data/raw/`:

```text
data/raw/ml-1m.zip
data/raw/ml-1m/ratings.dat
data/raw/ml-1m/movies.dat
data/raw/ml-1m/users.dat
```

Expected archive integrity:

```text
MD5:    c4d9eecfca2ab87c1945afe126590906
SHA256: a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20
```

See `data/README.md` and `data/manifests/ml-1m.yaml` for details.

## Reproduce the canonical split

After placing MovieLens 1M under `data/raw/`:

```bash
python scripts/prepare_data.py --config config/publication.yaml --version v1
```

Expected canonical counts:

```text
train:       805,443
validation:   97,383
test:         97,383
final fit:   902,826
users:         6,040
items:         3,706
```

The final protocol records the canonical processed-data and split hashes in `config/final_evaluation_v1.yaml`.

## Hyperparameter tuning

Tuning is validation-only. The test split is not exposed by the tuning CLI.

Example dry run:

```bash
python scripts/tune_models.py --model all --dry-run
```

The selected publication hyperparameters are already recorded in:

```text
config/final_hyperparameters_v1.yaml
```

## Final evaluation

The final-run script is preflight-only unless the explicit final-test flag is supplied.

Preflight:

```bash
python scripts/run_final.py
```

The publication results included in this release correspond to the fixed protocol in:

```text
config/final_evaluation_v1.yaml
```

and the selected hyperparameters in:

```text
config/final_hyperparameters_v1.yaml
```

## Secondary analyses

Preference-aware sensitivity preflight:

```bash
python scripts/run_preference_sensitivity.py
```

Gaussian EF-EMB stability preflight:

```bash
python scripts/run_stability_ablation.py
```

Both analyses are prespecified and do not replace the primary full-catalog test evaluation.

## Authoritative publication artifacts

Compact publication outputs are included under `results/`:

- `results/final-ml1m-v1-1874aeba/`
- `results/step60-aggregation-v1.1/`
- `results/step61-figures-v1.3/`
- `results/publication-results-freeze-v2/`

The large internal development/audit logs, historical result versions, raw MovieLens files, processed data, and canonical split CSVs are intentionally excluded from the public code release.

Key frozen hashes:

```text
Final evaluation protocol:
1874aeba214b3a049b439fa4d6b8aabbb8c20e6aa98fdfbeb94c0bb6348002f8

Final hyperparameters:
9aae84dd59d7c34c82a67563139bdb27256874ba79c645c749cbcf788d72c981

Processed ML-1M artifact:
97747ad05c8843a565a9a08d095f5283093471f01c41bddd5be16d49c6c2e3ec

Canonical split:
4387af2c66afbc14bb94526260a8f6cde9cf4601f70d47c4f6301022b01709b8
```

## Main primary result

Mean NDCG@10 across final seeds 2026, 2027, and 2028:

| Rank | Model | NDCG@10 |
|---:|---|---:|
| 1 | LogisticPCA | 0.5888 ± 0.0020 |
| 2 | BernoulliMF | 0.3026 ± 0.0007 |
| 3 | p-emb | 0.2908 ± 0.0006 |
| 4 | PoissonMF | 0.2834 ± 0.0016 |
| 5 | b-emb | 0.2831 ± 0.0004 |

Complete metrics for all 13 models are in `results/step60-aggregation-v1.1/primary_summary.csv`.

## Tests

The repository includes scientific/model/pipeline tests under `tests/`. After installing dependencies, run:

```bash
pytest -q
```

Some integration tests require the MovieLens artifacts produced by `scripts/prepare_data.py`.

## Data and code provenance

- Model semantics: `docs/model_spec.md`
- Publication ground truth: `docs/publication_ground_truth_v1.yaml`
- Dataset provenance note: `docs/dataset_manifest_provenance_note_v1.md`
- Final evaluation protocol: `config/final_evaluation_v1.yaml`
- Selected hyperparameters: `config/final_hyperparameters_v1.yaml`

## Citation

A `CITATION.cff` file is included. The repository DOI will be added after the versioned GitHub release is archived with Zenodo.

## License

The code release is prepared for the MIT License. MovieLens data are subject to the GroupLens terms and are not included in this repository.
