# Paper 2 — Dataset Manifest Provenance Note v1

## Purpose

This note explains why an early MovieLens 1M source manifest still contains:

- `preprocessing.status: not_started`
- `split.status: not_created`

even though preprocessing and canonical splitting are complete.

## Explanation

`data/manifests/ml-1m.yaml` is an early **source-state provenance snapshot**.

It records the verified raw MovieLens 1M source, raw statistics, intended publication role, and the state of the pipeline at the time that source manifest was created.

It is not intended to function as a mutable current-project-status file.

Later stages generated separate versioned manifests for the completed artifacts.

## Completed processed-data authority

The completed processed dataset is represented by:

`data/processed/ml-1m/v1/preparation_manifest.yaml`

That manifest records:

- raw rows: 1,000,209
- users: 6,040
- items: 3,706
- processed rows after filtering: 1,000,209
- reindexed user/item mappings
- SHA-256 values for the processed ratings and mapping files

The canonical processed ratings artifact is:

`data/processed/ml-1m/v1/ratings.csv`

## Completed split authority

The completed canonical split is represented by:

`data/splits/ml-1m/v1/split_manifest.yaml`

It records:

- strategy: per-user random
- split seed: 2026
- train rows: 805,443
- validation rows: 97,383
- test rows: 97,383
- users: 6,040
- items: 3,706
- training items: 3,706
- cold validation items: 0
- cold test items: 0

The canonical split SHA-256 is:

`4387af2c66afbc14bb94526260a8f6cde9cf4601f70d47c4f6301022b01709b8`

## Immutability rule

Do not edit the early source manifest merely to make its historical status fields look current.

Doing so would erase provenance and can break hash-linked records.

The correct interpretation is:

- `data/manifests/ml-1m.yaml` = historical source-state snapshot
- `data/processed/ml-1m/v1/preparation_manifest.yaml` = completed processed-data authority
- `data/splits/ml-1m/v1/split_manifest.yaml` = completed split authority
- `docs/current_project_status_v1.md` = current human-readable project status

Future current-state updates should create or revise current-facing documentation rather than silently rewriting hashed historical source manifests.
