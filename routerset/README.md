---
pretty_name: routerset
license: mit
size_categories:
- 10K<n<100K
task_categories:
- image-classification
- zero-shot-image-classification
language:
- en
tags:
- remote-sensing
- earth-observation
- multispectral
- multilabel
- satellite-imagery
viewer: false
---

# routerset

`routerset` is a materialized multi-label remote sensing dataset assembled from the current `phi2FM` downstream sources.

## What is kept in this repo

- `multilabel_dataset/manifest.jsonl`: canonical record manifest
- `multilabel_dataset/images/`: materialized `.npy` samples
- `multilabel_dataset/label_vocab.json`: label vocabulary
- `multilabel_dataset/summary.json`: dataset build summary
- `multilabel_dataset/materialization_summary.json`: materialization results
- `multilabel_dataset/plots/`: dataset analysis graphs

The repo is intentionally minimal. Redundant root-level metadata copies and training-ready derivative exports were removed.
When local compatibility copies still exist at the dataset root, `multilabel_dataset/` remains canonical and rebuilt manifests are synchronized from that source of truth.

## Current snapshot

- Records: `15,731`
- Materialized samples: `15,731` `.npy` files
- Included datasets: `fire`, `burned_area`, `anomaly_detection`, `worldfloods`, `lc`, `roads`
- Missing dataset: `building`
- Label cardinality: min `0`, max `4`, mean `1.2793`

### Per-dataset totals

- `fire`: `1,600`
- `burned_area`: `1,299`
- `anomaly_detection`: `9,216`
- `worldfloods`: `2,353`
- `lc`: `63`
- `roads`: `1,200`

### Status counts

- `positive`: `14,251`
- `explicit_negative`: `949`
- `below_threshold`: `531`

### Most frequent labels

- `water`: `7,570`
- `land`: `4,789`
- `cloud`: `3,123`
- `turbid_water`: `1,623`
- `road_present`: `1,031`
- `burned_area`: `688`

## Analysis plots

The dataset graphs are stored in `multilabel_dataset/plots/`:

- `coverage_distributions.png`
- `dataset_label_heatmap.png`
- `dataset_split_counts.png`
- `label_cooccurrence.png`
- `label_frequency.png`
- `status_and_cardinality.png`
- `plot_summary.json`

## Notes

- Arrays are stored as `.npy` materializations.
- The dataset was built from the current local `phi2FM` downstream sources and then uploaded to this Hugging Face dataset repository.
- `burned_area` was rebuilt from the original OEOBench `256x256` source scenes. The raw routerset artifact now stores one native `7x256x256` burned-area record per selected source scene instead of eight derived `64x128` subpatches.
- `roads` raw rows are rebuilt as `256x256x10` `uint16` 2x2 mosaics from the published `500_shot_roads` archive. They keep native `road_present` coverage and add heuristic `cloud` / `land` / `water` weak labels under `label_source = native+heuristic_weak`.
- `anomaly_detection` raw rows are rebuilt as deterministic `8x256x256` tiles from the original source zarr, with aligned edge tiles when needed, instead of `4096x4096` full-scene rows.
- The local rebuild helpers are `scripts/rebuild_routerset_burned_area_from_source.py` and `scripts/rebuild_routerset_roads_anomaly_from_source.py`. They back up the existing metadata snapshot, rewrite the raw `.npy` files and manifests from source, refresh summary/audit metadata, and can publish the corrected snapshot to Hugging Face.
- A corrected canonical `8x256x256` export for local training and audit can be generated with `make routerset-materialize`, which writes to `outputs/routerset/materialized_256/` by default. Raw `roads`/`lc` records are mapped to the student Sentinel-2 layout and scaled by `1/10000`; float-domain student experts (`fire`, `burned_area`, `worldfloods`, `anomaly_detection`) are channel-adapted and then min-max normalized per image before padding. The canonical raw routerset snapshot now stores native `256x256` burned-area scenes, `256x256` anomaly tiles, and `256x256` roads mosaics.
- Add `--selected-only` to `scripts/materialize_routerset_dataset.py` when exporting only a subset of experts and you need a self-contained manifest without passthrough rows from the other tasks.
- A clean variant can be generated with `make routerset-materialize-clean`, which writes `manifest_256.jsonl`, `fault_rows_256.jsonl`, and `fault_report.json` under `outputs/routerset/materialized_256_clean/` by default.
- A full tile-by-tile audit can be generated with `make routerset-audit MATERIALIZED_DATASET_DIR=outputs/routerset/fix27March`, which writes `audit/tile_audit.jsonl`, `audit/audit_summary.json`, and RGB / false-RGB plot previews under that dataset root.
- A full file-by-file audit over the rebuilt raw routerset snapshot can be generated with `make routerset-raw-audit ROUTERSET_DIR=routerset`, which writes `summary.json`, `file_audit.jsonl`, `sample_rows.json`, and summary plots under `routerset/multilabel_dataset/audit_raw/`.
- An executed raw-sample gallery notebook can be generated with `env PYTHONPATH=src .venv/bin/python scripts/generate_routerset_raw_audit_notebook.py --execute`, which writes [notebooks/routerset_raw_audit.ipynb](/shared/home/rdelprete/PythonProjects/hydranet-phisat2/notebooks/routerset_raw_audit.ipynb).
- The clean export only removes objective row-level artifact faults such as all-zero materialized tiles. Split-level problems, for example `fire` validation having no positive rows, stay reported as blockers instead of being rewritten silently.
- The dedicated notebook for the rebuilt `fix27March` artifact is `notebooks/routerset_fix27March_audit.ipynb`.
