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

- Records: `15,512`
- Materialized samples: `15,512` `.npy` files
- Included datasets: `fire`, `burned_area`, `anomaly_detection`, `worldfloods`, `lc`, `roads`
- Missing dataset: `building`
- Label cardinality: min `0`, max `5`, mean `0.8083`

### Per-dataset totals

- `fire`: `1,600`
- `burned_area`: `6,660`
- `anomaly_detection`: `36`
- `worldfloods`: `2,353`
- `lc`: `63`
- `roads`: `4,800`

### Status counts

- `positive`: `10,514`
- `explicit_negative`: `2,777`
- `below_threshold`: `2,221`

### Most frequent labels

- `cloud`: `4,159`
- `road_present`: `2,702`
- `burned_area`: `2,087`
- `water`: `1,710`
- `land`: `1,448`

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
- A corrected canonical `8x256x256` export for local training and audit can be generated with `make routerset-materialize`, which writes to `outputs/routerset/materialized_256/` by default. Raw `roads`/`lc` records are mapped to the student Sentinel-2 layout and scaled by `1/10000`; float-domain student experts (`fire`, `burned_area`, `worldfloods`, `anomaly_detection`) are channel-adapted and then min-max normalized per image before padding. `burned_area` uses centered padding for undersized tiles so the native `64x128` content is not anchored in the top-left corner.
- Add `--selected-only` to `scripts/materialize_routerset_dataset.py` when exporting only a subset of experts and you need a self-contained manifest without passthrough rows from the other tasks.
- A clean variant can be generated with `make routerset-materialize-clean`, which writes `manifest_256.jsonl`, `fault_rows_256.jsonl`, and `fault_report.json` under `outputs/routerset/materialized_256_clean/` by default.
- A full tile-by-tile audit can be generated with `make routerset-audit MATERIALIZED_DATASET_DIR=outputs/routerset/fix27March`, which writes `audit/tile_audit.jsonl`, `audit/audit_summary.json`, and RGB / false-RGB plot previews under that dataset root.
- The clean export only removes objective row-level artifact faults such as all-zero materialized tiles. Split-level problems, for example `fire` validation having no positive rows, stay reported as blockers instead of being rewritten silently.
- The dedicated notebook for the rebuilt `fix27March` artifact is `notebooks/routerset_fix27March_audit.ipynb`.
