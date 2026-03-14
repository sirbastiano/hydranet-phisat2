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
