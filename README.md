# HydraNet

<p align="center">
  <img src="logo.svg" alt="HydraNet logo" width="280" height="280" />
</p>

<p align="center">
  <strong>Modular multi-task satellite modeling with shared encoders, task experts, and routerset-ready MoE workflows.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python version" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white"/></a>
  <a href="https://pytorch.org/"><img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch"/></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue"/></a>
  <a href="docs/index.html"><img alt="Docs" src="https://img.shields.io/badge/Docs-View-1f6feb?logo=gitbook"/></a>
</p>

<p align="center">
  <a href="#installation"><img alt="Get Started" src="https://img.shields.io/badge/🚀%20Get%20Started-2ea44f?style=for-the-badge&logo=rocket"/></a>
  <a href="#student-moe-and-phidranet"><img alt="MoE Guide" src="https://img.shields.io/badge/🧠%20MoE%20Guide-0A66C2?style=for-the-badge"/></a>
  <a href="docs/hydranet_moe_model.html"><img alt="Architecture" src="https://img.shields.io/badge/Diagram%20Guide-6f42c1?style=for-the-badge"/></a>
</p>

## Installation

```
uv sync
```

Or install in editable mode:

```
pip install -e .
```

If you prefer not to install, run with `PYTHONPATH=src`.

## Loading Models

### Student (default: Myriad optimized) - Namely the `HydraNet`, which is the default when calling `load_student()` without arguments.

```python
from hydranet import load_student, available_student_presets

model = load_student()  # uses "myriad_optimized" by default
print(available_student_presets())

custom = load_student(
    preset="myriad_optimized",
    n_classes=4,  # overrides allowed
)
```

### Teacher (downstream) - Namely the `PhiSatNet`, which is the default when calling `load_teacher()` without arguments.

```python
from hydranet import load_teacher

teacher = load_teacher(
    task="segmentation",
    pretrained_path=None,
    input_dim=8,
    output_dim=3,
    depths=[2, 2, 6, 2],
    dims=[64, 128, 256, 512],
    img_size=224,
)
```

## Quick Start with Pre-trained Weights

Load pre-trained models from HuggingFace and split them into components (see [notebooks/start_here.ipynb](notebooks/start_here.ipynb)):

```python
import hydranet

# Load model with automatic weight download from HuggingFace
model = hydranet.load_student(
    preset='checkpoint',      # Matches HF checkpoint architecture
    task='burned_area',
    n_shots=5000,
    training='finetuning',
    auto_load_weights=True,
    checkpoint_selection='best',  # Select by HF artifacts metrics instead of newest-only
    weights_dir='../weights',
    strict=False             # Allows task-specific head mismatches
)

# Split the model into ENCODER, BOTTLENECK, and DECODER components
components = hydranet.split_model(model)

# Access individual components
encoder = components.encoder
bottleneck = components.bottleneck
decoder = components.decoder

print(components)  # Display component summary
```

**Available:** 139 pre-trained model configurations from `sirbastiano/hydranet-phisat2` on HuggingFace.

## Student MoE and `phidranet`

This repository also includes a routerset-backed student Mixture-of-Experts path that assembles:

- one shared student encoder
- one learned routing switcher
- task-specific student decoder experts

For a simple model overview with diagrams, see [docs/hydranet_moe_model.html](docs/hydranet_moe_model.html).

The default expert set is aligned to routerset:

- `anomaly_detection`
- `burned_area`
- `fire`
- `lc`
- `worldfloods`

`roads` is currently excluded from the default expert set because the available downstream checkpoints are under audit. If needed, include it explicitly with `--experts`.

Recommended workflow:

```bash
export MICROMAMBA=/shared/home/rdelprete/bin/micromamba
export MAMBA_ROOT_PREFIX=/tmp/micromamba-hydranet-root
export MICROMAMBA_PREFIX=/tmp/hydranet-phisat2-mamba

make routerset-download MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
make routerset-materialize MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
make routerset-materialize-clean MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
make routerset-audit MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX MATERIALIZED_DATASET_DIR=outputs/routerset/fix27March
make routerset-raw-audit MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX ROUTERSET_DIR=routerset
make clean-cache MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
make train-prepare MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
make smoketest-preflight MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX
```

The canonical dataset root is `routerset/multilabel_dataset/`. The routerset training contract is fixed to `8x256x256`. Large `anomaly_detection` inputs are expanded into deterministic `256x256` tiles. Non-tiled tensors larger than `256` are cut out deterministically, and non-tiled tensors smaller than `256` are zero-padded after channel normalization. `burned_area` raw records were rebuilt from the original OEOBench source and now live in the routerset snapshot as native `7x256x256` scenes instead of downstream `64x128` subpatches. Raw `roads` and `lc` tiles follow the original Phi2FM student preprocessing contract before padding: Sentinel-2 bands are mapped into the 8-channel student layout and scaled by `1/10000`. The float-domain student experts `fire`, `worldfloods`, `burned_area`, and `anomaly_detection` now follow the local Phi2FM downstream contract before padding: channel adaptation first, then per-image channel-wise min-max normalization. Legacy undersized `burned_area` artifacts still use centered padding when encountered, but the canonical routerset raw snapshot no longer depends on that workaround. The preflight `dataset_report.json` now exposes per-expert normalization modes, compatibility-path source-record counts, and sampled true raw shapes and value-range stats for manifest-shaped rows so swapped-coordinate tile fallbacks and manifest-shape assumptions are visible.

To rebuild the raw burned-area slice from source and refresh the local routerset snapshot, use:

```bash
env PYTHONPATH=src python3 scripts/rebuild_routerset_burned_area_from_source.py --routerset-dir routerset
```

For a concrete corrected dataset export, use:

```bash
make routerset-materialize \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  MATERIALIZED_DATASET_DIR=outputs/routerset/materialized_256
```

This writes a canonical dataset artifact under `outputs/routerset/materialized_256/` with:

- `manifest_256.jsonl`
- `images/<expert>/<split>/*.npy`
- `materialization_summary.json`
- `dataset_report.json`
- `label_vocab.json` when present upstream

For a clean export that drops objective row-level artifact faults such as all-zero materialized tiles and writes an explicit fault audit, use:

```bash
make routerset-materialize-clean \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  MATERIALIZED_CLEAN_DATASET_DIR=outputs/routerset/materialized_256_clean
```

The clean export writes `fault_rows_256.jsonl` and `fault_report.json` next to the manifest. It does not invent labels or silently repair split semantics, so unresolved blockers such as `fire` validation having zero positive rows remain reported in `fault_report.json`.

When exporting only a subset of experts, add `--selected-only` to emit a self-contained manifest that drops unselected passthrough rows.

For a full file-by-file audit over a materialized export, use:

```bash
make routerset-audit \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  MATERIALIZED_DATASET_DIR=outputs/routerset/fix27March
```

This writes `audit/tile_audit.jsonl`, `audit/audit_summary.json`, and plot artifacts under the selected dataset root.

For a full file-by-file audit over the rebuilt raw routerset snapshot itself, use:

```bash
make routerset-raw-audit \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  ROUTERSET_DIR=routerset
```

This writes `summary.json`, `file_audit.jsonl`, `sample_rows.json`, and plot artifacts under `routerset/multilabel_dataset/audit_raw/`.

To generate an executed sample-gallery notebook for the rebuilt raw snapshot, use:

```bash
env PYTHONPATH=src .venv/bin/python scripts/generate_routerset_raw_audit_notebook.py --execute
```

Inspection notebooks:

- [notebooks/routerset_materialized_inspection.ipynb](notebooks/routerset_materialized_inspection.ipynb)
- [notebooks/routerset_fix27March_audit.ipynb](notebooks/routerset_fix27March_audit.ipynb)
- [notebooks/routerset_raw_audit.ipynb](notebooks/routerset_raw_audit.ipynb)

The `--prepare-only` path is now the canonical training-readiness step. It configures a local runtime/cache root, writes `runtime_environment.json`, prefetches the six expert checkpoints into a local weights directory, runs the dataset/checkpoint preflight, and executes a startup gate that loads one routerset sample and probes Lightning import before any fit starts.
The checked-in CLI defaults now use a `120` second startup-gate timeout because cold Lightning imports in this environment can exceed one minute.

Canonical environment presets for the `make` targets:

```bash
export MICROMAMBA=/shared/home/rdelprete/bin/micromamba
export MAMBA_ROOT_PREFIX=/tmp/micromamba-hydranet-root
export MICROMAMBA_PREFIX=/tmp/hydranet-phisat2-mamba
export PYTHONPATH=src
```

For the canonical one-shot full-training path, use:

```bash
make full-train MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX RELEASE_NAME=v1 TRAIN_OUTPUT_DIR=outputs/moe/train_v1
```

This runs preflight, startup gate, full training/export, and post-export smoke validation in one invocation. A successful run updates `summary.json` to mark the `smoke` stage complete and keeps the bundle, reports, and smoke/inference artifacts under the same timestamped output directory.

Canonical prepare -> full-train -> smoke verification flow:

```bash
make train-prepare \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  RELEASE_NAME=v1 \
  TRAIN_OUTPUT_DIR=outputs/moe/train_v1 \
  RUNTIME_ROOT=outputs/moe/train_v1/runtime

make full-train \
  MICROMAMBA_PREFIX=$MICROMAMBA_PREFIX \
  RELEASE_NAME=v1 \
  TRAIN_OUTPUT_DIR=outputs/moe/train_v1 \
  RUNTIME_ROOT=outputs/moe/train_v1/runtime

python - <<'PY'
from pathlib import Path
import json

summary = json.loads(Path("outputs/moe/train_v1/summary.json").read_text(encoding="utf-8"))
smoke = json.loads(Path("outputs/moe/train_v1/smoke_test_summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "completed"
assert summary["failure_stage"] is None
assert smoke["status"] == "completed"
print("full training + smoke verification passed")
PY
```

Direct script equivalent for the same full-training flow:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python scripts/full_train_moe.py \
  --routerset-dir routerset \
  --manifest-path routerset/multilabel_dataset/manifest_moe_train.jsonl \
  --runtime-root outputs/moe/train_v1/runtime \
  --output-dir outputs/moe/train_v1 \
  --release-name v1 \
  --training finetuning \
  --n-shots 5000 \
  --batch-size 4 \
  --max-epochs 20 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --target-size 256 \
  --target-channels 8 \
  --threshold 0.5 \
  --accelerator cpu \
  --devices 1
```

For a fast functional smoke test of the full MoE path, use:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python scripts/smoke_test_moe.py \
  --routerset-dir routerset \
  --release-name smoke_v1 \
  --runtime-root /tmp/hydranet-runtime
```

This smoke run rebuilds the routerset split if needed, runs preflight, trains for one epoch, reloads the exported bundle, and writes an inference report under the smoke output directory.

If Lightning startup fails in the local environment, training now writes `startup_log.txt`, `startup_stage.json`, and a failed `summary.json` before exiting.

This creates:

- run artifacts, config, and reports under `outputs/moe/<timestamp>/`
- runtime caches and downloaded checkpoints under `outputs/moe/<timestamp>/runtime/` unless `--runtime-root` overrides it
- the final release bundle under `outputs/moe/<timestamp>/bundle/phidranet_v1/`
- smoke verification outputs under `outputs/moe/<timestamp>/smoke_test_summary.json` and `outputs/moe/<timestamp>/inference/`

Troubleshooting:

- `SIGTERM`: if `make smoketest` or `make full-train` is terminated by the shell or job runner, treat the run as incomplete. Inspect `outputs/moe/<timestamp>/summary.json`, `startup_stage.json`, and `startup_log.txt` to confirm the last completed stage before rerunning the entire command into a fresh `TRAIN_OUTPUT_DIR`.
- Startup timeout: if `startup_gate.json` reports `status=failed` with `error_type=TimeoutError`, increase `--startup-timeout-seconds` on the direct script path or rerun after confirming the local micromamba prefix is on `/tmp` rather than NFS-backed storage.
- NaN-loss recovery: if `summary.json` reports `error_type=NonFiniteBatchError`, the run intentionally stops before export. Review `summary.json` and `startup_log.txt` for `non_finite_loss` details, then rerun from `make train-prepare` after adjusting the training preset or dataset state; do not reuse the failed run directory as a release candidate.

The final bundle can be reloaded with:

```python
from hydranet import load_student_moe_bundle

model = load_student_moe_bundle("outputs/moe/<timestamp>/bundle/phidranet_v1/student_moe_bundle.pt")
```

More detail is in `moe_works_report.md`.

Cleanup policy for stale failed runs:

- delete the full failed run directory under `outputs/moe/<timestamp>/`
- keep the newest failed run for each failure mode until the issue is resolved
- do not manually prune individual files inside a failed run; keep `summary.json`, startup diagnostics, and smoke outputs together

## Finding the Myriad-Optimized Model

The `myriad_optimized` preset was selected through systematic architecture search (see [notebooks/FindHydraNet.ipynb](notebooks/FindHydraNet.ipynb)):

1. **Search Space**: Evaluated 100+ configurations varying:
   - Base filters: 8, 16, 32
   - Channel multipliers: [1,2,3], [1,2,3,4], etc.
   - Depth: 2, 3, 4 blocks
   - Patch sizes: 28×28 to 224×224

2. **Optimization Criteria**:
   - Total inference time across 4096×4096 tile
   - Parameter count vs accuracy trade-off
   - Hardware constraints (Myriad X VPU)

3. **Result**: 16 base filters, [1,2,3,4] multipliers, depth=3
   - ~60K parameters
   - Optimal latency/accuracy balance
   - Fits Myriad X memory and compute budget

The analysis used 3D visualization (parameters × depth × latency) to identify the Pareto-optimal configuration.

## Repo Layout

- `src/hydranet/`: package entrypoint and public loading helpers
- `src/hydranet/models/`: student + teacher model implementations
- `scripts/export_onnx.py`: ONNX export script (writes to `outputs/onnx/`)
- `examples/`: quick usage snippets
- `notebooks/`: interactive tutorials and model selection analysis
