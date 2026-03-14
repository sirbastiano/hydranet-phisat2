# How To Run

This repo should be run with a local micromamba environment, not the NFS-backed `.venv`. The heavy Python stack imports cleanly from a local prefix on `/tmp`; the old `.venv` stalls on `/shared/home`.

## 1. Create The Local Runtime

```bash
export MICROMAMBA=/shared/home/rdelprete/bin/micromamba
export MAMBA_ROOT_PREFIX=/tmp/micromamba-hydranet-root
export MICROMAMBA_PREFIX=/tmp/hydranet-phisat2-mamba

$MICROMAMBA create -y -p "$MICROMAMBA_PREFIX" -c pytorch -c conda-forge \
  python=3.9 pip pdm-backend pyyaml matplotlib numpy pandas scipy tqdm lmdb tabulate \
  pytorch=2.4.0 torchvision=0.19.0 torchaudio=2.4.0 cpuonly \
  timm pytorch-lightning huggingface_hub torchinfo
```

This repo uses `PYTHONPATH=src`, so an editable install is not required for the standard `make` targets below.

Optional sanity check:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python - <<'PY'
import torch
import hydranet.moe_training
print(torch.__version__)
print("hydranet.moe_training imported")
PY
```

## 2. Clean Caches

Project and local runtime cleanup:

```bash
export MAMBA_ROOT_PREFIX=/tmp/micromamba-hydranet-root
make clean-cache
```

What this clears:

- repo bytecode and pytest caches
- micromamba package caches
- common temp/build caches under `/tmp` and `/var/tmp`

Home-directory cache deletion under `/shared/home` can still be slow because that filesystem is NFS-backed.

## 3. Download Routerset

```bash
make routerset-download MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX"
```

Canonical dataset paths after download:

- manifest: `routerset/multilabel_dataset/manifest.jsonl`
- rebuilt MoE manifest: `routerset/multilabel_dataset/manifest_moe_train.jsonl`
- images: `routerset/multilabel_dataset/images/`

## 4. Prepare Training

This is the canonical readiness step. It configures a local runtime root, rebuilds the MoE split manifest, resolves checkpoints, and runs the startup gate.

```bash
make train-prepare MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX"
```

Useful overrides:

```bash
make train-prepare \
  MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX" \
  RELEASE_NAME=prep_local \
  TRAIN_OUTPUT_DIR=outputs/moe/prepare_local \
  RUNTIME_ROOT=/tmp/hydranet-runtime \
  ACCELERATOR=cpu \
  DEVICES=1
```

Expected artifacts:

- `outputs/moe/prepare_<utc_timestamp>/runtime_environment.json`
- `outputs/moe/prepare_<utc_timestamp>/preflight_report.json`
- `outputs/moe/prepare_<utc_timestamp>/startup_gate.json`
- `outputs/moe/prepare_<utc_timestamp>/prepare_report.json`

## 5. Preflight Only

Use this if you want to validate the rebuilt manifest and dataset without entering training.

```bash
make smoketest-preflight MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX"
```

## 6. Smoke Test

This is the fastest end-to-end MoE run. It rebuilds splits, runs preflight, trains the switcher for one epoch, saves the bundle, reloads it, and runs validation inference.

```bash
make smoketest MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX"
```

Expected outputs:

- smoke artifacts: `outputs/moe/smoke_<utc_timestamp>/`
- rebuilt manifest: `routerset/multilabel_dataset/manifest_moe_train.jsonl`
- release bundle: `outputs/moe/smoke_<utc_timestamp>/bundle/phidranet_smoke_v1/`

## 7. Full Training

This is the canonical one-shot training/export/smoke path. It runs preflight, startup gate, fit/export, then post-export smoke verification under one output root.

Expected environment variables:

- `MICROMAMBA=/shared/home/rdelprete/bin/micromamba`
- `MAMBA_ROOT_PREFIX=/tmp/micromamba-hydranet-root`
- `MICROMAMBA_PREFIX=/tmp/hydranet-phisat2-mamba`
- `PYTHONPATH=src`

Canonical full-training preset:

```bash
make full-train \
  MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX" \
  RELEASE_NAME=v1 \
  TRAIN_OUTPUT_DIR=outputs/moe/train_v1 \
  RUNTIME_ROOT=outputs/moe/train_v1/runtime \
  ACCELERATOR=cpu \
  DEVICES=1
```

Direct script equivalent:

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

Documented prepare -> train -> smoke verification flow:

```bash
make train-prepare \
  MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX" \
  RELEASE_NAME=v1 \
  TRAIN_OUTPUT_DIR=outputs/moe/train_v1 \
  RUNTIME_ROOT=outputs/moe/train_v1/runtime \
  ACCELERATOR=cpu \
  DEVICES=1

make full-train \
  MICROMAMBA_PREFIX="$MICROMAMBA_PREFIX" \
  RELEASE_NAME=v1 \
  TRAIN_OUTPUT_DIR=outputs/moe/train_v1 \
  RUNTIME_ROOT=outputs/moe/train_v1/runtime \
  ACCELERATOR=cpu \
  DEVICES=1

python - <<'PY'
from pathlib import Path
import json

run_dir = Path("outputs/moe/train_v1")
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
smoke = json.loads((run_dir / "smoke_test_summary.json").read_text(encoding="utf-8"))
assert summary["status"] == "completed"
assert summary["failure_stage"] is None
assert smoke["status"] == "completed"
print(run_dir / "bundle" / "phidranet_v1")
PY
```

## 8. Output Path Map

For a canonical run rooted at `outputs/moe/train_v1/`:

- run root: `outputs/moe/train_v1/`
- runtime root: `outputs/moe/train_v1/runtime/`
- runtime checkpoint cache: `outputs/moe/train_v1/runtime/weights/`
- startup diagnostics: `outputs/moe/train_v1/startup_log.txt`, `outputs/moe/train_v1/startup_stage.json`, `outputs/moe/train_v1/startup_gate.json`
- preflight artifacts: `outputs/moe/train_v1/preflight_report.json`, `outputs/moe/train_v1/dataset_report.json`, `outputs/moe/train_v1/checkpoint_report.json`
- training artifacts: `outputs/moe/train_v1/config.json`, `outputs/moe/train_v1/metrics.json`, `outputs/moe/train_v1/baseline_summary.json`, `outputs/moe/train_v1/routing_predictions.jsonl`, `outputs/moe/train_v1/summary.json`
- release bundle: `outputs/moe/train_v1/bundle/phidranet_v1/`
- smoke artifacts: `outputs/moe/train_v1/smoke_test_summary.json`, `outputs/moe/train_v1/inference/summary.json`, `outputs/moe/train_v1/inference/routing_predictions.jsonl`

Retention policy for stale failed runs:

- remove old failed runs by deleting the whole `outputs/moe/<timestamp>/` directory
- keep one recent failed run per failure mode until the underlying issue is fixed
- if `RUNTIME_ROOT` points at shared storage, treat it as a separate cache/checkpoint area and avoid deleting it during ordinary failed-run cleanup

## 9. Troubleshooting

- `SIGTERM` during `make full-train` or `make smoketest`
  Recheck `summary.json`, `startup_stage.json`, and `startup_log.txt` in the run directory to see the last recorded stage. Treat the run as failed and rerun into a fresh output directory instead of trying to continue from a partial bundle.
- Startup timeout before fit
  If `startup_gate.json` shows `status: failed` and `error_type: TimeoutError`, keep the runtime on a local `/tmp` micromamba prefix and rerun the direct script with a higher `--startup-timeout-seconds` value. The run should fail before `fit_started`; if it does not, the environment has drifted from the documented contract.
- NaN or Inf loss during fit
  If `summary.json` reports `error_type: NonFiniteBatchError`, inspect the `non_finite_loss` section in `summary.json` and the `non_finite_loss_detected` event in `startup_log.txt`. Use that batch/sample context to adjust the training preset or dataset inputs, then rerun from `make train-prepare`; the failed run must not be reused as a release candidate because export is intentionally blocked.

## 10. Direct Commands

Dataset download:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python scripts/download_routerset.py --local-dir routerset
```

Prepare only:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python scripts/train_moe_switcher.py \
  --prepare-only \
  --routerset-dir routerset \
  --rebuild-splits \
  --rebuilt-manifest-out routerset/multilabel_dataset/manifest_moe_train.jsonl \
  --runtime-root /tmp/hydranet-runtime \
  --output-dir outputs/moe/prepare_manual \
  --release-name prep_v1
```

Smoke test:

```bash
PYTHONPATH=src $MICROMAMBA run -p "$MICROMAMBA_PREFIX" python scripts/smoke_test_moe.py \
  --routerset-dir routerset \
  --release-name smoke_v1 \
  --runtime-root /tmp/hydranet-runtime
```

## 11. Known Limitation

Micromamba fixes the Python-package startup stall. It does not remove NFS latency from `routerset/` or other repo data still read from `/shared/home`. If prepare or training blocks after Python startup, the next fix is to move the active dataset working set to local storage as well.
