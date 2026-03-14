# Full-Training Orchestration Contract

This document defines the canonical full-training contract for the routerset-backed student MoE pipeline.

## Canonical Stages

Every full run follows these stages in order:

1. `preflight`
2. `startup_gate`
3. `training`
4. `export`
5. `smoke`

The canonical one-shot operator entrypoint is `scripts/full_train_moe.py` or `make full-train`.

Stage intent:

- `preflight`: validate the manifest, canonical tensor shape contract, and checkpoint availability.
- `startup_gate`: load one training sample and verify Lightning can be imported before long-running fit work begins.
- `training`: write `config.json`, dataset/checkpoint reports, build the MoE, and call `trainer.fit(...)`.
- `export`: write the run bundle plus release artifacts.
- `smoke`: reload the exported bundle and run validation inference through `scripts/smoke_test_moe.py`.

## Run Directory Contract

The canonical run directory is `outputs/moe/<timestamp>/`.

`summary.json` is the source-of-truth run summary for training and export stages. It records:

- the canonical stage list
- which stages completed before the summary was written
- the expected artifact filenames for the run directory
- the next required stage when the full five-stage contract is not yet complete

Expected run-directory artifact names:

- `runtime_environment.json`
- `startup_log.txt`
- `startup_stage.json`
- `preflight_report.json`
- `startup_gate.json`
- `config.json`
- `dataset_report.json`
- `checkpoint_report.json`
- `baseline_summary.json`
- `metrics.json`
- `student_moe_bundle.pt`
- `routing_predictions.jsonl`
- `summary.json`

Expected release artifact names under `outputs/phidranet/phidranet_<release_name>/`:

- `student_moe_bundle.pt`
- `config.json`
- `metrics.json`
- `baseline_summary.json`
- `routing_predictions.jsonl`
- `dataset_report.json`
- `checkpoint_report.json`
- `release_manifest.json`
- `DEPLOY.md`

## Failure Contract

Failures must still write `outputs/moe/<timestamp>/summary.json` when the output directory has already been created.

Required failure details:

- `status`
- `failure_stage`
- `startup_stage`
- `manifest_path`
- `config_path` when `config.json` was already written
- `startup_log_path`
- `startup_stage_path`
- `error_type`
- `error`

Example startup-progress failure:

- if a failure happens after `trainer.fit(...)` becomes eligible to start, the recorded `failure_stage` is `fit_started`
- the failure summary still includes `config_path`, `startup_log_path`, and `startup_stage_path`

## Negative Contract

Malformed or missing manifests must fail before long-running work starts.

That means:

- no `fit_started` stage is recorded
- no trainer is created
- no bundle export is attempted
- the failure summary still records the resolved `manifest_path` and startup log paths

## Smoke Contract

The `smoke` stage is executed by `scripts/smoke_test_moe.py` or `make smoketest`.

For the full-training path, `scripts/full_train_moe.py` runs the smoke validation after export and rewrites `summary.json` so the canonical stage list is fully complete.

Smoke-specific outputs live alongside the training run under the same `outputs/moe/<timestamp>/` root:

- `smoke_test_summary.json`
- `inference/summary.json`
- `inference/routing_predictions.jsonl`

The smoke stage is the final gate for considering the full five-stage contract complete.
