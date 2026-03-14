## US-002 ExecPlan

### Goal
Add a dedicated one-shot full-training command path that runs the canonical training/export flow and the final smoke validation in one invocation.

### Scope
- `src/hydranet/moe_training.py`
- `scripts/full_train_moe.py`
- `Makefile`
- `tests/test_moe_training.py`
- `README.md`
- `full_training_contract.md`

### Non-goals
- No changes to model architecture, optimizer behavior, or release artifact layout
- No changes to the existing `train-prepare`, `smoketest-preflight`, or `smoketest` command semantics

### Invariants
- Existing training and smoke helpers remain reusable entrypoints
- Missing required CLI arguments fail in argparse before any output directory is created
- Training-stage failures keep the current `summary.json` failure contract
- Full-flow smoke failures still write `failure_stage`

### Steps
1. Add a dedicated `run_full_training(...)` orchestration helper that calls `train_switcher(...)`, runs exported-bundle smoke validation, and updates `summary.json` for full success or smoke-stage failure.
2. Add a single-invocation CLI script and Make target for the new full-training path.
3. Add regression tests for full-flow success, smoke-stage failure reporting, and CLI missing-argument handling.
4. Update the operator docs to expose the canonical one-shot command.
5. Run targeted and required repo validation, then commit and record progress.

### Validation
- `pytest tests/test_moe_training.py`
- `python scripts/full_train_moe.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the full-training helper, CLI, and target if they break downstream consumers.
- The change is file-based only; no schema migration or state cleanup is required.

## US-003 ExecPlan

### Goal
Strengthen preflight so dataset and checkpoint readiness failures are caught before training starts, with deterministic per-expert checkpoint reporting.

### Scope
- `src/hydranet/moe_training.py`
- `tests/test_moe_training.py`

### Non-goals
- No changes to model architecture, training hyperparameters, or smoke/export orchestration
- No changes to the routerset rebuild algorithm beyond validation/reporting needs

### Invariants
- Preflight remains the source of truth for readiness checks before training begins
- `dataset_report.json`, `checkpoint_report.json`, and `preflight_report.json` stay under the selected output directory
- Training must still abort before fit when required experts or checkpoints are missing
- Checkpoint reporting must be deterministic for a given expert/training/n-shot/runtime-root combination

### Steps
1. Tighten dataset validation so preflight explicitly checks split-level record counts and required expert presence.
2. Replace the checkpoint readiness helper with a per-expert report that records source path, deterministic local path, and success/failure status without losing failure details.
3. Validate the checkpoint report after writing it so negative cases still leave an actionable artifact on disk before aborting.
4. Add regression tests for successful reports and explicit missing-checkpoint failures.
5. Run targeted and required repo validation, then record the outcome in progress logs and commit.

### Validation
- `pytest tests/test_moe_training.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the preflight/checkpoint helper changes if they break checkpoint acquisition or report consumers.
- The change is file-based only; no migration or cleanup beyond removing generated run artifacts is required.

## US-004 ExecPlan

### Goal
Stabilize startup gate diagnostics so maintainer-facing failures clearly record timeout, traceback, and an explicit startup failure stage before fit begins.

### Scope
- `src/hydranet/moe_training.py`
- `tests/test_moe_training.py`

### Non-goals
- No changes to checkpoint readiness, training hyperparameters, or smoke/export orchestration
- No changes to CLI surface beyond the existing startup timeout behavior

### Invariants
- `startup_gate.json`, `startup_log.txt`, and `startup_stage.json` remain the source-of-truth artifacts for startup diagnostics
- Timeout or import failures must abort before `fit_started`
- Successful startup-gate behavior and skipped-gate behavior must remain compatible with existing prepare/train flows

### Steps
1. Expand the startup-gate report so it records a top-level status, timeout, and failure traceback when the probe fails.
2. Tighten training-stage startup failure handling so the recorder marks an explicit `startup_failed` stage and logs the gate transition timestamps before re-raising.
3. Add regression tests for probe failure details and the negative case where startup failure does not proceed to fit.
4. Run targeted and required repo validation, then record the outcome in logs and commit.

### Validation
- `pytest tests/test_moe_training.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the startup gate/reporting changes if downstream tooling depends on the previous gate JSON shape.
- The change is file-based only; removing generated run artifacts is sufficient recovery.

## US-005 ExecPlan

### Goal
Abort fit immediately when training or validation loss becomes non-finite, while writing batch-level diagnostics to the startup log and run summary before any export can succeed.

### Scope
- `src/hydranet/moe_lightning.py`
- `src/hydranet/moe_training.py`
- `tests/test_moe_training.py`
- `full_training_contract.md`

### Non-goals
- No changes to model architecture, optimizer settings, checkpoint discovery, or smoke-stage behavior
- No changes to the public CLI surface beyond the added failure diagnostics in existing artifacts

### Invariants
- Failures that occur after fit becomes eligible still report `failure_stage` from the fit boundary
- `summary.json` and `startup_log.txt` remain the source-of-truth artifacts for training failures
- Non-finite loss must prevent bundle export and must not mark the run as successful
- Existing startup-gate and preflight behaviors remain unchanged

### Steps
1. Raise a structured non-finite-loss error from the Lightning step with batch index, sample ids, expert context, and loss metadata.
2. Teach `train_switcher(...)` to detect that structured failure, append the diagnostic event to `startup_log.txt` without changing the fit-stage contract, and persist a config snapshot plus batch diagnostics into `summary.json`.
3. Add regression coverage for the NaN/Inf failure path and update the training contract doc for the new failure artifact fields.
4. Run targeted and required repo validation, then record progress, review the diff for security/perf/regression risk, and commit.

### Validation
- `pytest tests/test_moe_training.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the structured non-finite-loss diagnostics if they cause trainer compatibility issues.
- The change is file-based only; failed-run artifacts can be removed by deleting the affected output directory.

## US-006 ExecPlan

### Goal
Standardize the artifact layout so every run keeps runtime, checkpoint, config, report, inference, and bundle outputs under one predictable output root, with a documented cleanup policy for stale failed runs.

### Scope
- `src/hydranet/moe_training.py`
- `scripts/train_moe_switcher.py`
- `scripts/full_train_moe.py`
- `scripts/smoke_test_moe.py`
- `tests/test_moe_training.py`
- `README.md`
- `howtorun.md`
- `full_training_contract.md`

### Non-goals
- No changes to training hyperparameters, model architecture, or stage ordering
- No changes to routerset data preparation beyond where artifacts are written and documented

### Invariants
- Startup, dataset, checkpoint, summary, and smoke artifacts remain under the selected timestamped run directory
- Runtime caches and downloaded checkpoints remain under `runtime_root`
- Bundle/release artifacts must not be written outside `output_dir`
- Failed runs still write the same summary/failure signals needed for diagnosis

### Steps
1. Add a single artifact-layout definition in `moe_training.py` that describes run, runtime, checkpoints, bundle, and inference locations.
2. Route release creation through the run-local bundle root and reject release paths that escape the selected output root.
3. Add regression coverage for the new release location and the negative case where a bundle path is requested outside the allowed roots.
4. Update the training contract and operator docs with the final layout plus a cleanup policy for stale failed runs.
5. Run targeted and required validation, review the diff for security/performance/regression risk, then record progress and commit.

### Validation
- `pytest tests/test_moe_training.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the layout helper and release-root enforcement if any downstream tooling relies on the old external release location.
- Existing run directories remain file-based; rollback is deleting affected output folders and re-running with the previous code.
