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
