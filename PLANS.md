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
