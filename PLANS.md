## US-001 ExecPlan

### Goal
Define the full-training orchestration contract in one written source and make the run summary record the canonical stages and expected artifact names.

### Scope
- `src/hydranet/moe_training.py`
- `tests/test_moe_training.py`
- `full_training_contract.md`
- `README.md`
- `howtorun.md`

### Non-goals
- No new one-shot training command
- No changes to training hyperparameters, model behavior, or release layout

### Invariants
- Existing `make train-prepare`, `make smoketest-preflight`, and `make smoketest` flows remain the same
- Startup and failure summaries keep existing path fields
- Missing or malformed manifests must fail before `fit_started`

### Steps
1. Add a source-of-truth contract payload for canonical stages and artifact names to training summaries.
2. Validate the success summary against the required artifacts that should exist for a completed run.
3. Add regression tests for summary contract, `fit_started` failure recording, and fail-fast manifest handling.
4. Publish the contract in a dedicated markdown file and link it from the existing run docs.
5. Run targeted and required repo validation, then commit and record progress.

### Validation
- `pytest tests/test_moe_training.py`
- `make train-prepare`
- `make smoketest`
- `make smoketest-preflight`
- `pytest`
- `ruff check .`

### Rollback
- Revert the summary contract helper and tests if they break downstream consumers.
- The change is file-based only; no schema migration or state cleanup is required.
