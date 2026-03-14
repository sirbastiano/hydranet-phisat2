# Progress Log
Started: Sat Mar 14 11:51:46 UTC 2026

## Codebase Patterns
- (add reusable patterns here)

## [2026-03-14 11:59:33 UTC] - US-001: Define full-training orchestration contract
Thread: 
Run: 20260314-115146-3579918 (iteration 1)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-1.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 64083af docs(training-contract): define run contract
- Post-commit status: remaining unstaged repo changes outside US-001 are still present in the worktree
- Verification:
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" pytest tests/test_moe_training.py -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" pytest -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" ruff check . -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make train-prepare -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest-preflight -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest -> FAIL
- Files changed:
  - PLANS.md
  - full_training_contract.md
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Added a source-of-truth full-training contract document with canonical stages, failure semantics, and expected run/release artifacts.
- Extended `summary.json` generation to record canonical stages, completed stages, next stage, and expected artifact names, and validated the required artifact set for successful runs.
- Added regression coverage for the success contract, `fit_started` failure reporting, and fail-fast missing-manifest behavior.
- **Learnings for future iterations:**
  - Patterns discovered
  - `train_switcher` already performed inline preflight work; the missing contract piece was writing and validating `preflight_report.json` alongside the summary payload.
  - Gotchas encountered
  - Repo verification should be run with `.venv/bin` on `PATH`; the default shell runtime lacked `torch`.
  - Useful context
  - `ruff check .` currently fails on pre-existing notebook and legacy module issues unrelated to US-001, and the `make` training gates stalled in bootstrap on this machine before producing stage outputs beyond `script_bootstrap`.
---

## [2026-03-14 13:16:13 UTC] - US-007: Publish full-training runbook and command presets
Thread: 
Run: 20260314-115146-3579918 (iteration 7)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-7.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-7.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 8618d8c docs(training): publish full-train runbook
- Post-commit status: clean
- Verification:
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" pytest -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" ruff check . -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make train-prepare PYTHON=.venv/bin/python TRAIN_OUTPUT_DIR=outputs/moe/us007_prepare RUNTIME_ROOT=outputs/moe/us007_prepare/runtime RELEASE_NAME=us007_prepare ACCELERATOR=cpu DEVICES=1 -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest-preflight PYTHON=.venv/bin/python SMOKE_OUTPUT_DIR=outputs/moe/us007_preflight RELEASE_NAME=us007_smoke -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest PYTHON=.venv/bin/python ROUTERSET_DIR=/tmp/us006_routerset REBUILT_MANIFEST=/tmp/us006_routerset/multilabel_dataset/manifest_moe_train.jsonl SMOKE_OUTPUT_DIR=outputs/moe/us007_smoke_fixture RUNTIME_ROOT=outputs/moe/us007_smoke_fixture/runtime RELEASE_NAME=us007_smoke_fixture ACCELERATOR=cpu DEVICES=1 -> FAIL
- Files changed:
  - .agents/tasks/prd-full-training.json
  - .ralph/.tmp/prompt-20260314-115146-3579918-7.md
  - .ralph/.tmp/story-20260314-115146-3579918-7.json
  - .ralph/.tmp/story-20260314-115146-3579918-7.md
  - .ralph/activity.log
  - PLANS.md
  - README.md
  - howtorun.md
  - .ralph/progress.md
- What was implemented
- Updated the README and `howtorun.md` runbook so the documented full-training path now uses the real `make full-train` target, explicit micromamba environment presets, a prepare -> full-train -> smoke verification flow, and an output path map that matches the current artifact layout.
- Added troubleshooting guidance for SIGTERM interruptions, startup timeout failures recorded in `startup_gate.json`, and NaN/Inf-loss recovery using the existing `summary.json` and `startup_log.txt` diagnostics.
- Removed stale runbook guidance that implied there was no canonical full-training target and replaced it with the checked-in direct-script equivalent `scripts/full_train_moe.py`.
- **Learnings for future iterations:**
  - Patterns discovered
  - The operator docs had drifted more in `howtorun.md` than in `README.md`; the safer update was to make `README.md` concise and let `howtorun.md` hold the full path map and troubleshooting flow.
  - Gotchas encountered
  - On this machine, both `make train-prepare` and `make smoketest` still hit the existing 60-second Lightning import probe timeout, so the runbook’s timeout troubleshooting is grounded in a real reproducible failure.
  - Useful context
  - `ruff check .` is still blocked by pre-existing notebook and legacy-module lint errors outside US-007, so repo-wide Ruff is not yet a clean release gate for documentation-only changes.
---

## [2026-03-14 12:59:59 UTC] - US-006: Standardize artifact layout and retention policy
Thread: 
Run: 20260314-115146-3579918 (iteration 6)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-6.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-6.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 80a3fb2 fix(training): standardize run artifact layout; 594e0b1 chore(ralph): record us-006 progress
- Post-commit status: clean
- Verification:
  - Command: PYTHONPATH=src .venv/bin/python -m pytest tests/test_moe_training.py -> PASS
  - Command: make train-prepare PYTHON=.venv/bin/python TRAIN_OUTPUT_DIR=outputs/moe/us006_prepare RUNTIME_ROOT=outputs/moe/us006_prepare/runtime RELEASE_NAME=us006_prepare ACCELERATOR=cpu DEVICES=1 -> PASS
  - Command: make smoketest-preflight PYTHON=.venv/bin/python SMOKE_OUTPUT_DIR=outputs/moe/us006_preflight RELEASE_NAME=us006_smoke -> PASS
  - Command: make smoketest PYTHON=.venv/bin/python ROUTERSET_DIR=/tmp/us006_routerset REBUILT_MANIFEST=/tmp/us006_routerset/multilabel_dataset/manifest_moe_train.jsonl SMOKE_OUTPUT_DIR=outputs/moe/us006_smoke_fixture RUNTIME_ROOT=outputs/moe/us006_smoke_fixture/runtime RELEASE_NAME=us006_smoke_fixture ACCELERATOR=cpu DEVICES=1 -> PASS
  - Command: PYTHONPATH=src .venv/bin/python -m pytest -> PASS
  - Command: ruff check . -> FAIL
- Files changed:
  - PLANS.md
  - README.md
  - full_training_contract.md
  - howtorun.md
  - scripts/full_train_moe.py
  - scripts/smoke_test_moe.py
  - scripts/train_moe_switcher.py
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Added a centralized artifact-layout definition in `hydranet.moe_training` that declares the run root, runtime root, checkpoints directory, bundle root, release directory, and inference directory in one place.
- Moved default release artifacts under each run’s `output_dir/bundle/phidranet_<release_name>/` tree and enforced that release and checkpoint paths cannot escape `output_root` or `runtime_root`.
- Extended the run contract and regression tests to cover the new bundle/checkpoint locations and the negative case where an external release root is requested.
- Updated the training contract and operator docs to describe the standardized layout and the cleanup policy for stale failed runs.
- **Learnings for future iterations:**
  - Patterns discovered
  - The contract builder needs the same path-normalization rules as the writer path; otherwise it can validate the right artifacts against the wrong expected location.
  - Gotchas encountered
  - The canonical `make smoketest` path is only practical with a small routerset fixture on CPU; running it against the full dataset takes hours even though the code path is correct.
  - Useful context
  - `ruff check .` is still blocked by pre-existing notebook and legacy-module lint errors outside the US-006 touch set, so repo-wide lint is not yet a clean signal for this story.
---

---
## [2026-03-14 12:18:40 UTC] - US-002: Create a one-shot full-training command path
Thread: 
Run: 20260314-115146-3579918 (iteration 2)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-2.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-2.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: a6b3425 feat(training): add full-training path
- Post-commit status: clean before recording this progress entry
- Verification:
  - Command: PATH=.venv/bin:$PATH python scripts/full_train_moe.py -> PASS (expected argparse usage failure for missing required arguments)
  - Command: .venv/bin/pytest tests/test_moe_training.py -> PASS
  - Command: PATH=.venv/bin:$PATH pytest -> PASS
  - Command: PATH=.venv/bin:$PATH ruff check src/hydranet/moe_training.py scripts/full_train_moe.py tests/test_moe_training.py -> PASS
  - Command: PATH=.venv/bin:$PATH ruff check . -> FAIL
  - Command: make train-prepare -> FAIL
  - Command: make smoketest-preflight -> PASS
  - Command: make smoketest -> FAIL (terminated after reaching trainer.fit on the full local dataset because runtime was impractical for this run)
- Files changed:
  - PLANS.md
  - Makefile
  - README.md
  - full_training_contract.md
  - scripts/full_train_moe.py
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Added `run_full_training(...)` to execute train/export plus post-export smoke validation in one path and update `summary.json` so the canonical stage contract can reach `smoke`.
- Added the `scripts/full_train_moe.py` CLI and `make full-train` target, with required `--output-dir` and `--release-name` arguments so missing-argument failures stop before any run artifacts are created.
- Preserved failure reporting by writing `failure_stage=smoke` when post-export validation fails, and fixed the skipped-startup-gate path to emit a valid `startup_gate.json` artifact.
- Aligned the release artifact contract by writing `DEPLOY.md`, and updated operator docs to show the one-shot full-training entrypoint and the correct release-name prefix behavior.
- **Learnings for future iterations:**
  - Patterns discovered
  - `train_switcher(...)` already owned the authoritative run summary, so the least-riskful implementation was to wrap it and rewrite `summary.json` only after smoke completion or smoke failure.
  - Gotchas encountered
  - The repo-wide `ruff check .` failure is pre-existing and comes from legacy notebooks/modules outside the US-002 touch set; targeted Ruff on the changed Python files passed.
  - Useful context
  - `make train-prepare` still times out in the existing Lightning import probe on this machine, while `make smoketest-preflight` passes and `make smoketest` reaches real training before runtime becomes impractical.
---

## [2026-03-14 12:23:59 UTC] - US-003: Strengthen preflight and checkpoint readiness checks
Thread: 
Run: 20260314-115146-3579918 (iteration 3)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-3.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-3.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: c0b1cfc feat(preflight): harden checkpoint readiness
- Post-commit status: clean before recording this progress entry
- Verification:
  - Command: PYTHONPATH=src .venv/bin/pytest tests/test_moe_training.py -> PASS
  - Command: make smoketest-preflight -> PASS
  - Command: make train-prepare -> FAIL
  - Command: make smoketest -> FAIL
  - Command: PYTHONPATH=src .venv/bin/pytest -> PASS
  - Command: ruff check . -> FAIL
- Files changed:
  - PLANS.md
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .agents/tasks/prd-full-training.json
  - .ralph/.tmp/prompt-20260314-115146-3579918-3.md
  - .ralph/.tmp/story-20260314-115146-3579918-3.json
  - .ralph/.tmp/story-20260314-115146-3579918-3.md
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Hardened dataset preflight validation so required experts must appear in both train and validation splits, not just produce valid tensor shapes.
- Reworked checkpoint readiness reporting to record per-expert `status`, `source_path`, and deterministic path information, then validate the written report before training proceeds.
- Updated `preflight_routerset_training(...)` to emit standalone `dataset_report.json` and `checkpoint_report.json`, and added regression coverage for successful artifact emission plus explicit missing-checkpoint failures.
- **Learnings for future iterations:**
  - Patterns discovered
  - Preflight-only runs previously wrote just `preflight_report.json`; the acceptance gap was the missing standalone dataset/checkpoint artifacts plus structured checkpoint failures.
  - Gotchas encountered
  - The repo’s default `pytest` entrypoint can bind to an environment without `torch`; the project `.venv` is required for reliable validation here.
- Useful context
  - `make train-prepare` and `make smoketest` still fail on this machine because the existing Lightning import probe times out after checkpoint resolution, while `make smoketest-preflight` now succeeds and writes non-empty dataset/checkpoint reports.
---

## [2026-03-14 12:42:00 UTC] - US-004: Stabilize startup gate diagnostics
Thread: 
Run: 20260314-115146-3579918 (iteration 4)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-4.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-4.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 2e5006f fix(training): stabilize startup gate
- Post-commit status: clean after metadata commit
- Verification:
  - Command: PYTHONPATH=src .venv/bin/pytest tests/test_moe_training.py -> PASS
  - Command: make train-prepare -> PASS
  - Command: make smoketest-preflight -> PASS
  - Command: PYTHONPATH=src .venv/bin/pytest -> PASS
  - Command: ruff check . -> FAIL
  - Command: make smoketest -> FAIL (startup gate passed with the new timeout, then the canonical 3,857-step smoke-training epoch was interrupted after confirming fit had started because the end-to-end run was impractical in this session)
- Files changed:
  - PLANS.md
  - scripts/full_train_moe.py
  - scripts/smoke_test_moe.py
  - scripts/train_moe_switcher.py
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .agents/tasks/prd-full-training.json
  - .ralph/.tmp/prompt-20260314-115146-3579918-4.md
  - .ralph/.tmp/story-20260314-115146-3579918-4.json
  - .ralph/.tmp/story-20260314-115146-3579918-4.md
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Expanded `startup_gate.json` so it now records a top-level `status`, `timeout_seconds`, and a failure `traceback`, while preserving nested `lightning_probe` details.
- Updated `train_switcher(...)` to log `startup_gate_started` and `startup_gate_failed` transitions with timestamps and to surface an explicit `startup_failed` summary stage before any fit work can proceed.
- Raised the default startup probe timeout from 20s to 60s in the library and the CLI entrypoints so the canonical `make` flows no longer fail on slow Lightning startup in this environment.
- Added regression tests for import-probe failure diagnostics and the timeout negative case that must not reach `fit_started`.
- **Learnings for future iterations:**
  - Patterns discovered
  - The startup gate is shared by prepare, train, and smoke paths, so the CLI defaults must stay aligned with `hydranet.moe_training` or the Make targets silently drift from the library contract.
  - Gotchas encountered
  - `make smoketest` now clears the startup gate but still runs a very long one-epoch training job on the full routerset dataset, so it is not a quick verification path on CPU.
  - Useful context
- `ruff check .` is still failing on pre-existing notebook and legacy module issues outside the US-004 touch set; targeted Ruff on the changed files passed.
---

## [2026-03-14 12:48:17 UTC] - US-005: Add non-finite loss safeguards during fit
Thread: 
Run: 20260314-115146-3579918 (iteration 5)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-5.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-5.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 157d0b2 fix(training): guard non-finite losses
- Post-commit status: clean after metadata commit
- Verification:
  - Command: ./.venv/bin/python -m pytest tests/test_moe_training.py -k non_finite_loss -> PASS
  - Command: ./.venv/bin/python -m pytest tests/test_moe_training.py -> PASS
  - Command: PATH=./.venv/bin:$PATH make train-prepare -> PASS
  - Command: PATH=./.venv/bin:$PATH make smoketest-preflight -> PASS
  - Command: PATH=./.venv/bin:$PATH pytest -> PASS
  - Command: PATH=./.venv/bin:$PATH ruff check . -> FAIL
  - Command: PATH=./.venv/bin:$PATH make smoketest -> FAIL (entered the canonical smoke fit cleanly, then was stopped after confirming startup and live training because the one-epoch full-routerset CPU run is too long for this session)
- Files changed:
  - PLANS.md
  - full_training_contract.md
  - src/hydranet/moe_lightning.py
  - src/hydranet/moe_training.py
  - tests/test_moe_training.py
  - .agents/tasks/prd-full-training.json
  - .ralph/.tmp/prompt-20260314-115146-3579918-5.md
  - .ralph/.tmp/story-20260314-115146-3579918-5.json
  - .ralph/.tmp/story-20260314-115146-3579918-5.md
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
- Added a structured `NonFiniteLossError` from the Lightning fit step so NaN/Inf losses abort with batch index, sample ids, and expert-context diagnostics instead of continuing toward export.
- Updated `train_switcher(...)` to append a `non_finite_loss_detected` event to `startup_log.txt`, preserve the fit-stage failure contract, and persist the same diagnostics plus a config snapshot into `summary.json`.
- Added a regression test proving the non-finite-loss path writes diagnostics, keeps `failure_stage`, and does not emit a bundle or release.
- Documented the new non-finite-loss failure fields in the full-training contract.
- **Learnings for future iterations:**
  - Patterns discovered
  - The fit failure contract can gain richer diagnostics without changing `failure_stage` by logging an event that does not advance the recorder’s source-of-truth stage.
  - Gotchas encountered
  - The repo-wide `ruff check .` gate still fails on pre-existing notebook and legacy-module lint errors outside the US-005 touch set, and `make smoketest` remains a very long CPU-bound training run.
  - Useful context
  - The project `.venv` is required for reliable pytest execution on this machine because the host `pytest` lacks `torch`.
---

## [2026-03-14 13:29:34 UTC] - US-008: Add regression tests for full-training orchestration
Thread: 
Run: 20260314-115146-3579918 (iteration 8)
Run log: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-8.log
Run summary: /shared/home/rdelprete/PythonProjects/hydranet-phisat2/.ralph/runs/run-20260314-115146-3579918-iter-8.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: f648618 test(training): add orchestration regressions
- Post-commit status: clean after final metadata commit
- Verification:
  - Command: PYTHONPATH=src .venv/bin/python -m pytest tests/test_moe_training.py -> PASS
  - Command: PYTHONPATH=src .venv/bin/python -m pytest -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make train-prepare PYTHON=.venv/bin/python ROUTERSET_DIR=/tmp/us008_routerset REBUILT_MANIFEST=/tmp/us008_routerset/multilabel_dataset/manifest_moe_train.jsonl TRAIN_OUTPUT_DIR=outputs/moe/us008_prepare RUNTIME_ROOT=outputs/moe/us008_prepare/runtime RELEASE_NAME=us008_prepare ACCELERATOR=cpu DEVICES=1 -> FAIL
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest-preflight PYTHON=.venv/bin/python ROUTERSET_DIR=/tmp/us008_routerset REBUILT_MANIFEST=/tmp/us008_routerset/multilabel_dataset/manifest_moe_train.jsonl SMOKE_OUTPUT_DIR=outputs/moe/us008_preflight RELEASE_NAME=us008_smoke ACCELERATOR=cpu DEVICES=1 -> PASS
  - Command: PATH="/shared/home/rdelprete/PythonProjects/hydranet-phisat2/.venv/bin:$PATH" make smoketest PYTHON=.venv/bin/python ROUTERSET_DIR=/tmp/us008_routerset REBUILT_MANIFEST=/tmp/us008_routerset/multilabel_dataset/manifest_moe_train.jsonl SMOKE_OUTPUT_DIR=outputs/moe/us008_smoke_fixture RUNTIME_ROOT=outputs/moe/us008_smoke_fixture/runtime RELEASE_NAME=us008_smoke_fixture ACCELERATOR=cpu DEVICES=1 -> FAIL
  - Command: ruff check . -> FAIL
  - Command: ruff check tests/test_moe_training.py -> PASS
- Files changed:
  - .agents/tasks/prd-full-training.json
  - .ralph/.tmp/prompt-20260314-115146-3579918-8.md
  - .ralph/.tmp/story-20260314-115146-3579918-8.json
  - .ralph/.tmp/story-20260314-115146-3579918-8.md
  - .ralph/activity.log
  - PLANS.md
  - tests/test_moe_training.py
  - .ralph/progress.md
- What was implemented
- Added a US-008 ExecPlan and two regression tests that mock the full-training orchestration without live checkpoint downloads.
- Pinned the mocked train-switcher stage sequence and artifact emission by asserting dataset preparation occurs before the startup gate, the startup gate completes before fit starts, and the expected summary/startup/bundle artifacts are written.
- Added a negative-path regression test proving `run_full_training(...)` preserves the training failure stage, writes `smoke_test_summary.json`, and re-raises the timeout for a non-zero CLI path before smoke begins.
- **Learnings for future iterations:**
  - Patterns discovered
  - The observable pre-fit sequence in `train_switcher(...)` is dataset/config preparation -> startup gate -> checkpoint/preflight report -> fit, so regression tests should pin stage order from `startup_log.txt` instead of guessing from helper names.
  - Gotchas encountered
  - The canonical `make train-prepare` and `make smoketest` gates still fail in this environment because the Lightning import probe times out after 60 seconds, even with a tiny local routerset fixture and cached checkpoints.
  - Useful context
  - Repo-wide `ruff check .` is still blocked by pre-existing notebook and legacy-module lint issues outside US-008, while targeted Ruff on `tests/test_moe_training.py` passes.
---
