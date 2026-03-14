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
