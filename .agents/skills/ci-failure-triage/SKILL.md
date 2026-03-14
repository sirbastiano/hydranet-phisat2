---
name: ci-failure-triage
description: Summarize failing CI, test, or build logs and derive the minimal repro path. Use when logs are large, failures are noisy, or CI broke unexpectedly. Do not use for fresh feature implementation or when the failure is already fully reproduced and understood locally.
---
## Use when
- CI is failing and the logs are large or noisy
- a local test/build failure needs quick triage
- you need a compact failure summary before reading raw logs

## Do not use when
- the task is normal feature work with no failure to investigate
- the failing command and root cause are already known
- the issue is primarily architectural rather than diagnostic

## Procedure
1. Identify the exact failing command, job, or step.
2. Summarize the failure in 3-6 bullets:
   - failing command
   - first concrete error
   - likely root cause
   - likely owning files / subsystem
   - minimal repro idea
3. Read only the smallest raw log slice needed to confirm the summary.
4. Produce a minimal local repro command.
5. State whether the next best step is:
   - implementer for code changes
   - test-engineer for stronger repro/coverage
   - dependency-curator for dependency/tooling issues
   - migration-planner if rollout/state caused the failure

## Output
- Failure summary
- Minimal repro
- Suspected root cause
- Next specialist or skill
- Validation command to confirm the fix

## Log budget rules
- prefer summaries before full logs
- avoid pasting raw multi-thousand-line logs into context when a targeted slice will do
- if summarization is uncertain, say so and identify the missing raw evidence
