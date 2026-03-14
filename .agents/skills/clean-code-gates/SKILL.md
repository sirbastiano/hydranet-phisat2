---
name: clean-code-gates
description: Apply a clean-code and schematic-readability review to changed code. Use when a change is non-trivial, a refactor is involved, or maintainability matters. Do not use for docs-only or generated-output-only changes.
---
## Review checklist
Check the changed code for:
- cohesion and single responsibility
- top-level flow that reads like a schematic
- precise naming and clear ownership
- explicit side effects and failure paths
- controlled branching and nesting
- low duplication
- avoidance of boolean-flag APIs, magic values, and stringly-typed control flow
- tests that cover behavior rather than incidental internals

## Verdict scale
- `PASS` — no material maintainability concerns
- `WARN` — acceptable only with explicit residual debt
- `BLOCK` — materially harms maintainability or future change safety

## Output
- Verdict
- Findings by severity
- Suggested simplifications
- Residual debt or follow-up

## Common block reasons
- parsing, policy, orchestration, I/O, and persistence mixed together
- hidden temporal coupling or ambient state
- complexity increase without necessity
- debug leftovers or broad silent catches
