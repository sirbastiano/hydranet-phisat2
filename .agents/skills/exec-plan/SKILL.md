---
name: exec-plan
description: Create or refine a PLANS.md / ExecPlan for multi-step coding work with explicit invariants, validation, and rollback notes. Use when a change is non-trivial, cross-file, risky, stateful, or has unclear blast radius. Do not use for one-file typo fixes, formatting-only edits, or trivial doc tweaks.
---
## Use when
- the task touches multiple files, packages, or layers
- the blast radius is unclear
- a migration, rollout, dependency upgrade, or architecture change is involved
- you need a durable plan that implementation and review can follow

## Do not use when
- the task is a trivial one-file fix with obvious validation
- the work is docs-only and has no behavioral risk
- the request is already fully specified and no sequencing or rollback thinking is needed

## Outputs
Produce or update `PLANS.md` or an equivalent section in the handoff with:
- Goal
- Scope / non-goals
- Invariants and contracts to preserve
- Files / layers likely to change
- Ordered execution steps
- Validation plan
- Rollback / recovery notes
- Risks / blockers
- Definition of done

## ExecPlan template
See `PLANS.template.md` in this skill folder.

## Quality bar
- keep the plan concise and operational
- make dependencies between steps explicit
- call out what could invalidate the plan
- prefer reversible intermediate steps when possible
