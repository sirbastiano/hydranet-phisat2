---
name: migration-safety
description: Plan staged schema, data, config, and rollout-sensitive changes with forward and rollback paths. Use when a change affects state, storage, data compatibility, or rollout order. Do not use for pure code-only features with no persistence or rollout concerns.
---
## Use when
- schema or data shape changes
- compatibility windows matter
- backfills or staged rollouts are needed
- rollback safety matters

## Do not use when
- the change is code-only and stateless
- there is no migration, rollout, or compatibility impact

## Output
- Migration goal
- Compatibility strategy
- Forward plan
- Rollback plan
- Validation / dry-run commands
- Risks and approvals needed

## Checklist
- additive before destructive
- backfill before cutover when possible
- identify downstream consumers and generated-client impacts
- call out irreversible operations explicitly
- include post-deploy verification
