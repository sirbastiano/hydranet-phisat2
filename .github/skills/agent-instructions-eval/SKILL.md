---
name: agent-instructions-eval
description: Evaluate AGENTS.md, skills, custom agents, and instruction files with a small routing-focused prompt set. Use when changing repository instructions or agent behavior. Do not use for ordinary product-code changes unrelated to the agent system.
---
## Use when
- `AGENTS.md` changes
- files under `.github/agents/`, `.github/skills/`, `.agents/skills/`, or `.codex/` change
- you changed a skill description, routing rule, tool scope, or output contract

## Do not use when
- the task changes only product code and not the agent system
- there is no behavior change in instruction files

## Success criteria
A minimal instruction eval should answer:
- Did the right skill or specialist trigger?
- Did adjacent tasks correctly *not* trigger it?
- Were expected commands or tools used?
- Did the output follow the requested contract or schema?

## Required prompt set
Start with 4-10 prompts:
- at least one explicit positive trigger
- at least one implicit positive trigger
- at least one negative trigger for an adjacent task
- at least one noisy/realistic prompt

A starter CSV is in `prompts.csv.example`.

## Output contract
When practical, score the run against `result.schema.json` or an equivalent schema so results are comparable across revisions.

## Evaluation loop
1. Define the must-pass checks first.
2. Run the prompt set.
3. Record routing behavior, commands/tools used, and outputs.
4. Add newly discovered misses as new prompts.
5. Keep the set small and focused on the failure modes you care about.

## Deliverable
- Prompt cases run
- Pass/fail by case
- False positives / false negatives
- Tool-scope or command surprises
- Instruction changes still needed
