# MoE Works Report

## Purpose

This document explains the Mixture-of-Experts (MoE) student path added to this repository, how it is assembled from existing HydraNet student checkpoints, how the local `routerset/` artifact is used for routing supervision, and how the final `phidranet` release package is created.

The implementation is centered around these files:

- `src/hydranet/models/moe_student.py`
- `src/hydranet/loading.py`
- `src/hydranet/moe_training.py`
- `scripts/train_moe_switcher.py`
- `scripts/infer_moe_switcher.py`

## High-Level Design

The MoE model keeps the original HydraNet student architecture split into three conceptual parts:

1. A shared encoder
2. A learned routing switcher
3. A set of task-specific decoder experts

The idea is:

- run the input once through a shared encoder
- compute routing logits from the encoder bottleneck
- choose one or more experts
- run only the selected decoders

In v1, the decoder experts come from existing student checkpoints for downstream tasks, while the switcher is newly trained to predict which expert should be activated.

## Core Components

### `SharedStudentEncoder`

`SharedStudentEncoder` wraps the encoder path of a `PhisatNet` student model:

- `encoders`
- `pools`
- `bottleneck`

Its forward pass returns:

- bottleneck features
- skip connections from each encoder stage

Those skip connections are reused by whichever decoder experts are activated.

### `StudentDecoderExpert`

`StudentDecoderExpert` wraps the decoder side of one student checkpoint:

- `upsamplers`
- `decoders`
- `final_conv`

Each expert is tied to one task name such as:

- `anomaly_detection`
- `burned_area`
- `fire`
- `lc`
- `roads`
- `worldfloods`

Each expert can have a different output channel count because the downstream tasks do not share the same output head shape.

### `MoESwitcher`

`MoESwitcher` is a small MLP over the encoder bottleneck:

- adaptive global average pooling
- flatten
- linear
- GELU
- dropout
- final linear layer to expert logits

It predicts one routing logit per expert.

### `MoEStudent`

`MoEStudent` owns:

- one shared encoder
- one switcher
- a dictionary of decoder experts

Its forward output is a structured dictionary:

- `routing_logits`
- `routing_probs`
- `active_experts`
- `expert_outputs`

`expert_outputs` is intentionally a dict keyed by expert name because decoder outputs are not forced into one common tensor shape.

## Routing Logic

Routing is handled inside `MoEStudent.route(...)`.

The rule is fixed in v1:

1. Apply `sigmoid` to the routing logits
2. Mark experts active when `prob >= threshold`
3. If more experts pass than allowed, keep only the highest-probability `top_k`
4. If no expert passes the threshold, fall back to `argmax`

This means the model always activates at least one expert.

At inference time, only the experts selected by the union of `active_experts` across the batch are executed, and only those appear in `expert_outputs`.

## How Expert Assembly Works

Expert assembly is done through `load_student_moe(...)` in `src/hydranet/loading.py`.

### Checkpoint selection

The loader reads `src/hydranet/model_weights.csv` and resolves student checkpoints by:

- `training`
- `task`
- `n_shots`

The default v1 configuration is:

- `training="finetuning"`
- `n_shots=5000`

### Default expert set

The default MoE expert set is aligned to the full routerset task coverage:

- `anomaly_detection`
- `burned_area`
- `fire`
- `lc`
- `roads`
- `worldfloods`

The copied `routerset/` artifact is heterogeneous, so the training pipeline now includes deterministic adapters that normalize every source family into the student’s shared 8-channel, channel-first input contract.

### Architecture compatibility checks

Before assembly, all selected student models are validated by `validate_student_architecture_compatibility(...)`.

The following fields must match across experts:

- `n_channels`
- `base_filters`
- `depth`
- `channel_multipliers`
- derived internal channel sizes

If they do not match, MoE assembly raises an error instead of silently producing a broken shared encoder.

### Encoder source

The shared encoder is copied from one chosen task checkpoint.

The default rule is:

- use `encoder_source_task` if explicitly provided
- otherwise use the alphabetically first expert name

With the default expert set, that is `anomaly_detection`.

## Bundle Format

The MoE can be serialized into a single-file bundle with:

- `save_student_moe_bundle(...)`
- `load_student_moe_bundle(...)`

The saved bundle contains:

- a bundle type marker
- a bundle version
- model config
- full MoE state dict
- optional run metadata

The config section stores:

- threshold
- `top_k`
- encoder source task
- encoder architecture config
- switcher config
- per-expert metadata

Per-expert metadata includes:

- expert name
- output channel count
- training mode
- `n_shots`
- resolved checkpoint path when available

This lets the model be reconstructed without re-querying Hugging Face or the original individual checkpoint files.

## Routerset Training Data

The routing supervision pipeline is implemented in `src/hydranet/moe_training.py`.

### Why routerset is not used directly as semantic labels

`routerset/multilabel_dataset/manifest.jsonl` contains fields like:

- `source_dataset`
- `label_names`
- `record_status`

For this v1 implementation, routing targets are taken from `source_dataset`, not from the semantic `label_names`.

So the switcher learns:

- fire sample -> activate `fire`
- anomaly detection sample -> activate `anomaly_detection`
- worldfloods sample -> activate `worldfloods`

That makes the current training target effectively one-hot for the default expert set.

### How sample paths are resolved

The dataset adapter reconstructs local `.npy` paths from manifest rows using:

- `source_dataset`
- `source_split`
- `source_sample_id`
- `patch_x`
- `patch_y`
- `patch_width`
- `patch_height`

This becomes a filename like:

- `0000010_0_0_full_full.npy`
- `0060356_0_256_256_256.npy`

and is resolved under:

- `routerset/multilabel_dataset/images/<source_dataset>/<source_split>/...`

### How all routerset sources are adapted

The full copied `routerset/` dataset mixes incompatible array layouts.

Examples found locally:

- `fire`: `(8, H, W)`
- `worldfloods`: `(8, H, W)`
- `anomaly_detection`: `(8, H, W)`
- `burned_area`: `(7, H, W)`
- `lc`: `(H, W, 10)`
- `roads`: channel-last or non-8-channel arrays depending on source material

HydraNet student models are built around fixed 8-channel input, so the training pipeline applies deterministic source adapters:

- `fire`, `worldfloods`, `anomaly_detection`: kept as channel-first when already compatible
- `burned_area`: padded from 7 channels to 8
- `lc`: channel-last tensors are moved to channel-first and truncated from 10 channels to 8
- `roads`: channel-last tensors are moved to channel-first and truncated from 10 channels to 8

All sources are then converted into the canonical training tensor shape `8 x 256 x 256` without resizing.

For large `anomaly_detection` inputs, the training path does not collapse the whole scene into one small image. Instead, it expands each full frame into deterministic non-overlapping `256 x 256` tiles and trains on those tiles as individual routing samples.

For non-tiled samples:

- if the spatial size is larger than `256`, the tensor is deterministically cut out to `256 x 256`
- if the spatial size is smaller than `256`, the tensor is zero-padded onto a `256 x 256` canvas

## Dataset Adapter

The training dataset class is `RoutersetMoEDataset`.

Its behavior is:

- read `routerset/multilabel_dataset/manifest.jsonl`
- keep only rows whose `source_dataset` is in the selected expert set
- keep only rows for the requested split
- skip rows whose backing `.npy` file is missing
- load arrays as float tensors
- normalize every array into channel-first `8 x H x W`
- tile large `anomaly_detection` frames into deterministic `256 x 256` samples
- cut out non-tiled sources that are larger than the target window
- zero-pad non-tiled sources that are smaller than the target window

Each item returns:

- `image`
- `target`
- `expert_name`
- `source_sample_id`
- `image_path`
- `record_status`
- `tile_origin`

The target is a one-hot vector over the selected experts.

## Lightning Training Flow

Training uses:

- `RoutersetMoEDataModule`
- `MoESwitcherLightningModule`

The routerset dataset/preflight path is importable without Lightning. Lightning is loaded lazily only for real training, and failed startup now writes `startup_log.txt` plus `startup_stage.json`.

### Frozen modules

In v1:

- the shared encoder is frozen
- all decoder experts are frozen
- only the switcher is trained

This keeps the first version simple and isolates the routing problem from decoder fine-tuning.

### Loss

The switcher is trained with:

- `BCEWithLogitsLoss`

This matches the MoE routing head design and leaves room for future multi-hot expert supervision, even though the current routerset target policy is one-hot.

### Logged metrics

The Lightning module logs:

- loss
- precision
- recall
- F1

The prediction threshold is the same threshold used by MoE routing.

## Training Script

`scripts/train_moe_switcher.py` is the command-line entrypoint for switcher training.

It supports two workflows:

- `--preflight-only` for dataset/checkpoint validation without training
- full training plus final release-package creation

In full training mode it:

- builds the MoE with `load_student_moe(...)`
- creates the routerset DataModule
- saves run config and baseline summary
- saves dataset and checkpoint reports
- trains the switcher
- writes metrics
- saves the bundled MoE checkpoint
- writes validation routing predictions
- creates the final `outputs/phidranet/phidranet_<name>/` release directory

Default outputs land under:

- `outputs/moe/<timestamp>/`

Saved run artifacts include:

- `config.json`
- `baseline_summary.json`
- `dataset_report.json`
- `checkpoint_report.json`
- `metrics.json`
- `student_moe_bundle.pt`
- `routing_predictions.jsonl`
- `summary.json`

Saved release artifacts include:

- `student_moe_bundle.pt`
- `config.json`
- `metrics.json`
- `dataset_report.json`
- `checkpoint_report.json`
- `routing_predictions.jsonl`
- `release_manifest.json`
- `DEPLOYMENT.md`

## Inference Script

`scripts/infer_moe_switcher.py` loads a saved bundle and runs routing inference on routerset validation data.

It:

- loads the bundled MoE
- builds the routerset validation dataloader
- runs routing prediction
- writes `routing_predictions.jsonl`
- writes a small `summary.json`

This script is aimed at routing inspection rather than downstream task evaluation.

## Smoke Test Script

`scripts/smoke_test_moe.py` runs a fast functional check of the complete MoE path.

It performs these steps in one command:

- rebuild routerset splits for MoE training if needed
- run preflight on the rebuilt manifest
- train the switcher for one epoch
- reload the saved bundle
- run validation routing inference
- write a compact smoke summary artifact

This is intended for trainability verification, not final model quality.

## Preflight Validation

`preflight_routerset_training(...)` validates the training inputs before a final run.

It reports:

- per-split sample counts
- per-task shape distributions before normalization and at final training shape
- selected expert set
- resolved checkpoint path per expert
- target image size and channel count

It also enforces the canonical training gate:

- every selected expert must have positive samples in both the train and validation split
- every emitted training tensor must have the exact final shape `8 x 256 x 256`

If either condition fails, preflight raises an error and the canonical run is blocked until routerset is rebuilt.

The CLI exposes this with:

```bash
PYTHONPATH=src python3 scripts/train_moe_switcher.py --preflight-only
```

This is the recommended first step before starting a canonical `phidranet` training run.

## Final `phidranet` Release Package

The final creation workflow now produces a release-style artifact directory under:

- `outputs/phidranet/phidranet_<name>/`

That directory is intended to be the handoff artifact for the trained student-with-MoE-decoders model.

It contains:

- the final bundle
- the exact training config
- the final metrics snapshot
- the dataset and checkpoint reports used to create it
- a validation routing report
- a release manifest
- a short deployment note

## Baseline Capture

`capture_baseline_summary(...)` writes a simple structural snapshot of the assembled MoE:

- encoder source task
- threshold
- `top_k`
- number of experts
- per-expert parameter counts
- encoder parameter count
- switcher parameter count

This is not a full benchmark, but it gives a reproducible baseline artifact for each run.

## Tests Added

Two test modules were added:

- `tests/test_moe_student.py`
- `tests/test_moe_training.py`

They cover:

- architecture mismatch detection
- routing threshold and fallback behavior
- bundle save/load round-trip behavior
- active-expert-only output behavior
- routerset path/token reconstruction
- routerset dataset filtering
- tiny Lightning fit on a synthetic routerset-like fixture

## Current v1 Constraints

The implementation is functional, but it is intentionally narrow.

### Constraint 1: Routerset-backed targets are source-based

The switcher currently learns expert IDs from `source_dataset`, not from semantic label combinations in `label_names`.

That means:

- this is routing by originating dataset/task
- not yet routing by semantic content across tasks

### Constraint 2: Input adaptation is deterministic and simple

The routerset adapter now supports all routerset task families, but the adaptation policy is intentionally simple:

- transpose when channels are last
- truncate channels when there are more than 8
- zero-pad when there are fewer than 8

This is reproducible and enough for the current release path, but it is still a heuristic adapter rather than a learned source-specific stem.

For `anomaly_detection`, the deterministic policy is tile expansion rather than whole-image resize.

### Constraint 3: Experts are frozen

Only the switcher is trained in v1.

This is good for a stable first integration, but it means the experts are not being co-adapted to the routing policy.

### Constraint 4: Batch execution still evaluates the union of active experts

For efficiency, the model executes only experts that are active somewhere in the batch. It does not yet do per-sample sparse execution scheduling beyond that.

### Constraint 5: No fused output tensor

The canonical output is:

- routing outputs
- a dict of per-expert decoder outputs

There is no unified merged prediction tensor in this version.

## Recommended Next Steps

The most useful follow-up improvements would be:

1. Add a semantic label-to-expert target builder for true multi-hot supervision
2. Add richer evaluation reports for routing quality and expert usage balance
3. Optionally support partial fine-tuning of the encoder or experts after switcher warm-up
4. Replace simple channel truncation with more principled source-specific feature selection if needed
5. Add explicit release versioning policy if these artifacts are to be published externally

## Practical Summary

In the current implementation, the MoE system works like this:

- load several compatible HydraNet student checkpoints
- copy one encoder to become shared
- copy each checkpoint decoder to become an expert
- train a small routing head on local routerset source labels
- save everything into a single bundle
- package the final trained model as a `phidranet` release directory
- reload the bundle later and activate only the selected decoder experts at inference

That gives this repository a working, routerset-backed MoE student path and a final release-package creation flow without changing the base HydraNet student architecture.
