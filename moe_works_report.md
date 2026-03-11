# MoE Works Report

## Purpose

This document explains the Mixture-of-Experts (MoE) student path added to this repository, how it is assembled from existing HydraNet student checkpoints, how the local `routerset/` artifact is used for routing supervision, and what the current v1 limitations are.

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
- `fire`
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

Although the model catalog contains more student tasks, the MoE defaults were narrowed to the routerset-compatible set:

- `anomaly_detection`
- `fire`
- `worldfloods`

This is deliberate. The copied `routerset/` artifact is heterogeneous, and only those source families are currently compatible with the student’s 8-channel, channel-first input contract.

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

`routerset/manifest.jsonl` contains fields like:

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

- `routerset/images/<source_dataset>/<source_split>/...`

### Why only a subset of routerset is used

The full copied `routerset/` dataset mixes incompatible array layouts.

Examples found locally:

- `fire`: `(8, H, W)`
- `worldfloods`: `(8, H, W)`
- `anomaly_detection`: `(8, H, W)`
- `burned_area`: `(7, H, W)`
- `lc`: `(H, W, 10)`
- `roads`: channel layout incompatible with the current student input contract

HydraNet student models are currently built around fixed 8-channel input, so v1 intentionally excludes:

- `burned_area`
- `lc`
- `roads`

from routerset-backed switcher training.

## Dataset Adapter

The training dataset class is `RoutersetMoEDataset`.

Its behavior is:

- read `routerset/manifest.jsonl`
- keep only rows whose `source_dataset` is in the selected expert set
- keep only the default compatible source families
- keep only rows for the requested split
- skip rows whose backing `.npy` file is missing
- load arrays as float tensors
- require channel-first `8 x H x W`
- resize to a fixed square target size using bilinear interpolation

Each item returns:

- `image`
- `target`
- `expert_name`
- `source_sample_id`
- `image_path`
- `record_status`
- `label_names`

The target is a one-hot vector over the selected experts.

## Lightning Training Flow

Training uses:

- `RoutersetMoEDataModule`
- `MoESwitcherLightningModule`

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

It:

- builds the MoE with `load_student_moe(...)`
- creates the routerset DataModule
- saves run config and baseline summary
- trains the switcher
- writes metrics
- saves the bundled MoE checkpoint
- writes validation routing predictions

Default outputs land under:

- `outputs/moe/<timestamp>/`

Saved artifacts include:

- `config.json`
- `baseline_summary.json`
- `metrics.json`
- `student_moe_bundle.pt`
- `routing_predictions.jsonl`
- `summary.json`

## Inference Script

`scripts/infer_moe_switcher.py` loads a saved bundle and runs routing inference on routerset validation data.

It:

- loads the bundled MoE
- builds the routerset validation dataloader
- runs routing prediction
- writes `routing_predictions.jsonl`
- writes a small `summary.json`

This script is aimed at routing inspection rather than downstream task evaluation.

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

### Constraint 2: Input compatibility is limited

The routerset adapter currently supports only samples that already match the HydraNet student input layout:

- channel-first
- 8 channels

Other source families are excluded instead of being adapted.

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

1. Add deterministic adapters for non-8-channel routerset sources
2. Expand the default expert set after those adapters are defined
3. Add a semantic label-to-expert target builder for true multi-hot supervision
4. Add richer evaluation reports for routing quality and expert usage balance
5. Optionally support partial fine-tuning of the encoder or experts after switcher warm-up

## Practical Summary

In the current implementation, the MoE system works like this:

- load several compatible HydraNet student checkpoints
- copy one encoder to become shared
- copy each checkpoint decoder to become an expert
- train a small routing head on local routerset source labels
- save everything into a single bundle
- reload the bundle later and activate only the selected decoder experts at inference

That gives this repository a working, routerset-backed MoE student path without changing the base HydraNet student architecture.
