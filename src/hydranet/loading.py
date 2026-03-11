"""Public model-loading helpers for HydraNet."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .models.student import (
    DEFAULT_STUDENT_CONFIG,
    PhisatNet,
    create_phisatnet,
    list_phisatnet_configs,
)
from .models.moe_student import (
    MoEStudent,
    build_moe_student_from_models,
)
from .models.teacher import PhiSatNetDownstream
from .weights import get_model_weights


def _make_student_checkpoint_compatible(model: PhisatNet, state_dict: dict) -> dict:
    """
    Apply backward-compatible key fixes for student checkpoints.

    - Maps legacy classifier head weights to `final_conv.*` when needed.
    - Fills missing ConvNeXt layer-scale gamma parameters from model defaults.
    """
    patched_state = dict(state_dict)
    model_state = model.state_dict()

    # Legacy head naming used in some checkpoints.
    if "final_conv.weight" not in patched_state and "classifier.weight" in patched_state:
        if model_state["final_conv.weight"].shape == patched_state["classifier.weight"].shape:
            patched_state["final_conv.weight"] = patched_state["classifier.weight"]
            patched_state.pop("classifier.weight", None)
    if "final_conv.bias" not in patched_state and "classifier.bias" in patched_state:
        if model_state["final_conv.bias"].shape == patched_state["classifier.bias"].shape:
            patched_state["final_conv.bias"] = patched_state["classifier.bias"]
            patched_state.pop("classifier.bias", None)

    # Older checkpoints may not include layer-scale gamma parameters.
    for key in model_state:
        if key.endswith(".convnext_block.gamma") and key not in patched_state:
            patched_state[key] = model_state[key]

    return patched_state


def load_student(
    preset: str = "checkpoint",  # Changed default to match HF checkpoint architecture
    *,
    task: Optional[str] = None,
    n_shots: Optional[int] = None,
    training: str = "finetuning",
    weights_dir: Optional[str] = None,
    auto_load_weights: bool = False,
    strict: bool = False,  # Allow loading with architecture mismatches
    **overrides: object,
) -> PhisatNet:
    """
    Load the student model with optional automatic weight loading.
    
    Args:
        preset: Model preset configuration (default: 'checkpoint' matches HF checkpoints)
        task: Task name for weight loading (e.g., 'anomaly_detection', 'worldfloods')
        n_shots: Number of shots for weight loading (e.g., 50, 100, 500, 1000, 5000)
        training: Training type (default: 'finetuning')
        weights_dir: Directory to save/load weights
        auto_load_weights: If True and task/n_shots provided, load weights automatically
        strict: If True, require exact state_dict match. If False (default), allow mismatches
        **overrides: Additional model configuration overrides
    
    Returns:
        PhisatNet model instance (with weights loaded if requested)
    
    Example:
        >>> # Load model without weights
        >>> model = load_student()
        
        >>> # Load model with automatic weight download
        >>> model = load_student(
        ...     task='anomaly_detection',
        ...     n_shots=1000,
        ...     auto_load_weights=True
        ... )
    """
    import torch
    import torch.nn as nn
    
    model = create_phisatnet(config=preset, **overrides)
    
    # Load weights if requested
    if auto_load_weights and task is not None and n_shots is not None:
        weight_paths = get_model_weights(
            training=training,
            model='student',
            task=task,
            n_shots=n_shots,
            download_dir=weights_dir,
            latest_only=True,
        )
        if weight_paths:
            print(f"Loading weights from: {weight_paths[0]}")
            state_dict = torch.load(weight_paths[0], map_location='cpu', weights_only=True)

            # Auto-align output classes to checkpoint head when user did not set n_classes.
            if "n_classes" not in overrides and "classifier.weight" in state_dict:
                ckpt_out = int(state_dict["classifier.weight"].shape[0])
                model_out = int(model.final_conv.out_channels)
                if ckpt_out != model_out:
                    model.final_conv = nn.Conv2d(model.final_conv.in_channels, ckpt_out, kernel_size=1)
                    model.n_classes = ckpt_out
                    print(f"  Adjusted n_classes to checkpoint head: {ckpt_out}")

            state_dict = _make_student_checkpoint_compatible(model, state_dict)
            
            # Load with strict=False by default to handle classifier vs segmentation head differences
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            
            if not strict:
                if missing_keys:
                    print(f"  Note: {len(missing_keys)} keys not found in checkpoint (using random init)")
                if unexpected_keys:
                    print(f"  Note: {len(unexpected_keys)} keys in checkpoint not used (e.g., task-specific heads)")
    
    return model


def available_student_presets() -> list[str]:
    """Return available student preset names."""
    return list_phisatnet_configs()


def load_teacher(
    *,
    task: str,
    input_dim: int,
    output_dim: int,
    depths: Sequence[int],
    dims: Sequence[int],
    pretrained_path: Optional[str] = None,
    n_shots: Optional[int] = None,
    training: str = "finetuning",
    weights_dir: Optional[str] = None,
    auto_load_weights: bool = False,
    img_size: int = 224,
    norm_foundation: str = "group",
    norm_downstream: str = "batch",
    activation: str = "gelu",
    freeze_body: bool = False,
) -> PhiSatNetDownstream:
    """
    Load the teacher model (PhiSatNetDownstream) with optional automatic weight loading.
    
    Args:
        task: Task type (e.g., 'anomaly_detection', 'worldfloods')
        input_dim: Input dimension
        output_dim: Output dimension
        depths: Model depths sequence
        dims: Model dimensions sequence
        pretrained_path: Path to pretrained weights (if not using auto_load_weights)
        n_shots: Number of shots for auto weight loading (e.g., 50, 100, 500, 1000, 5000)
        training: Training type for auto weight loading (default: 'finetuning')
        weights_dir: Directory to save/load weights
        auto_load_weights: If True and n_shots provided, download and load latest weights
        img_size: Input image size (default: 224)
        norm_foundation: Foundation normalization type (default: 'group')
        norm_downstream: Downstream normalization type (default: 'batch')
        activation: Activation function (default: 'gelu')
        freeze_body: Whether to freeze the body (default: False)
    
    Returns:
        PhiSatNetDownstream model instance (with weights loaded if requested)
    
    Example:
        >>> # Load teacher with automatic weight download
        >>> model = load_teacher(
        ...     task='anomaly_detection',
        ...     input_dim=4,
        ...     output_dim=2,
        ...     depths=[2, 2, 6, 2],
        ...     dims=[64, 128, 256, 512],
        ...     n_shots=1000,
        ...     auto_load_weights=True
        ... )
    """
    import torch
    
    # If auto_load_weights is enabled, download the latest weights
    final_pretrained_path = pretrained_path
    if auto_load_weights and n_shots is not None:
        weight_paths = get_model_weights(
            training=training,
            model='teacher',
            task=task,
            n_shots=n_shots,
            download_dir=weights_dir,
            latest_only=True,
        )
        if weight_paths:
            final_pretrained_path = weight_paths[0]
            print(f"Using auto-downloaded weights: {final_pretrained_path}")
    
    return PhiSatNetDownstream(
        pretrained_path=final_pretrained_path,
        task=task,
        input_dim=input_dim,
        output_dim=output_dim,
        depths=list(depths),
        dims=list(dims),
        img_size=img_size,
        norm_foundation=norm_foundation,
        norm_downstream=norm_downstream,
        activation=activation,
        freeze_body=freeze_body,
    )


def _load_student_catalog_rows() -> list[dict[str, str]]:
    catalog_path = Path(__file__).parent / "model_weights.csv"
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row["model"] == "student" and row.get("task")]


def _resolve_default_moe_experts(
    *,
    allowed_tasks: Optional[Sequence[str]] = None,
    training: str = "finetuning",
    n_shots: int = 5000,
) -> list[str]:
    rows = _load_student_catalog_rows()
    available = {
        row["task"]
        for row in rows
        if row["training"] == training and row.get("n_shots") == str(float(n_shots))
    }
    if allowed_tasks is not None:
        available &= set(allowed_tasks)
    if not available:
        raise ValueError(
            f"No student expert checkpoints available for training={training!r}, n_shots={n_shots}."
        )
    return sorted(available)


def load_student_moe(
    *,
    expert_tasks: Optional[Sequence[str]] = None,
    allowed_default_tasks: Sequence[str] = ("anomaly_detection", "fire", "worldfloods"),
    encoder_source_task: Optional[str] = None,
    preset: str = "checkpoint",
    training: str = "finetuning",
    n_shots: int = 5000,
    weights_dir: Optional[str] = None,
    auto_load_weights: bool = True,
    strict: bool = False,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    switcher_hidden_dim: Optional[int] = None,
    switcher_dropout: float = 0.0,
    **overrides: object,
) -> MoEStudent:
    """
    Assemble a MoE student from multiple student checkpoints.

    By default this uses the routerset-compatible expert subset.
    """
    selected_tasks = list(expert_tasks or _resolve_default_moe_experts(
        allowed_tasks=allowed_default_tasks,
        training=training,
        n_shots=n_shots,
    ))
    student_models: dict[str, PhisatNet] = {}
    expert_metadata: dict[str, dict[str, object]] = {}

    for task in selected_tasks:
        checkpoint_path = None
        if auto_load_weights:
            weight_paths = get_model_weights(
                training=training,
                model="student",
                task=task,
                n_shots=n_shots,
                download_dir=weights_dir,
                latest_only=True,
            )
            checkpoint_path = weight_paths[0] if weight_paths else None
        model = load_student(
            preset=preset,
            task=task,
            n_shots=n_shots,
            training=training,
            weights_dir=weights_dir,
            auto_load_weights=False,
            strict=strict,
            **overrides,
        )
        if checkpoint_path is not None:
            import torch
            import torch.nn as nn

            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if "n_classes" not in overrides and "classifier.weight" in state_dict:
                ckpt_out = int(state_dict["classifier.weight"].shape[0])
                model_out = int(model.final_conv.out_channels)
                if ckpt_out != model_out:
                    model.final_conv = nn.Conv2d(model.final_conv.in_channels, ckpt_out, kernel_size=1)
                    model.n_classes = ckpt_out
            state_dict = _make_student_checkpoint_compatible(model, state_dict)
            model.load_state_dict(state_dict, strict=strict)
        student_models[task] = model
        expert_metadata[task] = {
            "training": training,
            "n_shots": int(n_shots),
            "checkpoint_path": checkpoint_path,
            "threshold": float(threshold),
        }

    return build_moe_student_from_models(
        student_models,
        encoder_source_task=encoder_source_task,
        threshold=threshold,
        top_k=top_k,
        switcher_hidden_dim=switcher_hidden_dim,
        switcher_dropout=switcher_dropout,
        expert_metadata=expert_metadata,
    )


def save_student_moe_bundle(
    model: MoEStudent,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Save a single-file bundle containing the shared encoder, switcher, and decoder experts."""
    import torch

    bundle_path = Path(path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_type": "hydranet_student_moe",
        "bundle_version": 1,
        "model_config": model.get_bundle_config(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, bundle_path)
    return bundle_path


def load_student_moe_bundle(path: str | Path, *, map_location: str = "cpu") -> MoEStudent:
    """Load a MoE student from a bundle saved by `save_student_moe_bundle`."""
    import torch

    bundle = torch.load(path, map_location=map_location, weights_only=False)
    if bundle.get("bundle_type") != "hydranet_student_moe":
        raise ValueError(f"Unsupported bundle type: {bundle.get('bundle_type')!r}")

    config = dict(bundle["model_config"])
    encoder_config = dict(config["encoder_config"])
    source_task = config["encoder_source_task"]
    expert_configs: dict[str, dict[str, object]] = {
        name: dict(value) for name, value in config["experts"].items()
    }

    student_models: dict[str, PhisatNet] = {}
    for name, expert in expert_configs.items():
        model = create_phisatnet(
            config="checkpoint",
            n_channels=int(encoder_config["n_channels"]),
            n_classes=int(expert["output_channels"]),
            base_filters=int(encoder_config["base_filters"]),
            depth=int(encoder_config["depth"]),
            channel_multipliers=list(encoder_config["channel_multipliers"]),
        )
        student_models[name] = model

    moe_model = build_moe_student_from_models(
        student_models,
        encoder_source_task=source_task,
        threshold=float(config["threshold"]),
        top_k=int(config["top_k"]),
        switcher_hidden_dim=int(config["switcher"]["hidden_dim"]),
        switcher_dropout=float(config["switcher"]["dropout"]),
        expert_metadata=expert_configs,
    )
    moe_model.load_state_dict(bundle["state_dict"])
    return moe_model
