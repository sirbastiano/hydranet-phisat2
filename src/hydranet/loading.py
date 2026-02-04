"""Public model-loading helpers for HydraNet."""

from __future__ import annotations

from typing import Optional, Sequence

from .models.student import (
    DEFAULT_STUDENT_CONFIG,
    PhisatNet,
    create_phisatnet,
    list_phisatnet_configs,
)
from .models.teacher import PhiSatNetDownstream


def load_student(preset: str = DEFAULT_STUDENT_CONFIG, **overrides: object) -> PhisatNet:
    """Load the student model with a preset (default: Myriad optimized)."""
    return create_phisatnet(config=preset, **overrides)


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
    img_size: int = 224,
    norm_foundation: str = "group",
    norm_downstream: str = "batch",
    activation: str = "gelu",
    freeze_body: bool = False,
) -> PhiSatNetDownstream:
    """Load the teacher model (PhiSatNetDownstream)."""
    return PhiSatNetDownstream(
        pretrained_path=pretrained_path,
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
