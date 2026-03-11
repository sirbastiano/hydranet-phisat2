"""Mixture-of-experts student model built from HydraNet student checkpoints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import torch
import torch.nn as nn

from .student import PhisatNet


def student_architecture_signature(model: PhisatNet) -> dict:
    """Return the architecture fields that must match across shared experts."""
    return {
        "n_channels": int(model.n_channels),
        "base_filters": int(model.base_filters),
        "depth": int(model.depth),
        "channel_multipliers": list(model.channel_multipliers),
        "channels": list(model.channels),
    }


def validate_student_architecture_compatibility(models: Mapping[str, PhisatNet]) -> dict:
    """Validate that the provided student checkpoints share the same encoder/decoder shape."""
    if not models:
        raise ValueError("At least one student model is required to build a MoE student.")

    items = list(models.items())
    reference_name, reference_model = items[0]
    reference_signature = student_architecture_signature(reference_model)

    for name, model in items[1:]:
        signature = student_architecture_signature(model)
        if signature != reference_signature:
            raise ValueError(
                "Student architecture mismatch between "
                f"{reference_name} and {name}: {reference_signature} != {signature}"
            )

    return reference_signature


class SharedStudentEncoder(nn.Module):
    """Encoder path extracted from a HydraNet student checkpoint."""

    def __init__(
        self,
        encoders: nn.ModuleList,
        pools: nn.ModuleList,
        bottleneck: nn.Module,
        *,
        depth: int,
        channels: Iterable[int],
        n_channels: int,
    ) -> None:
        super().__init__()
        self.encoders = encoders
        self.pools = pools
        self.bottleneck = bottleneck
        self.depth = int(depth)
        self.channels = list(channels)
        self.n_channels = int(n_channels)
        self.out_channels = int(self.channels[-1])

    @classmethod
    def from_student(cls, model: PhisatNet) -> "SharedStudentEncoder":
        return cls(
            encoders=deepcopy(model.encoders),
            pools=deepcopy(model.pools),
            bottleneck=deepcopy(model.bottleneck),
            depth=model.depth,
            channels=model.channels,
            n_channels=model.n_channels,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        current = x

        for index in range(self.depth):
            current = self.encoders[index](current)
            skips.append(current)
            if index < self.depth - 1:
                current = self.pools[index](current)

        current = self.pools[-1](current)
        current = self.bottleneck(current)
        return current, skips


class StudentDecoderExpert(nn.Module):
    """Decoder branch extracted from one student checkpoint."""

    def __init__(
        self,
        upsamplers: nn.ModuleList,
        decoders: nn.ModuleList,
        final_conv: nn.Module,
        *,
        depth: int,
        output_channels: int,
        task: str,
    ) -> None:
        super().__init__()
        self.upsamplers = upsamplers
        self.decoders = decoders
        self.final_conv = final_conv
        self.depth = int(depth)
        self.output_channels = int(output_channels)
        self.task = task

    @classmethod
    def from_student(cls, task: str, model: PhisatNet) -> "StudentDecoderExpert":
        return cls(
            upsamplers=deepcopy(model.upsamplers),
            decoders=deepcopy(model.decoders),
            final_conv=deepcopy(model.final_conv),
            depth=model.depth,
            output_channels=model.n_classes,
            task=task,
        )

    def forward(self, bottleneck_features: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        current = bottleneck_features
        for index in range(self.depth):
            current = self.upsamplers[index](current)
            skip = skip_connections[self.depth - 1 - index]
            current = torch.cat([current, skip], dim=1)
            current = self.decoders[index](current)
        return self.final_conv(current)


class MoESwitcher(nn.Module):
    """Small bottleneck classifier that predicts expert activations."""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(in_channels, 32))
        self.in_channels = int(in_channels)
        self.num_experts = int(num_experts)
        self.hidden_dim = hidden
        self.dropout = float(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.in_channels, hidden),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden, self.num_experts),
        )

    def forward(self, bottleneck_features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(bottleneck_features)
        return self.classifier(pooled)


@dataclass
class ExpertMetadata:
    """Serializable metadata for one decoder expert."""

    expert_name: str
    output_channels: int
    training: str
    n_shots: int
    checkpoint_path: Optional[str]
    threshold: float


class MoEStudent(nn.Module):
    """Student encoder with a learned routing head and task-specific decoders."""

    def __init__(
        self,
        encoder: SharedStudentEncoder,
        switcher: MoESwitcher,
        experts: Mapping[str, StudentDecoderExpert],
        *,
        threshold: float = 0.5,
        top_k: Optional[int] = None,
        encoder_source_task: Optional[str] = None,
        expert_metadata: Optional[Mapping[str, Mapping[str, object]]] = None,
        encoder_config: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("MoEStudent requires at least one decoder expert.")

        self.encoder = encoder
        self.switcher = switcher
        self.experts = nn.ModuleDict(experts)
        self.expert_names = list(self.experts.keys())
        self.threshold = float(threshold)
        self.top_k = int(top_k) if top_k is not None else len(self.expert_names)
        self.encoder_source_task = encoder_source_task or self.expert_names[0]
        self.encoder_config = dict(encoder_config or {})
        self.expert_metadata: Dict[str, Dict[str, object]] = {
            name: dict(meta) for name, meta in (expert_metadata or {}).items()
        }

        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.top_k > len(self.expert_names):
            raise ValueError("top_k cannot exceed the number of experts.")

    @property
    def num_experts(self) -> int:
        return len(self.expert_names)

    def route(self, routing_logits: torch.Tensor) -> tuple[torch.Tensor, List[List[str]]]:
        routing_probs = torch.sigmoid(routing_logits)
        top_k = min(self.top_k, self.num_experts)
        active_names: List[List[str]] = []

        for row_logits, row_probs in zip(routing_logits, routing_probs):
            selected = [index for index, prob in enumerate(row_probs.tolist()) if prob >= self.threshold]
            if not selected:
                selected = [int(torch.argmax(row_logits).item())]

            selected = sorted(selected, key=lambda index: float(row_probs[index]), reverse=True)[:top_k]
            active_names.append([self.expert_names[index] for index in selected])

        return routing_probs, active_names

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        bottleneck_features, skip_connections = self.encoder(x)
        routing_logits = self.switcher(bottleneck_features)
        routing_probs, active_experts = self.route(routing_logits)
        active_union = {name for names in active_experts for name in names}

        expert_outputs = {
            name: expert(bottleneck_features, skip_connections)
            for name, expert in self.experts.items()
            if name in active_union
        }

        return {
            "routing_logits": routing_logits,
            "routing_probs": routing_probs,
            "active_experts": active_experts,
            "expert_outputs": expert_outputs,
        }

    def get_bundle_config(self) -> dict:
        """Return the metadata required to reconstruct this model from a bundle."""
        return {
            "threshold": self.threshold,
            "top_k": self.top_k,
            "encoder_source_task": self.encoder_source_task,
            "encoder_config": dict(self.encoder_config),
            "switcher": {
                "in_channels": self.switcher.in_channels,
                "num_experts": self.switcher.num_experts,
                "hidden_dim": self.switcher.hidden_dim,
                "dropout": self.switcher.dropout,
            },
            "experts": {
                name: {
                    "task": expert.task,
                    "output_channels": expert.output_channels,
                    **self.expert_metadata.get(name, {}),
                }
                for name, expert in self.experts.items()
            },
        }


def freeze_module(module: nn.Module) -> None:
    """Disable gradients for all parameters in a module."""
    for parameter in module.parameters():
        parameter.requires_grad = False


def build_moe_student_from_models(
    student_models: Mapping[str, PhisatNet],
    *,
    encoder_source_task: Optional[str] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    switcher_hidden_dim: Optional[int] = None,
    switcher_dropout: float = 0.0,
    expert_metadata: Optional[MutableMapping[str, MutableMapping[str, object]]] = None,
) -> MoEStudent:
    """Assemble a MoE student from already-loaded student checkpoints."""
    signature = validate_student_architecture_compatibility(student_models)
    expert_names = sorted(student_models)
    source_task = encoder_source_task or expert_names[0]
    if source_task not in student_models:
        raise ValueError(f"encoder_source_task {source_task!r} is not part of the selected experts.")

    encoder = SharedStudentEncoder.from_student(student_models[source_task])
    experts = {
        name: StudentDecoderExpert.from_student(name, student_models[name])
        for name in expert_names
    }
    switcher = MoESwitcher(
        in_channels=encoder.out_channels,
        num_experts=len(experts),
        hidden_dim=switcher_hidden_dim,
        dropout=switcher_dropout,
    )

    metadata = {name: dict(meta) for name, meta in (expert_metadata or {}).items()}
    return MoEStudent(
        encoder=encoder,
        switcher=switcher,
        experts=experts,
        threshold=threshold,
        top_k=top_k,
        encoder_source_task=source_task,
        expert_metadata=metadata,
        encoder_config=signature,
    )
