"""HydraNet package entrypoint."""

from .loading import available_student_presets, load_student, load_teacher
from .models.student import DEFAULT_STUDENT_CONFIG, STUDENT_CONFIGS, PhisatNet
from .models.teacher import PhiSatNetDownstream
from .weights import (
    get_available_combinations,
    get_model_weights,
    list_available_weights,
)

__all__ = [
    "DEFAULT_STUDENT_CONFIG",
    "STUDENT_CONFIGS",
    "PhisatNet",
    "PhiSatNetDownstream",
    "available_student_presets",
    "load_student",
    "load_teacher",
    "get_model_weights",
    "list_available_weights",
    "get_available_combinations",
]
