# HydraNet (PhiSat2)

Clean Python package layout for student/teacher models, with explicit loading helpers.

## Installation

```
uv sync
```

Or install in editable mode:

```
pip install -e .
```

If you prefer not to install, run with `PYTHONPATH=src`.

## Loading Models

### Student (default: Myriad optimized)

```python
from hydranet import load_student, available_student_presets

model = load_student()  # uses "myriad_optimized" by default
print(available_student_presets())

custom = load_student(
    preset="small",
    n_classes=4,  # overrides allowed
)
```

### Teacher (downstream)

```python
from hydranet import load_teacher

teacher = load_teacher(
    task="segmentation",
    pretrained_path=None,
    input_dim=8,
    output_dim=3,
    depths=[2, 2, 6, 2],
    dims=[64, 128, 256, 512],
    img_size=224,
)
```

## Repo Layout

- `src/hydranet/`: package entrypoint and public loading helpers
- `src/hydranet/models/`: student + teacher model implementations
- `scripts/export_onnx.py`: ONNX export script (writes to `outputs/onnx/`)
- `examples/`: quick usage snippets
