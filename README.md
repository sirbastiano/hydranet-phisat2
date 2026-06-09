# HydraNet

Python package for loading models.

<p align="center">
    <img src="logo.png" alt="HydraNet Logo" width="400"/>
</p>

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

### Student (default: checkpoint-compatible) - Namely the `HydraNet`, which is the default when calling `load_student()` without arguments.

```python
from hydranet import load_student, available_student_presets

model = load_student()  # uses "checkpoint" by default
print(available_student_presets())

custom = load_student(
    preset="checkpoint",
    n_classes=4,  # overrides allowed
)
```

### Teacher (downstream) - Namely the `PhiSatNet`, which is the default when calling `load_teacher()` without arguments.

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

## Quick Start with Pre-trained Weights

Load pre-trained models from HuggingFace and split them into components (see [notebooks/start_here.ipynb](notebooks/start_here.ipynb)):

```python
import hydranet

# Load model with automatic weight download from HuggingFace
model = hydranet.load_student(
    preset='checkpoint',      # Matches HF checkpoint architecture
    task='burned_area',
    n_shots=5000,
    training='linear_probing',
    auto_load_weights=True,
    weights_dir='../weights',
    strict=False             # Allows task-specific head mismatches
)

# Split the model into ENCODER, BOTTLENECK, and DECODER components
components = hydranet.split_model(model)

# Access individual components
encoder = components.encoder
bottleneck = components.bottleneck
decoder = components.decoder

print(components)  # Display component summary
```

**Available:** 139 pre-trained model configurations from `sirbastiano/hydranet-phisat2` on HuggingFace.

## Repo Layout

- `src/hydranet/`: package entrypoint and public loading helpers
- `src/hydranet/models/`: student + teacher model implementations
- `scripts/export_onnx.py`: ONNX export script (writes to `outputs/onnx/`)
- `examples/`: quick usage snippets
- `notebooks/`: interactive tutorials and model selection analysis
