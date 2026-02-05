# HydraNet (PhiSat2)

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

### Student (default: Myriad optimized) - Namely the `HydraNet`, which is the default when calling `load_student()` without arguments.

```python
from hydranet import load_student, available_student_presets

model = load_student()  # uses "myriad_optimized" by default
print(available_student_presets())

custom = load_student(
    preset="myriad_optimized",
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

## Finding the Myriad-Optimized Model

The `myriad_optimized` preset was selected through systematic architecture search (see [notebooks/FindHydraNet.ipynb](notebooks/FindHydraNet.ipynb)):

1. **Search Space**: Evaluated 100+ configurations varying:
   - Base filters: 8, 16, 32
   - Channel multipliers: [1,2,3], [1,2,3,4], etc.
   - Depth: 2, 3, 4 blocks
   - Patch sizes: 28×28 to 224×224

2. **Optimization Criteria**:
   - Total inference time across 4096×4096 tile
   - Parameter count vs accuracy trade-off
   - Hardware constraints (Myriad X VPU)

3. **Result**: 16 base filters, [1,2,3,4] multipliers, depth=3
   - ~60K parameters
   - Optimal latency/accuracy balance
   - Fits Myriad X memory and compute budget

The analysis used 3D visualization (parameters × depth × latency) to identify the Pareto-optimal configuration.

## Repo Layout

- `src/hydranet/`: package entrypoint and public loading helpers
- `src/hydranet/models/`: student + teacher model implementations
- `scripts/export_onnx.py`: ONNX export script (writes to `outputs/onnx/`)
- `examples/`: quick usage snippets
- `notebooks/`: interactive tutorials and model selection analysis
