from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


MARKDOWN_INTRO = """# Routerset Float Student Inference

This notebook runs the `student` checkpoints for the float-domain experts:

- `anomaly_detection`
- `burned_area`
- `fire`
- `worldfloods`

It uses the corrected materialized dataset at `outputs/routerset/fix31March_floatminmax_centerpad_firefix_selected/`.

For each expert, the notebook shows representative train and validation patches and compares:

- the **fixed** Phi2FM-aligned float preprocessing now used in HydraNet
- the **old** Hydra float path that only adapted channels and skipped per-image min-max normalization

The goal is visual inspection of how the student decoders respond to the corrected input contract.
"""


CELL_IMPORTS = """from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import get_ipython
from IPython.display import Markdown, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists() and (PROJECT_ROOT.parent / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from hydranet import load_student
from hydranet.moe_training import crop_or_pad_routerset_tensor, _reduce_expert_output_to_routing_score

DATASET_ROOT = PROJECT_ROOT / 'outputs' / 'routerset' / 'fix31March_floatminmax_centerpad_firefix_selected'
MANIFEST_PATH = DATASET_ROOT / 'manifest_256.jsonl'
FAULT_REPORT_PATH = DATASET_ROOT / 'fault_report.json'

EXPERTS = ['anomaly_detection', 'burned_area', 'fire', 'worldfloods']
TARGET_SIZE = 256
TARGET_CHANNELS = 8
DEVICE = 'cpu'

torch.set_grad_enabled(False)
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('png')
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.max_open_warning'] = 0

print('project_root:', PROJECT_ROOT)
print('dataset_root:', DATASET_ROOT)
print('manifest_exists:', MANIFEST_PATH.exists())
print('fault_report_exists:', FAULT_REPORT_PATH.exists())
"""


CELL_SELECT_ROWS = """def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_manifest_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


rows = load_manifest_rows(MANIFEST_PATH)
fault_report = json.loads(FAULT_REPORT_PATH.read_text(encoding='utf-8'))

rows_by_expert_split: dict[tuple[str, str], list[dict]] = {}
for row in rows:
    expert = row['source_dataset']
    split = row.get('moe_split', row['source_split'])
    if expert not in EXPERTS:
        continue
    rows_by_expert_split.setdefault((expert, split), []).append(row)

selected_rows: list[dict] = []
for expert in EXPERTS:
    for split in ('train', 'validation'):
        group = sorted(
            rows_by_expert_split[(expert, split)],
            key=lambda row: (str(row['source_sample_id']), str(row['materialized_image_path'])),
        )
        selected_rows.append(group[0])

summary_rows = [
    {
        'expert': row['source_dataset'],
        'split': row.get('moe_split', row['source_split']),
        'sample_id': row['source_sample_id'],
        'labels': ', '.join(row.get('label_names') or []),
        'materialized_image_path': row['materialized_image_path'],
        'materialized_from': row['materialized_from'],
    }
    for row in selected_rows
]

print('selected rows:')
print(json.dumps(summary_rows, indent=2))
print('quarantined faults by expert:', json.dumps(fault_report.get('excluded_rows_by_expert', {}), indent=2, sort_keys=True))
print('training_ready:', fault_report.get('training_ready'))
"""


CELL_HELPERS = """RGB_CHANNELS = (2, 1, 0)       # B04, B03, B02
FALSE_RGB_CHANNELS = (4, 2, 1) # B08, B04, B03


def old_float_channel_adapter(array: np.ndarray, target_channels: int = TARGET_CHANNELS) -> np.ndarray:
    current = np.array(array, dtype=np.float32, copy=True)
    current = np.nan_to_num(current, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    if current.ndim != 3:
        raise ValueError(f'Expected CHW array, got {current.shape}')
    channels, height, width = current.shape
    if channels > target_channels:
        current = current[:target_channels]
    elif channels < target_channels:
        pad = np.zeros((target_channels - channels, height, width), dtype=np.float32)
        current = np.concatenate([current, pad], axis=0)
    return current.astype(np.float32, copy=False)


def build_old_patch(row: dict) -> np.ndarray:
    source_path = resolve_path(row['materialized_from'])
    source = np.load(source_path, mmap_mode='r')
    current = old_float_channel_adapter(source)
    if row['source_dataset'] == 'anomaly_detection':
        patch_y = int(row['patch_y'])
        patch_x = int(row['patch_x'])
        current = current[:, patch_y:patch_y + TARGET_SIZE, patch_x:patch_x + TARGET_SIZE]
    tensor = crop_or_pad_routerset_tensor(torch.from_numpy(np.array(current, copy=True)), target_size=TARGET_SIZE)
    return tensor.numpy().astype(np.float32, copy=False)


def load_fixed_patch(row: dict) -> np.ndarray:
    path = resolve_path(row['materialized_image_path'])
    return np.load(path).astype(np.float32, copy=False)


def normalize_display(chw: np.ndarray, channels: tuple[int, int, int]) -> np.ndarray:
    rgb = np.stack([chw[index] for index in channels], axis=-1).astype(np.float32)
    lo = np.percentile(rgb, 2.0)
    hi = np.percentile(rgb, 98.0)
    if hi <= lo:
        return np.clip(rgb, 0.0, 1.0)
    rgb = (rgb - lo) / (hi - lo)
    return np.clip(rgb, 0.0, 1.0)


def infer_patch(model: torch.nn.Module, chw: np.ndarray) -> dict:
    x = torch.from_numpy(np.array(chw, copy=True)).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)[0].detach().cpu().numpy()
    conf = probs.max(dim=1).values[0].detach().cpu().numpy()
    score = _reduce_expert_output_to_routing_score(logits.detach().cpu())
    values, counts = np.unique(pred, return_counts=True)
    histogram = {int(v): int(c) for v, c in zip(values, counts)}
    return {
        'logits': logits.detach().cpu(),
        'pred': pred,
        'conf': conf,
        'score': float(score),
        'histogram': histogram,
    }


def render_sample(row: dict, model: torch.nn.Module) -> None:
    fixed = load_fixed_patch(row)
    old = build_old_patch(row)

    fixed_out = infer_patch(model, fixed)
    old_out = infer_patch(model, old)

    disagreement = (fixed_out['pred'] != old_out['pred']).astype(np.float32)
    conf_delta = fixed_out['conf'] - old_out['conf']

    split = row.get('moe_split', row['source_split'])
    title = f"{row['source_dataset']} | {split} | sample={row['source_sample_id']}"
    weak_labels = ', '.join(row.get('label_names') or []) or '<none>'

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(title, fontsize=14)

    axes[0, 0].imshow(normalize_display(fixed, RGB_CHANNELS))
    axes[0, 0].set_title('Fixed RGB')
    axes[0, 1].imshow(normalize_display(fixed, FALSE_RGB_CHANNELS))
    axes[0, 1].set_title('Fixed False RGB')
    axes[0, 2].imshow(fixed_out['pred'], cmap='tab20')
    axes[0, 2].set_title(f"Fixed Pred | score={fixed_out['score']:.4f}")
    axes[0, 3].imshow(fixed_out['conf'], cmap='viridis')
    axes[0, 3].set_title('Fixed Max Confidence')

    axes[1, 0].imshow(old_out['pred'], cmap='tab20')
    axes[1, 0].set_title(f"Old Pred | score={old_out['score']:.4f}")
    axes[1, 1].imshow(old_out['conf'], cmap='viridis')
    axes[1, 1].set_title('Old Max Confidence')
    axes[1, 2].imshow(disagreement, cmap='magma', vmin=0.0, vmax=1.0)
    axes[1, 2].set_title(f"Class Disagreement | frac={disagreement.mean():.4f}")
    im = axes[1, 3].imshow(conf_delta, cmap='coolwarm')
    axes[1, 3].set_title('Fixed - Old Confidence')
    fig.colorbar(im, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    info = '\\n'.join([
        f"weak labels: {weak_labels}",
        f"selection bucket: {row.get('selection_bucket')}",
        f"materialized from: {row.get('materialized_from')}",
        f"fixed histogram: {fixed_out['histogram']}",
        f"old histogram: {old_out['histogram']}",
    ])
    fig.text(0.5, 0.01, info, ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=(0, 0.08, 1, 0.96))
    plt.show()
"""


CELL_LOAD_MODELS = """models = {}
model_logs = {}
for expert in EXPERTS:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = load_student(task=expert, training='finetuning', n_shots=5000, auto_load_weights=True)
    model.eval()
    model.to(DEVICE)
    models[expert] = model
    model_logs[expert] = [line for line in buffer.getvalue().splitlines() if line.strip()]

for expert in EXPERTS:
    print(f'## {expert}')
    for line in model_logs[expert][-4:]:
        print(line)
    print()
"""


CELL_RENDER = """for row in selected_rows:
    display(Markdown(f"## {row['source_dataset']} | {row.get('moe_split', row['source_split'])} | `{row['source_sample_id']}`"))
    render_sample(row, models[row['source_dataset']])
"""


CELL_FAULTS = """display(Markdown('## Quarantined Fault Summary'))
print(json.dumps(fault_report, indent=2, sort_keys=True))
"""


def build_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(MARKDOWN_INTRO),
            new_code_cell(CELL_IMPORTS),
            new_code_cell(CELL_SELECT_ROWS),
            new_code_cell(CELL_HELPERS),
            new_code_cell(CELL_LOAD_MODELS),
            new_code_cell(CELL_RENDER),
            new_code_cell(CELL_FAULTS),
        ],
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/routerset_float_student_inference.ipynb"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output.open("w", encoding="utf-8"))

    if args.execute:
        client = NotebookClient(notebook, timeout=1800, kernel_name="python3")
        client.execute(cwd=str(args.cwd.resolve()))
        nbformat.write(notebook, args.output.open("w", encoding="utf-8"))

    print(args.output)


if __name__ == "__main__":
    main()
