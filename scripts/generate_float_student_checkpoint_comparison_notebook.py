from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


MARKDOWN_INTRO = """# Float Student Default Checkpoints

This notebook inspects the current strong float-student defaults shipped by
HydraNet for:

- `anomaly_detection`
- `burned_area`
- `fire`
- `worldfloods`

Current catalog contract:

- each of these experts keeps only the strong `finetuning` `5000-shot`
  student checkpoint
- weaker `linear_probing` and lower-shot student variants are no longer part
  of the default workflow

The notebook uses the corrected float export at
`outputs/routerset/fix30March_floatminmax_selected_anomalyfix/` and renders
representative `train` and `validation` patches for each expert.
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
from hydranet.moe_training import _reduce_expert_output_to_routing_score

DATASET_ROOT = PROJECT_ROOT / 'outputs' / 'routerset' / 'fix30March_floatminmax_selected_anomalyfix'
MANIFEST_PATH = DATASET_ROOT / 'manifest_256.jsonl'
REPORT_PATH = DATASET_ROOT / 'student_default_checkpoint_summary.json'

EXPERTS = ['anomaly_detection', 'burned_area', 'fire', 'worldfloods']
VARIANT = {'label': 'FT-5000', 'training': 'finetuning', 'n_shots': 5000}
DEVICE = 'cpu'

torch.set_grad_enabled(False)
ip = get_ipython()
if ip is not None:
    ip.run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('png')
plt.rcParams['figure.figsize'] = (14, 4.5)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.max_open_warning'] = 0

print('project_root:', PROJECT_ROOT)
print('dataset_root:', DATASET_ROOT)
print('manifest_exists:', MANIFEST_PATH.exists())
print('variant:', VARIANT)
"""


CELL_HELPERS = """FALSE_RGB_CHANNELS = (4, 2, 1)  # B08, B04, B03


def load_manifest_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        'pred': pred,
        'conf': conf,
        'score': float(score),
        'histogram': histogram,
        'mean_conf': float(np.mean(conf)),
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = '| ' + ' | '.join(label for _, label in columns) + ' |'
    divider = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    body = []
    for row in rows:
        body.append('| ' + ' | '.join(str(row[key]) for key, _ in columns) + ' |')
    return '\\n'.join([header, divider, *body])
"""


CELL_SELECT_ROWS = """rows = load_manifest_rows(MANIFEST_PATH)
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

print(json.dumps([
    {
        'expert': row['source_dataset'],
        'split': row.get('moe_split', row['source_split']),
        'sample_id': row['source_sample_id'],
        'labels': row.get('label_names') or [],
        'materialized_image_path': row['materialized_image_path'],
    }
    for row in selected_rows
], indent=2))
"""


CELL_LOAD_MODELS = """models = {}
model_logs = {}
for expert in EXPERTS:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model = load_student(
            task=expert,
            training=VARIANT['training'],
            n_shots=VARIANT['n_shots'],
            auto_load_weights=True,
        )
    model.eval()
    model.to(DEVICE)
    models[expert] = model
    model_logs[expert] = [line for line in buffer.getvalue().splitlines() if line.strip()]

for expert in EXPERTS:
    print(f"{expert}: {model_logs[expert][-1] if model_logs[expert] else 'loaded'}")
"""


CELL_RENDER = """report = {'dataset_root': str(DATASET_ROOT), 'variant': VARIANT, 'per_patch': []}

for row in selected_rows:
    expert = row['source_dataset']
    split = row.get('moe_split', row['source_split'])
    image = np.load(resolve_path(row['materialized_image_path'])).astype(np.float32, copy=False)
    result = infer_patch(models[expert], image)

    report['per_patch'].append(
        {
            'expert': expert,
            'split': split,
            'sample_id': row['source_sample_id'],
            'score': result['score'],
            'mean_conf': result['mean_conf'],
            'histogram': result['histogram'],
        }
    )

    display(Markdown(f"## {expert} | {split} | `{row['source_sample_id']}`"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"{expert} | {split} | sample={row['source_sample_id']}", fontsize=14)

    axes[0].imshow(normalize_display(image, FALSE_RGB_CHANNELS))
    axes[0].set_title('False RGB')

    axes[1].imshow(result['pred'], cmap='tab20')
    axes[1].set_title(f"{VARIANT['label']} prediction\\nscore={result['score']:.4f}")

    im = axes[2].imshow(result['conf'], cmap='viridis', vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Confidence\\nmean={result['mean_conf']:.4f}")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()

    display(
        Markdown(
            markdown_table(
                [
                    {
                        'variant': VARIANT['label'],
                        'score': f"{result['score']:.4f}",
                        'mean_conf': f"{result['mean_conf']:.4f}",
                        'histogram': result['histogram'],
                    }
                ],
                [
                    ('variant', 'variant'),
                    ('score', 'routing score'),
                    ('mean_conf', 'mean confidence'),
                    ('histogram', 'pred histogram'),
                ],
            )
        )
    )

REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('summary_report:', REPORT_PATH)
"""


def build_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(MARKDOWN_INTRO),
            new_code_cell(CELL_IMPORTS),
            new_code_cell(CELL_HELPERS),
            new_code_cell(CELL_SELECT_ROWS),
            new_code_cell(CELL_LOAD_MODELS),
            new_code_cell(CELL_RENDER),
        ],
        metadata={
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.13',
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('notebooks/float_student_checkpoint_comparison.ipynb'),
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output.open('w', encoding='utf-8'))

    if args.execute:
        client = NotebookClient(notebook, timeout=3600, kernel_name='python3')
        client.execute(cwd=str(args.cwd.resolve()))
        nbformat.write(notebook, args.output.open('w', encoding='utf-8'))

    print(args.output)


if __name__ == '__main__':
    main()
