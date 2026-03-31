from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


MARKDOWN_INTRO = """# Historical Float Student Checkpoint Comparison

This notebook records the pre-prune audit that justified removing weak
float-domain student checkpoint variants from the default HydraNet catalog.

Current catalog contract:

- `anomaly_detection`, `burned_area`, `fire`, and `worldfloods` now keep only
  the strong `finetuning` `5000-shot` student checkpoint
- the `linear_probing` and lower-shot student variants shown below are
  historical audit data, not part of the current default workflow

Important scope:

- this is **not** a comparison of different model families from the public
  Phi2FM repo
- HydraNet currently ships one strong default student checkpoint per float
  expert; the comparisons below explain why the weaker variants were pruned

The notebook uses the corrected float export at
`outputs/routerset/fix30March_floatminmax_selected_anomalyfix/` and preserves
representative `train` and `validation` patches for the historical audit.
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
from IPython.display import Markdown, display

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists() and (PROJECT_ROOT.parent / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from hydranet import load_student
from hydranet.moe_training import _reduce_expert_output_to_routing_score

# Historical audit notebook: current float-student defaults keep only FT-5000.

DATASET_ROOT = PROJECT_ROOT / 'outputs' / 'routerset' / 'fix30March_floatminmax_selected_anomalyfix'
MANIFEST_PATH = DATASET_ROOT / 'manifest_256.jsonl'
REPORT_PATH = DATASET_ROOT / 'student_checkpoint_comparison_summary.json'

EXPERTS = ['anomaly_detection', 'burned_area', 'fire', 'worldfloods']
VARIANTS = [
    {'label': 'FT-50', 'training': 'finetuning', 'n_shots': 50},
    {'label': 'LP-50', 'training': 'linear_probing', 'n_shots': 50},
    {'label': 'FT-5000', 'training': 'finetuning', 'n_shots': 5000},
    {'label': 'LP-5000', 'training': 'linear_probing', 'n_shots': 5000},
]
BASELINE_LABEL = 'FT-5000'
DEVICE = 'cpu'

torch.set_grad_enabled(False)
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.max_open_warning'] = 0

print('project_root:', PROJECT_ROOT)
print('dataset_root:', DATASET_ROOT)
print('manifest_exists:', MANIFEST_PATH.exists())
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
    for variant in VARIANTS:
        key = (expert, variant['training'], variant['n_shots'])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model = load_student(
                task=expert,
                training=variant['training'],
                n_shots=variant['n_shots'],
                auto_load_weights=True,
            )
        model.eval()
        model.to(DEVICE)
        models[key] = model
        model_logs[key] = [line for line in buffer.getvalue().splitlines() if line.strip()]

for expert in EXPERTS:
    print(f'## {expert}')
    for variant in VARIANTS:
        key = (expert, variant['training'], variant['n_shots'])
        print(f\"{variant['label']}: {model_logs[key][-1] if model_logs[key] else 'loaded'}\")
    print()
"""


CELL_RENDER = """report = {'dataset_root': str(DATASET_ROOT), 'baseline': BASELINE_LABEL, 'per_patch': []}

for row in selected_rows:
    expert = row['source_dataset']
    split = row.get('moe_split', row['source_split'])
    image = np.load(resolve_path(row['materialized_image_path'])).astype(np.float32, copy=False)

    outputs = {}
    for variant in VARIANTS:
        key = (expert, variant['training'], variant['n_shots'])
        outputs[variant['label']] = infer_patch(models[key], image)

    baseline = outputs[BASELINE_LABEL]
    table_rows = []
    for variant in VARIANTS:
        result = outputs[variant['label']]
        disagree = float(np.mean(result['pred'] != baseline['pred']))
        conf_delta = float(np.mean(result['conf'] - baseline['conf']))
        table_rows.append(
            {
                'variant': variant['label'],
                'score': f\"{result['score']:.4f}\",
                'mean_conf': f\"{float(np.mean(result['conf'])):.4f}\",
                'disagree_vs_ft5000': f\"{disagree:.4f}\",
                'conf_delta_vs_ft5000': f\"{conf_delta:.4f}\",
                'histogram': result['histogram'],
            }
        )

    report['per_patch'].append(
        {
            'expert': expert,
            'split': split,
            'sample_id': row['source_sample_id'],
            'metrics': table_rows,
        }
    )

    display(Markdown(f\"## {expert} | {split} | `{row['source_sample_id']}`\"))

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    fig.suptitle(f\"{expert} | {split} | sample={row['source_sample_id']}\", fontsize=14)

    axes[0].imshow(normalize_display(image, FALSE_RGB_CHANNELS))
    axes[0].set_title('False RGB')

    for index, variant in enumerate(VARIANTS, start=1):
        result = outputs[variant['label']]
        axes[index].imshow(result['pred'], cmap='tab20')
        axes[index].set_title(f\"{variant['label']}\\nscore={result['score']:.4f}\")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()

    display(
        Markdown(
            markdown_table(
                table_rows,
                [
                    ('variant', 'variant'),
                    ('score', 'routing score'),
                    ('mean_conf', 'mean confidence'),
                    ('disagree_vs_ft5000', 'disagree vs FT-5000'),
                    ('conf_delta_vs_ft5000', 'conf delta vs FT-5000'),
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
        default=Path("notebooks/float_student_checkpoint_comparison.ipynb"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output.open("w", encoding="utf-8"))

    if args.execute:
        client = NotebookClient(notebook, timeout=3600, kernel_name="python3")
        client.execute(cwd=str(args.cwd.resolve()))
        nbformat.write(notebook, args.output.open("w", encoding="utf-8"))

    print(args.output)


if __name__ == "__main__":
    main()
