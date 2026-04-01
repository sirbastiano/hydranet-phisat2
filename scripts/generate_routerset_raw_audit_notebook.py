from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

MARKDOWN_INTRO = """# Raw Routerset Audit

This notebook audits the rebuilt raw routerset snapshot under `routerset/multilabel_dataset`.

It does three things:

- runs a per-file integrity audit against the current raw manifest
- renders summary plots and dataset-level stats
- shows representative train and validation samples for each dataset with RGB and false-RGB views

The goal is to inspect the current routerset artifact directly, not one of the older materialized exports.
"""

CELL_IMPORTS = """from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import Image, Markdown, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists() and (PROJECT_ROOT.parent / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from hydranet.routerset_raw_audit import (
    RGB_CHANNELS,
    FALSE_RGB_CHANNELS,
    audit_raw_routerset_dataset,
    display_view,
    normalize_display,
)

ROUTERSET_DIR = PROJECT_ROOT / 'routerset'
DATASET_ROOT = ROUTERSET_DIR
AUDIT_DIR = DATASET_ROOT / 'audit_raw'

ip = get_ipython()
if ip is not None:
    ip.run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('png')
plt.rcParams['figure.figsize'] = (18, 5)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.max_open_warning'] = 0

if (AUDIT_DIR / 'summary.json').exists() and (AUDIT_DIR / 'sample_rows.json').exists():
    summary = json.loads((AUDIT_DIR / 'summary.json').read_text(encoding='utf-8'))
else:
    summary = audit_raw_routerset_dataset(DATASET_ROOT, output_dir=AUDIT_DIR)
sample_rows = json.loads((AUDIT_DIR / 'sample_rows.json').read_text(encoding='utf-8'))['rows']

print('dataset_root:', DATASET_ROOT)
print('audit_dir:', AUDIT_DIR)
print('row_count:', summary['row_count'])
print('status_counts:', summary['status_counts'])
"""

CELL_SUMMARY = """display(Markdown('## Summary'))
print(json.dumps(summary, indent=2))
"""

CELL_PLOTS = """display(Markdown('## Audit plots'))
for name in ['dataset_split_counts.png', 'status_counts.png', 'zero_fraction_boxplot.png']:
    path = AUDIT_DIR / name
    print(path.name)
    display(Image(filename=str(path)))
"""

CELL_GALLERY = """display(Markdown('## Sample gallery'))

for row in sample_rows:
    array = np.load(Path(row['image_path']), mmap_mode='r')
    chw = display_view(array)
    rgb = normalize_display(chw, channels=RGB_CHANNELS)
    false_rgb = normalize_display(chw, channels=FALSE_RGB_CHANNELS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB')
    axes[1].imshow(false_rgb)
    axes[1].set_title('False RGB')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    title = f"{row['source_dataset']} :: {row['source_split']} :: {row['source_sample_id']}"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    payload = {
        'dataset': row['source_dataset'],
        'split': row['source_split'],
        'sample_id': row['source_sample_id'],
        'image_path': row['image_path'],
        'shape': row.get('shape'),
        'dtype': row.get('dtype'),
        'layout': row.get('layout'),
        'min': row.get('min'),
        'max': row.get('max'),
        'mean': row.get('mean'),
        'zero_fraction': row.get('zero_fraction'),
        'stats_sampled': row.get('stats_sampled'),
        'label_names': row.get('label_names', []),
        'status': row.get('status'),
        'issue_codes': row.get('issue_codes', []),
        'note_codes': row.get('note_codes', []),
    }
    print(json.dumps(payload, indent=2))
"""


def build_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(MARKDOWN_INTRO),
            new_code_cell(CELL_IMPORTS),
            new_code_cell(CELL_SUMMARY),
            new_code_cell(CELL_PLOTS),
            new_code_cell(CELL_GALLERY),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('notebooks/routerset_raw_audit.ipynb'),
    )
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output.open('w', encoding='utf-8'))

    if args.execute:
        client = NotebookClient(notebook, timeout=3600, startup_timeout=240, kernel_name='python3')
        client.execute()
        nbformat.write(notebook, args.output.open('w', encoding='utf-8'))


if __name__ == '__main__':
    main()
