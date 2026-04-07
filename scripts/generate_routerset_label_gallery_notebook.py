from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


MARKDOWN_INTRO = """# Routerset Sample and Label Gallery

This notebook is a focused visual inspection pass over the current raw routerset snapshot under `routerset/`.

It does three things:

- summarizes label and split coverage per dataset
- selects representative rows with meaningful labels
- renders sample previews together with the label metadata stored in the manifest

The goal is to make label inspection easy without mixing it with the heavier raw-audit workflow.
"""


CELL_IMPORTS = """from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from IPython.display import Markdown, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').exists() and (PROJECT_ROOT.parent / 'src').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from hydranet.routerset_raw_audit import (
    FALSE_RGB_CHANNELS,
    RGB_CHANNELS,
    display_view,
    load_raw_routerset_manifest_rows,
    normalize_display,
    resolve_raw_routerset_image_path,
    resolve_raw_routerset_root,
)

ROUTERSET_INPUT = PROJECT_ROOT / 'routerset'
ROUTERSET_ROOT = resolve_raw_routerset_root(ROUTERSET_INPUT)
MANIFEST_ROWS = load_raw_routerset_manifest_rows(ROUTERSET_ROOT)
LABEL_VOCAB_PATH = ROUTERSET_ROOT / 'label_vocab.json'
LABEL_VOCAB = json.loads(LABEL_VOCAB_PATH.read_text(encoding='utf-8')).get('labels', []) if LABEL_VOCAB_PATH.exists() else []

ip = get_ipython()
if ip is not None:
    ip.run_line_magic('matplotlib', 'inline')
set_matplotlib_formats('png')
plt.rcParams['figure.figsize'] = (18, 5)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.max_open_warning'] = 0

print('routerset_root:', ROUTERSET_ROOT)
print('row_count:', len(MANIFEST_ROWS))
print('label_vocab_size:', len(LABEL_VOCAB))
print('label_vocab:', LABEL_VOCAB)
"""


CELL_DATASET_SUMMARY = """display(Markdown('## Dataset label summary'))

dataset_overview = []
for dataset in sorted({str(row['source_dataset']) for row in MANIFEST_ROWS}):
    rows = [row for row in MANIFEST_ROWS if row['source_dataset'] == dataset]
    split_counts = Counter(str(row['source_split']) for row in rows)
    positive_rows = [row for row in rows if row.get('label_names')]
    weak_rows = [row for row in rows if row.get('weak_label_names')]
    empty_rows = [row for row in rows if not row.get('label_names')]
    label_counter = Counter(label for row in rows for label in (row.get('label_names') or []))
    dataset_overview.append(
        {
            'dataset': dataset,
            'row_count': len(rows),
            'split_counts': dict(sorted(split_counts.items())),
            'positive_rows': len(positive_rows),
            'weak_label_rows': len(weak_rows),
            'empty_label_rows': len(empty_rows),
            'top_labels': label_counter.most_common(8),
        }
    )

print(json.dumps(dataset_overview, indent=2))
"""


CELL_SELECTION = """display(Markdown('## Selected gallery rows'))

def row_key(row: dict) -> tuple[str, str, str, int, int, str, str]:
    return (
        str(row.get('source_dataset', '')),
        str(row.get('source_split', '')),
        str(row.get('source_sample_id', '')),
        int(row.get('patch_x') or 0),
        int(row.get('patch_y') or 0),
        str(row.get('patch_width')),
        str(row.get('patch_height')),
    )


def richness(row: dict) -> tuple[float, float, float, str]:
    coverages = row.get('label_coverages') or {}
    return (
        float(len(row.get('label_names') or [])),
        float(len(row.get('weak_label_names') or [])),
        max((float(value) for value in coverages.values()), default=0.0),
        str(row.get('source_sample_id', '')),
    )


selected_rows = []
seen = set()
rows_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
for row in MANIFEST_ROWS:
    rows_by_group[(str(row['source_dataset']), str(row['source_split']))].append(row)

for group_key in sorted(rows_by_group):
    rows = sorted(rows_by_group[group_key], key=row_key)
    positive_rows = [row for row in rows if row.get('label_names')]
    candidate = max(positive_rows or rows, key=richness)
    key = row_key(candidate)
    if key not in seen:
        selected_rows.append(candidate)
        seen.add(key)

    weak_rows = [row for row in rows if row.get('weak_label_names')]
    if weak_rows:
        weak_candidate = max(weak_rows, key=richness)
        key = row_key(weak_candidate)
        if key not in seen:
            selected_rows.append(weak_candidate)
            seen.add(key)

gallery_index = [
    {
        'dataset': row['source_dataset'],
        'split': row['source_split'],
        'sample_id': row['source_sample_id'],
        'labels': row.get('label_names', []),
        'native_labels': row.get('native_label_names', []),
        'weak_labels': row.get('weak_label_names', []),
        'record_status': row.get('record_status'),
        'selection_bucket': row.get('selection_bucket'),
    }
    for row in selected_rows
]

print(json.dumps(gallery_index, indent=2))
"""


CELL_GALLERY = """display(Markdown('## Sample gallery'))

for row in selected_rows:
    image_path = resolve_raw_routerset_image_path(ROUTERSET_ROOT, row)
    array = np.load(image_path, mmap_mode='r')
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
        'record_status': row.get('record_status'),
        'selection_bucket': row.get('selection_bucket'),
        'label_source': row.get('label_source'),
        'label_names': row.get('label_names', []),
        'native_label_names': row.get('native_label_names', []),
        'weak_label_names': row.get('weak_label_names', []),
        'label_coverages': row.get('label_coverages', {}),
        'patch_y': row.get('patch_y'),
        'patch_x': row.get('patch_x'),
        'patch_height': row.get('patch_height'),
        'patch_width': row.get('patch_width'),
        'image_path': str(image_path),
        'shape': list(array.shape),
        'dtype': str(array.dtype),
    }
    print(json.dumps(payload, indent=2))
"""


def build_notebook() -> nbformat.NotebookNode:
    return new_notebook(
        cells=[
            new_markdown_cell(MARKDOWN_INTRO),
            new_code_cell(CELL_IMPORTS),
            new_code_cell(CELL_DATASET_SUMMARY),
            new_code_cell(CELL_SELECTION),
            new_code_cell(CELL_GALLERY),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/routerset_label_gallery.ipynb"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    notebook = build_notebook()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output.open("w", encoding="utf-8"))

    if args.execute:
        client = NotebookClient(notebook, timeout=3600, startup_timeout=240, kernel_name="python3")
        client.execute()
        nbformat.write(notebook, args.output.open("w", encoding="utf-8"))


if __name__ == "__main__":
    main()
