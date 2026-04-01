from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hydranet.routerset_raw_audit import (
    STATUS_ERROR,
    STATUS_OK,
    audit_raw_routerset_dataset,
    resolve_raw_routerset_image_path,
)


class TestRoutersetRawAudit(unittest.TestCase):
    def test_resolve_raw_routerset_image_path_uses_swapped_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'routerset' / 'multilabel_dataset'
            image_dir = root / 'images' / 'worldfloods' / 'train'
            image_dir.mkdir(parents=True)
            (root / 'manifest.jsonl').write_text('', encoding='utf-8')
            swapped = image_dir / '0000001_128_0_256_256.npy'
            np.save(swapped, np.zeros((8, 256, 256), dtype=np.float32))
            row = {
                'source_dataset': 'worldfloods',
                'source_split': 'train',
                'source_sample_id': '0000001',
                'patch_x': 0,
                'patch_y': 128,
                'patch_width': 256,
                'patch_height': 256,
            }

            resolved = resolve_raw_routerset_image_path(root.parent, row)

        self.assertEqual(resolved.name, swapped.name)

    def test_audit_raw_routerset_dataset_writes_summary_and_detects_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / 'routerset' / 'multilabel_dataset'
            (root / 'images' / 'burned_area' / 'train').mkdir(parents=True)
            (root / 'images' / 'fire' / 'validation').mkdir(parents=True)
            burned_path = root / 'images' / 'burned_area' / 'train' / '0000002_0_0_256_256.npy'
            np.save(burned_path, np.ones((7, 256, 256), dtype=np.float32))
            rows = [
                {
                    'source_dataset': 'burned_area',
                    'source_split': 'train',
                    'source_sample_id': '0000002',
                    'record_status': 'positive',
                    'selection_bucket': 'burned_area',
                    'label_names': ['burned_area'],
                    'patch_x': 0,
                    'patch_y': 0,
                    'patch_width': 256,
                    'patch_height': 256,
                },
                {
                    'source_dataset': 'fire',
                    'source_split': 'validation',
                    'source_sample_id': '0000005',
                    'record_status': 'positive',
                    'selection_bucket': 'active_fire',
                    'label_names': ['active_fire'],
                    'patch_x': 0,
                    'patch_y': 0,
                    'patch_width': None,
                    'patch_height': None,
                },
            ]
            manifest = root / 'manifest.jsonl'
            manifest.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')

            summary = audit_raw_routerset_dataset(root.parent, output_dir=root / 'audit_raw')
            audit_rows = [json.loads(line) for line in (root / 'audit_raw' / 'file_audit.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]

        self.assertEqual(summary['row_count'], 2)
        self.assertEqual(summary['per_dataset']['burned_area']['row_count'], 1)
        self.assertEqual(summary['per_dataset']['fire']['row_count'], 1)
        self.assertEqual(summary['status_counts'][STATUS_OK], 1)
        self.assertEqual(summary['status_counts'][STATUS_ERROR], 1)
        self.assertIn('missing_file', summary['issue_counts'])
        self.assertTrue((root / 'audit_raw' / 'summary.json').exists())
        self.assertEqual(audit_rows[0]['status'], STATUS_OK)
        self.assertEqual(audit_rows[1]['status'], STATUS_ERROR)
