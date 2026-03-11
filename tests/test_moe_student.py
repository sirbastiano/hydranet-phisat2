from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from hydranet import load_student_moe_bundle, save_student_moe_bundle
from hydranet.models.moe_student import build_moe_student_from_models
from hydranet.models.student import create_phisatnet


class TestMoEStudent(unittest.TestCase):
    def test_architecture_mismatch_raises(self) -> None:
        models = {
            "fire": create_phisatnet("checkpoint", n_classes=1),
            "worldfloods": create_phisatnet("checkpoint", n_classes=2, base_filters=32),
        }
        with self.assertRaisesRegex(ValueError, "Student architecture mismatch"):
            build_moe_student_from_models(models)

    def test_routing_threshold_top_k_and_fallback(self) -> None:
        models = {
            "anomaly_detection": create_phisatnet("checkpoint", n_classes=1),
            "fire": create_phisatnet("checkpoint", n_classes=2),
            "worldfloods": create_phisatnet("checkpoint", n_classes=3),
        }
        model = build_moe_student_from_models(models, threshold=0.7, top_k=2)

        logits = torch.tensor(
            [
                [5.0, 4.0, -4.0],
                [-4.0, -3.0, -2.0],
            ]
        )
        probs, active = model.route(logits)

        self.assertEqual(active[0], ["anomaly_detection", "fire"])
        self.assertEqual(active[1], ["worldfloods"])
        self.assertGreater(probs[0, 0].item(), 0.99)

    def test_bundle_round_trip_preserves_routing(self) -> None:
        torch.manual_seed(0)
        models = {
            "anomaly_detection": create_phisatnet("checkpoint", n_classes=1),
            "fire": create_phisatnet("checkpoint", n_classes=2),
            "worldfloods": create_phisatnet("checkpoint", n_classes=3),
        }
        model = build_moe_student_from_models(models, threshold=0.2, top_k=2)
        batch = torch.randn(2, 8, 32, 32)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "moe_bundle.pt"
            save_student_moe_bundle(model, bundle_path)
            restored = load_student_moe_bundle(bundle_path)

            expected = model(batch)
            actual = restored(batch)

        self.assertTrue(torch.allclose(expected["routing_logits"], actual["routing_logits"]))
        self.assertEqual(expected["active_experts"], actual["active_experts"])
        self.assertEqual(sorted(expected["expert_outputs"]), sorted(actual["expert_outputs"]))

    def test_forward_only_returns_active_experts(self) -> None:
        torch.manual_seed(1)
        models = {
            "anomaly_detection": create_phisatnet("checkpoint", n_classes=1),
            "fire": create_phisatnet("checkpoint", n_classes=2),
            "worldfloods": create_phisatnet("checkpoint", n_classes=3),
        }
        model = build_moe_student_from_models(models, threshold=0.99, top_k=1)
        batch = torch.randn(1, 8, 32, 32)
        outputs = model(batch)

        self.assertEqual(len(outputs["active_experts"][0]), 1)
        self.assertEqual(sorted(outputs["expert_outputs"]), outputs["active_experts"][0])


if __name__ == "__main__":
    unittest.main()
