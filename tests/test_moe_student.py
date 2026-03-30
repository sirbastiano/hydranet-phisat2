from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hydranet import PhiSatNetDownstream, load_student_moe_bundle, save_student_moe_bundle
from hydranet.loading import _load_student_catalog_rows, _resolve_default_moe_experts
from hydranet.models.moe_student import build_moe_student_from_models
from hydranet.models.student import create_phisatnet


class TestMoEStudent(unittest.TestCase):
    def test_float_student_catalog_only_keeps_strong_checkpoints(self) -> None:
        rows = _load_student_catalog_rows()
        expected = {
            "anomaly_detection": ("finetuning", "5000.0", "20260108"),
            "burned_area": ("finetuning", "5000.0", "20251216"),
            "fire": ("finetuning", "5000.0", "20251217"),
            "worldfloods": ("finetuning", "5000.0", "20251212"),
        }

        for task, keep in expected.items():
            task_rows = [row for row in rows if row["task"] == task]
            self.assertEqual(len(task_rows), 1)
            self.assertEqual(
                (task_rows[0]["training"], task_rows[0]["n_shots"], task_rows[0]["datetime"]),
                keep,
            )

    def test_default_routerset_experts_still_resolve_from_catalog(self) -> None:
        tasks = _resolve_default_moe_experts(
            allowed_tasks=("anomaly_detection", "burned_area", "fire", "lc", "worldfloods"),
            training="finetuning",
            n_shots=5000,
        )
        self.assertEqual(tasks, ["anomaly_detection", "burned_area", "fire", "lc", "worldfloods"])

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

    def test_load_student_moe_bundle_uses_safe_torch_load(self) -> None:
        torch.manual_seed(2)
        models = {
            "anomaly_detection": create_phisatnet("checkpoint", n_classes=1),
            "fire": create_phisatnet("checkpoint", n_classes=2),
        }
        model = build_moe_student_from_models(models, threshold=0.3, top_k=1)
        bundle_path = Path("/tmp/unused.pt")

        payload = {
            "bundle_type": "hydranet_student_moe",
            "bundle_version": 1,
            "model_config": model.get_bundle_config(),
            "state_dict": model.state_dict(),
            "metadata": {},
        }

        with patch("torch.load") as mocked_load:
            mocked_load.return_value = payload
            restored = load_student_moe_bundle(bundle_path)

        mocked_load.assert_called_once_with(bundle_path, map_location="cpu", weights_only=True)
        self.assertEqual(type(restored).__name__, "MoEStudent")

    def test_load_student_moe_bundle_requires_known_keys(self) -> None:
        malformed_bundle = {"bundle_type": "hydranet_student_moe"}
        with patch("torch.load") as mocked_load:
            mocked_load.return_value = malformed_bundle
            with self.assertRaisesRegex(ValueError, "Missing keys"):
                load_student_moe_bundle(Path("/tmp/unused.pt"))

    def test_load_teacher_pretrained_uses_safe_torch_load(self) -> None:
        torch.manual_seed(3)
        source = PhiSatNetDownstream(
            pretrained_path=None,
            task="classification",
            input_dim=3,
            output_dim=4,
            depths=[2, 2, 2, 2],
            dims=[16, 32, 64, 128],
            img_size=224,
            freeze_body=False,
        )

        checkpoint_state = {}
        for key, value in source.stem.state_dict().items():
            checkpoint_state[f"module.stem.{key}"] = value
        for key, value in source.encoder.state_dict().items():
            checkpoint_state[f"module.encoder.{key}"] = value

        with patch("hydranet.models.teacher.torch.load") as mocked_load:
            mocked_load.return_value = {"state_dict": checkpoint_state}
            loaded = PhiSatNetDownstream(
                pretrained_path=Path("/tmp/unused.pt"),
                task="classification",
                input_dim=3,
                output_dim=4,
                depths=[2, 2, 2, 2],
                dims=[16, 32, 64, 128],
                img_size=224,
                freeze_body=False,
            )

        mocked_load.assert_called_once_with(Path("/tmp/unused.pt"), map_location="cpu", weights_only=True)
        for key, value in source.stem.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.stem.state_dict()[key]))
        for key, value in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.encoder.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
