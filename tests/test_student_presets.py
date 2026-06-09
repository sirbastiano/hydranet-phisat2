import unittest

from hydranet import DEFAULT_STUDENT_CONFIG, available_student_presets, load_student
from hydranet.models.student import create_phisatnet


class StudentPresetTests(unittest.TestCase):
    def test_only_checkpoint_preset_is_available_and_default(self):
        self.assertEqual(DEFAULT_STUDENT_CONFIG, "checkpoint")
        self.assertEqual(available_student_presets(), ["checkpoint"])

        factory_model = create_phisatnet()
        loaded_model = load_student()

        self.assertEqual(factory_model.channels, [16, 32, 64, 128])
        self.assertEqual(loaded_model.channels, [16, 32, 64, 128])


if __name__ == "__main__":
    unittest.main()
