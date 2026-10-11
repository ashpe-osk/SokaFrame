import tempfile
import unittest
from pathlib import Path

from utils.run_directories import create_next_run_directory


class RunDirectoryTests(unittest.TestCase):
    def test_creates_sequential_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)

            created = [
                create_next_run_directory(output_root).name
                for _ in range(3)
            ]

            self.assertEqual(created, ["run_1", "run_2", "run_3"])
            self.assertTrue(
                all((output_root / name).is_dir() for name in created)
            )

    def test_uses_maximum_existing_run_number_and_ignores_other_folders(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for name in ("run_1", "run_2", "run_4", "debug", "models"):
                (output_root / name).mkdir()

            created = create_next_run_directory(output_root)

            self.assertEqual(created.name, "run_5")

    def test_new_run_name_has_no_timestamp_or_descriptive_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            created = create_next_run_directory(Path(directory))

            self.assertRegex(created.name, r"^run_[1-9]\d*$")
