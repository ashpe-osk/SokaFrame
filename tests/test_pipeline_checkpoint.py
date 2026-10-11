import tempfile
import unittest
from pathlib import Path

from utils.pipeline_checkpoint import (
    load_checkpoint,
    restore_checkpointed_file,
    save_checkpoint,
)


class PipelineCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trips_fingerprint_frame_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.checkpoint"
            fingerprint = {"video": "match.mp4", "frames": 1200}
            state = {"track_ids": {1, 2}, "values": [10, 20]}

            save_checkpoint(path, fingerprint, 100, state)

            self.assertEqual(load_checkpoint(path, fingerprint), (100, state))

    def test_checkpoint_rejects_changed_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.checkpoint"
            save_checkpoint(path, {"model": "best.pt"}, 100, {})

            with self.assertRaisesRegex(ValueError, "does not match"):
                load_checkpoint(path, {"model": "other.pt"})

    def test_checkpoint_can_resume_on_a_different_inference_device(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.checkpoint"
            saved_fingerprint = {
                "video": "match.mp4",
                "model": "best.pt",
                "backend": "auto",
                "device": "cpu",
            }
            expected_fingerprint = {
                "video": "match.mp4",
                "model": "best.pt",
                "backend": "auto",
                "device": "intel:gpu",
            }
            state = {"track_ids": {1, 2}}
            save_checkpoint(path, saved_fingerprint, 100, state)

            self.assertEqual(
                load_checkpoint(path, expected_fingerprint),
                (100, state),
            )

    def test_checkpoint_rejects_invalid_frame_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pipeline.checkpoint"
            save_checkpoint(path, {}, 0, {})

            with self.assertRaisesRegex(ValueError, "invalid state"):
                load_checkpoint(path, {})

    def test_resume_truncates_uncommitted_observation_and_evidence_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation_path = root / "observations.csv"
            evidence_path = root / "identity_association_evidence.csv"
            files = {
                observation_path: "header\ncommitted-observation\n",
                evidence_path: "header\ncommitted-evidence\n",
            }
            offsets = {}
            for path, content in files.items():
                path.write_text(content, encoding="utf-8")
                offsets[path.name] = path.stat().st_size

            checkpoint_path = root / "pipeline.checkpoint"
            save_checkpoint(
                checkpoint_path,
                {"video": "synthetic"},
                2,
                {
                    "observations_offset": offsets[observation_path.name],
                    "identity_evidence_offset": offsets[evidence_path.name],
                    "pending_output_frames": {2: {"frame_number": 2}},
                },
            )
            for path in files:
                with path.open("a", encoding="utf-8") as stream:
                    stream.write("uncommitted-after-checkpoint\n")

            _, state = load_checkpoint(
                checkpoint_path, {"video": "synthetic"}
            )
            for path, committed_content in files.items():
                with path.open("r+", encoding="utf-8") as stream:
                    restore_checkpointed_file(
                        stream, state[
                            "observations_offset"
                            if path == observation_path
                            else "identity_evidence_offset"
                        ]
                    )
                    stream.write("resumed-row\n")
                self.assertEqual(
                    path.read_text(encoding="utf-8"),
                    committed_content + "resumed-row\n",
                )
            self.assertEqual(
                sorted(state["pending_output_frames"]),
                [2],
            )


if __name__ == "__main__":
    unittest.main()
