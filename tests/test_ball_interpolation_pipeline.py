import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from main import _select_possession_ball_position, main
from utils.ball_tracking import BallPositionTracker
from utils.camera_motion import CameraMotion
from utils.phase1_analysis import TrackingStubWriter
from utils.pipeline_checkpoint import load_checkpoint, save_checkpoint
from utils.possession_utils import PossessionEstimator


class BallInterpolationPipelineTests(unittest.TestCase):
    def test_ball_gate_only_passes_direct_ball_detections_to_possession(self):
        position = (20.0, 30.0)
        candidate = SimpleNamespace(
            player_type="team_1",
            foot_xy=position,
            global_id=1,
        )

        self.assertEqual(
            _select_possession_ball_position(position, "real", True),
            position,
        )
        self.assertIsNone(
            _select_possession_ball_position(position, "interpolated", True)
        )
        self.assertEqual(
            _select_possession_ball_position(position, "interpolated", False),
            position,
        )
        gated_possession = PossessionEstimator()
        ungated_possession = PossessionEstimator()
        for _ in range(4):
            gated_possession.update(
                _select_possession_ball_position(
                    position, "interpolated", True
                ),
                [candidate],
            )
            ungated_possession.update(
                _select_possession_ball_position(
                    position, "interpolated", False
                ),
                [candidate],
            )
        self.assertEqual(gated_possession.state, "Unknown")
        self.assertEqual(ungated_possession.state, "team_1")

    def test_pending_interpolation_state_survives_checkpoint_round_trip(self):
        tracker = BallPositionTracker()
        tracker.update(
            [{"bbox": (5.0, 5.0, 15.0, 15.0), "confidence": 0.9}],
            1,
        )
        self.assertEqual(tracker.update([], 2), [])
        pending_output_frames = {
            2: {"frame": np.zeros((4, 4, 3), dtype=np.uint8)}
        }
        state = {
            "ball_tracker": tracker.__dict__.copy(),
            "pending_output_frames": pending_output_frames,
        }

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pipeline.checkpoint"
            fingerprint = {"ball_interpolation": "linear-buffered-v1"}
            save_checkpoint(checkpoint_path, fingerprint, 2, state)
            _, restored_state = load_checkpoint(
                checkpoint_path, fingerprint
            )

        restored_tracker = BallPositionTracker()
        restored_tracker.__dict__.update(restored_state["ball_tracker"])
        np.testing.assert_array_equal(
            restored_state["pending_output_frames"][2]["frame"],
            pending_output_frames[2]["frame"],
        )
        resolved = restored_tracker.update(
            [{"bbox": (25.0, 5.0, 35.0, 15.0), "confidence": 0.8}],
            3,
        )
        self.assertEqual(resolved[0]["frame_number"], 2)
        self.assertEqual(resolved[0]["position"], (20.0, 10.0))
        self.assertEqual(resolved[0]["source"], "interpolated")

    def test_buffered_pipeline_emits_interpolated_ball_rows_and_video_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "source.avi"
            video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (160, 120),
            )
            self.assertTrue(video_writer.isOpened())
            for frame_number in range(8):
                video_writer.write(
                    np.full((120, 160, 3), (45, 120, 45), dtype=np.uint8)
                )
            video_writer.release()

            stub_path = root / "tracking.pkl"
            motion = CameraMotion(
                None,
                (0.0, 0.0),
                0.0,
                1.0,
                0.0,
                0,
                0,
                False,
                False,
            )
            writer = TrackingStubWriter(
                stub_path,
                video_path=video_path,
                class_names={0: "ball"},
                fps=10.0,
                frame_size=(160, 120),
            )
            ball_centers = {1: 30.0, 4: 90.0, 8: 120.0}
            for frame_number in range(1, 9):
                center_x = ball_centers.get(frame_number)
                detections = (
                    [
                        {
                            "bbox": (
                                center_x - 2,
                                58.0,
                                center_x + 2,
                                62.0,
                            ),
                            "class_id": 0,
                            "object_type": "ball",
                            "confidence": 0.9,
                            "tracker_id": None,
                        }
                    ]
                    if center_x is not None
                    else []
                )
                writer.write_frame(
                    frame_number,
                    detections,
                    motion,
                    {},
                    {},
                    len(detections),
                    set(),
                )
            writer.close()

            modes = (
                ("legacy", []),
                ("online", ["--canonical-ids", "online"]),
                (
                    "online_ball_gate",
                    ["--canonical-ids", "online", "--ball-gate"],
                ),
                (
                    "online_team_smoothing",
                    ["--canonical-ids", "online", "--team-smoothing"],
                ),
                ("offline", ["--canonical-ids", "offline"]),
            )

            for mode_name, mode_flags in modes:
                output_root = root / f"output-{mode_name}"
                main(
                    [
                        "--video",
                        str(video_path),
                        "--model",
                        str(root / "unused.pt"),
                        "--use-stub",
                        "--stub-path",
                        str(stub_path),
                        "--output-dir",
                        str(output_root),
                        "--progress-every",
                        "0",
                        *mode_flags,
                    ]
                )

                run_dir = output_root / "run_1"
                for artifact in (
                    "tracked.mp4",
                    "observations.csv",
                    "identity_association_evidence.csv",
                    "pitch_rejected_detections.csv",
                ):
                    self.assertTrue(
                        (run_dir / artifact).is_file(),
                        f"{mode_name} did not produce {artifact}",
                    )
                with (run_dir / "observations.csv").open(
                    newline="", encoding="utf-8"
                ) as source:
                    ball_rows = [
                        row
                        for row in csv.DictReader(source)
                        if row["object_type"] == "ball"
                    ]
                self.assertEqual(len(ball_rows), 8)
                self.assertEqual(
                    [int(row["frame_number"]) for row in ball_rows],
                    list(range(1, 9)),
                )
                self.assertEqual(
                    [row["source"] for row in ball_rows],
                    [
                        "real",
                        "interpolated",
                        "interpolated",
                        "real",
                        "interpolated",
                        "interpolated",
                        "interpolated",
                        "real",
                    ],
                )
                self.assertAlmostEqual(float(ball_rows[1]["image_x"]), 50.0)
                self.assertAlmostEqual(float(ball_rows[2]["image_x"]), 70.0)
                self.assertAlmostEqual(float(ball_rows[4]["image_x"]), 97.5)
                self.assertAlmostEqual(float(ball_rows[5]["image_x"]), 105.0)
                self.assertAlmostEqual(float(ball_rows[6]["image_x"]), 112.5)
                self.assertEqual(ball_rows[1]["confidence"], "")

                output_video = cv2.VideoCapture(
                    str(run_dir / "tracked.mp4")
                )
                written_frames = 0
                while output_video.read()[0]:
                    written_frames += 1
                output_video.release()
                self.assertEqual(written_frames, 8)
                if mode_name == "offline":
                    offline_dir = run_dir / "offline_reconciliation"
                    for artifact in (
                        "canonical_id_map.csv",
                        "ambiguous_identity_matches.csv",
                        "candidate_identity_merges.csv",
                        "reconciliation_report.json",
                    ):
                        self.assertTrue((offline_dir / artifact).is_file())


if __name__ == "__main__":
    unittest.main()
