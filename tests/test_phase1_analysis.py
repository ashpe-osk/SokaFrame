import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.camera_motion import CameraMotion
from utils.phase1_analysis import (
    Phase1Analyzer,
    TrackingStubReader,
    TrackingStubWriter,
    camera_motion_from_dict,
)
from utils.team_color_utils import TeamColorAssigner


class Phase1AnalysisTests(unittest.TestCase):
    def test_tracking_stub_round_trips_detections_motion_and_end_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "source.mp4"
            video_path.touch()
            stub_path = root / "tracking.pkl"
            motion = CameraMotion(
                np.asarray(((1.0, 0.0, 4.0), (0.0, 1.0, -2.0))),
                (4.0, -2.0),
                0.0,
                1.0,
                0.9,
                12,
                15,
                False,
                True,
            )
            detection = {
                "bbox": (10.0, 20.0, 30.0, 70.0),
                "class_id": 2,
                "object_type": "player",
                "confidence": 0.91,
                "tracker_id": 8,
                "reid_feature": np.asarray((1.0, 0.0), dtype=np.float32),
            }
            writer = TrackingStubWriter(
                stub_path,
                video_path=video_path,
                class_names={2: "player"},
                fps=30,
                frame_size=(128, 96),
            )
            writer.write_frame(
                1,
                [detection],
                motion,
                {"outside_pitch": 2},
                {"player": 10},
                12,
                {7},
            )
            writer.close()

            reader = TrackingStubReader(
                stub_path,
                video_path=video_path,
                frame_size=(128, 96),
            )
            record = reader.read_frame(1)
            self.assertEqual(record["detections"][0]["tracker_id"], 8)
            np.testing.assert_array_equal(
                record["detections"][0]["reid_feature"], (1.0, 0.0)
            )
            self.assertEqual(record["lost_track_ids"], [7])
            self.assertEqual(
                record["pretracker_filter_counts"], {"outside_pitch": 2}
            )
            self.assertEqual(
                record["pretracker_valid_detection_counts"], {"player": 10}
            )
            self.assertEqual(record["pretracker_total_detection_count"], 12)
            restored_motion = camera_motion_from_dict(record["camera_motion"])
            np.testing.assert_allclose(
                restored_motion.transform, motion.transform
            )
            self.assertIsNone(reader.read_frame(2))
            reader.close()

    def test_analyzer_writes_rows_events_contact_sheet_and_case_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analyzer = Phase1Analyzer(root, fps=30)
            assigner = TeamColorAssigner()
            assigner.last_color_distances = {
                1: {TeamColorAssigner.TEAM_1: 0.1, TeamColorAssigner.TEAM_2: 0.8}
            }
            frame = np.zeros((96, 128, 3), dtype=np.uint8)
            first_detection = {
                "bbox": (20.0, 10.0, 44.0, 80.0),
                "class_id": 2,
                "object_type": "player",
                "confidence": 0.9,
                "tracker_id": 1,
            }
            overlapping_detection = {
                "bbox": (28.0, 14.0, 52.0, 82.0),
                "class_id": 2,
                "object_type": "player",
                "confidence": 0.88,
                "tracker_id": 2,
            }
            analyzer.observe(
                frame,
                1,
                [first_detection, overlapping_detection],
                {1: TeamColorAssigner.TEAM_1, 2: TeamColorAssigner.UNKNOWN},
                {1: 5, 2: 8},
                assigner,
                0,
            )
            analyzer.observe(
                frame,
                2,
                [first_detection],
                {1: TeamColorAssigner.TEAM_2},
                {1: 5},
                assigner,
                0,
            )
            analyzer.finish(2)

            with (root / "debug_detections.csv").open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            events = [
                json.loads(line)
                for line in (root / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(rows), 3)
            self.assertTrue(any(e["event"] == "player_boxes_overlap" for e in events))
            self.assertTrue(any(e["event"] == "tracker_disappears" for e in events))
            self.assertTrue(any(e["event"] == "team_label_changed" for e in events))
            self.assertTrue((root / "contact_sheet.jpg").is_file())
            self.assertTrue((root / "case_evidence.json").is_file())


if __name__ == "__main__":
    unittest.main()
