import unittest
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics.engine.results import Results

from utils.tracking_input_filter import PlayerTrackingInputFilter


class PlayerTrackingInputFilterTests(unittest.TestCase):
    def test_only_in_playing_area_player_and_goalkeeper_boxes_reach_tracker(self):
        frame = cv2.cvtColor(
            np.full((200, 240, 3), (60, 180, 150), dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )
        boxes = np.asarray(
            [
                (40, 50, 80, 150, 0.9, 2),
                (40, 0, 80, 30, 0.9, 2),
                (100, 50, 140, 150, 0.9, 3),
                (102, 52, 138, 148, 0.9, 2),
                (160, 50, 190, 145, 0.8, 1),
                (10, 10, 20, 20, 0.9, 0),
                (200, 50, 220, 100, 0.9, 4),
            ],
            dtype=np.float32,
        )
        result = Results(
            orig_img=frame,
            path="frame.jpg",
            names={
                0: "ball",
                1: "goalkeeper",
                2: "player",
                3: "referee",
                4: "staff",
            },
            boxes=boxes,
        )
        predictor = SimpleNamespace(results=[result])
        tracking_filter = PlayerTrackingInputFilter(
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
            referee_duplicate_confidence=0.3,
            referee_duplicate_iou=0.5,
            diagnostics_enabled=True,
        )
        tracking_filter.frame_number = 12

        tracking_filter(predictor)

        self.assertEqual(tracking_filter.selected_indices, [[0, 4]])
        self.assertEqual(len(predictor.results[0].boxes), 2)
        self.assertEqual(tracking_filter.total_detection_count, 7)
        self.assertEqual(tracking_filter.valid_detection_counts["player"], 1)
        self.assertEqual(
            tracking_filter.valid_detection_counts["goalkeeper"], 1
        )
        self.assertEqual(tracking_filter.filter_counts["outside_pitch"], 1)
        self.assertEqual(
            tracking_filter.filter_counts["overlapping_referee"], 1
        )
        self.assertEqual(
            [
                (
                    detection["frame_number"],
                    detection["class_id"],
                    detection["rejection_reason"],
                )
                for detection in tracking_filter.rejected_detections
            ],
            [
                (12, 2, "outside_pitch"),
                (12, 2, "overlapping_referee"),
            ],
        )
        self.assertEqual(
            tracking_filter.rejected_detections[0]["bbox"],
            (40.0, 0.0, 80.0, 30.0),
        )
        self.assertAlmostEqual(
            tracking_filter.rejected_detections[1]["confidence"], 0.9
        )

        tracker = SimpleNamespace(
            frame_id=1,
            tracked_stracks=[
                SimpleNamespace(
                    idx=0, track_id=17, frame_id=1, is_activated=True
                ),
                SimpleNamespace(
                    idx=1, track_id=23, frame_id=1, is_activated=True
                ),
            ],
        )
        detections = tracking_filter.attach_tracker_ids(tracker)

        self.assertEqual(detections[0]["tracker_id"], 17)
        self.assertEqual(detections[4]["tracker_id"], 23)
        self.assertTrue(detections[0]["pretracker_selected"])
        self.assertFalse(detections[1]["pretracker_selected"])
        self.assertEqual(
            detections[1]["pretracker_rejection_reason"],
            "outside_pitch",
        )
        self.assertEqual(
            detections[0]["pretracker_spatial_evidence"][
                "local_grass_support"
            ],
            1.0,
        )
        self.assertTrue(detections[0]["pretracker_spatial_evidence"]["inside_playing_area"])
        self.assertTrue(
            all(
                detection["tracker_id"] is None
                for index, detection in enumerate(detections)
                if index not in (0, 4)
            )
        )


if __name__ == "__main__":
    unittest.main()
