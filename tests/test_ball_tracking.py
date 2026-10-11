import unittest

from utils.ball_tracking import BallPositionTracker


class BallPositionTrackerTests(unittest.TestCase):
    @staticmethod
    def detection(x, y, confidence):
        return {
            "confidence": confidence,
            "bbox": (x - 5, y - 5, x + 5, y + 5),
        }

    def test_interpolates_short_gaps_between_real_detections(self):
        tracker = BallPositionTracker()
        detections = [
            self.detection(15, 15, 0.12),
            self.detection(40, 50, 0.22),
        ]

        first = tracker.update(detections, 1)
        self.assertEqual(
            first,
            [
                {
                    "frame_number": 1,
                    "position": (40.0, 50.0),
                    "source": "real",
                    "confidence": 0.22,
                }
            ],
        )
        self.assertEqual(tracker.update([], 2), [])
        self.assertEqual(tracker.update([], 3), [])

        resolved = tracker.update(
            [self.detection(65, 85, 0.8)], 4
        )

        self.assertEqual(
            [record["frame_number"] for record in resolved],
            [2, 3, 4],
        )
        self.assertEqual(
            [record["source"] for record in resolved],
            ["interpolated", "interpolated", "real"],
        )
        self.assertAlmostEqual(resolved[0]["position"][0], 40 + 25 / 3)
        self.assertAlmostEqual(resolved[0]["position"][1], 50 + 35 / 3)
        self.assertAlmostEqual(resolved[1]["position"][0], 40 + 50 / 3)
        self.assertAlmostEqual(resolved[1]["position"][1], 50 + 70 / 3)
        self.assertIsNone(resolved[0]["confidence"])
        self.assertEqual(resolved[2]["confidence"], 0.8)
        self.assertEqual(tracker.real_frames, 2)
        self.assertEqual(tracker.interpolated_frames, 2)

    def test_long_gaps_and_video_edges_are_missing_not_extrapolated(self):
        tracker = BallPositionTracker(max_gap_frames=2)

        self.assertEqual(tracker.update([], 1), [])
        self.assertEqual(tracker.update([], 2), [])
        before_first_real = tracker.update(
            [self.detection(10, 10, 0.9)], 3
        )
        self.assertEqual(
            [(row["frame_number"], row["source"]) for row in before_first_real],
            [(1, "missing"), (2, "missing"), (3, "real")],
        )

        self.assertEqual(tracker.update([], 4), [])
        self.assertEqual(tracker.update([], 5), [])
        released = tracker.update([], 6)
        self.assertEqual(released[0]["frame_number"], 4)
        self.assertEqual(released[0]["source"], "missing")
        after_long_gap = tracker.update(
            [self.detection(40, 40, 0.8)], 7
        )
        self.assertEqual(
            [(row["frame_number"], row["source"]) for row in after_long_gap],
            [(5, "missing"), (6, "missing"), (7, "real")],
        )

        trailing = tracker.update([], 8)
        self.assertEqual(trailing, [])
        self.assertEqual(
            [(row["frame_number"], row["source"]) for row in tracker.finish()],
            [(8, "missing")],
        )
        self.assertEqual(tracker.interpolated_frames, 0)
        self.assertEqual(tracker.unknown_frames, 6)

    def test_rejects_non_increasing_frame_numbers(self):
        tracker = BallPositionTracker()
        tracker.update([], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 1)


if __name__ == "__main__":
    unittest.main()
