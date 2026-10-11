import unittest

from utils.possession_utils import PossessionEstimator
from utils.team_color_utils import TeamColorAssigner


class PossessionEstimatorTests(unittest.TestCase):
    def setUp(self):
        self.estimator = PossessionEstimator(
            distance_threshold=70,
            confirmation_frames=4,
        )
        self.candidates = [
            {
                "tracker_id": 40,
                "object_type": "player",
                "player_type": TeamColorAssigner.TEAM_1,
                "foot_xy": (100, 100),
            },
            {
                "tracker_id": 70,
                "object_type": "goalkeeper",
                "player_type": TeamColorAssigner.TEAM_2,
                "foot_xy": (111, 100),
            },
            {
                "tracker_id": 80,
                "object_type": "player",
                "player_type": TeamColorAssigner.UNKNOWN,
                "foot_xy": (110, 100),
            },
        ]

    def test_nearest_tracker_team_player_wins_after_four_frames(self):
        first = self.estimator.update((110, 100), self.candidates)
        self.assertEqual(first["closest_player_id"], 40)
        self.assertAlmostEqual(first["closest_distance"], 10)
        self.assertEqual(first["candidate_count"], 1)
        self.assertEqual(first["state"], "Unknown")

        for _ in range(3):
            result = self.estimator.update((110, 100), self.candidates)

        self.assertEqual(result["state"], TeamColorAssigner.TEAM_1)
        self.assertEqual(result["player_id"], 40)
        self.assertEqual(self.estimator.team_1_possession_frames, 1)
        self.assertAlmostEqual(self.estimator.percentages()["team_1"], 25.0)
        self.assertAlmostEqual(self.estimator.percentages()["unknown"], 75.0)
        self.assertAlmostEqual(sum(self.estimator.percentages().values()), 100.0)

    def test_far_ball_becomes_loose_and_missing_ball_becomes_unknown(self):
        for _ in range(4):
            result = self.estimator.update((500, 500), self.candidates)
        self.assertEqual(result["state"], "Loose")
        self.assertGreater(result["closest_distance"], 70)

        missing = self.estimator.update(None, self.candidates)
        self.assertEqual(missing["state"], "Unknown")
        self.assertEqual(self.estimator.unknown_frames, 4)
        self.assertAlmostEqual(self.estimator.percentages()["unknown"], 80.0)
        self.assertEqual(
            sum(self.estimator.percentages().values()),
            100.0,
        )

    def test_unknown_player_and_referee_are_not_candidates(self):
        unknown_player = {
            "tracker_id": 8,
            "player_type": TeamColorAssigner.UNKNOWN,
            "foot_xy": (101, 100),
        }
        referee = {
            "tracker_id": 9,
            "player_type": TeamColorAssigner.REFEREE,
            "foot_xy": (102, 100),
        }
        result = self.estimator.update(
            (100, 100), [unknown_player, referee]
        )
        self.assertEqual(result["candidate_count"], 0)
        self.assertIsNone(result["closest_player_id"])

    def test_referee_with_team_label_and_goalkeeper_are_excluded(self):
        candidates = [
            {
                "global_id": 10,
                "object_type": "referee",
                "player_type": TeamColorAssigner.TEAM_1,
                "foot_xy": (100, 100),
            },
            {
                "global_id": 11,
                "object_type": "goalkeeper",
                "player_type": TeamColorAssigner.TEAM_2,
                "foot_xy": (101, 100),
            },
        ]
        result = self.estimator.update((100, 100), candidates)

        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["state"], "Unknown")
        self.assertEqual(self.estimator.team_control_percentages(), {
            "team_1": 0.0,
            "team_2": 0.0,
        })

    def test_possession_player_id_survives_botsort_tracker_id_change(self):
        first_candidates = [{
            "tracker_id": 40,
            "global_id": 7,
            "player_type": TeamColorAssigner.TEAM_1,
            "foot_xy": (100, 100),
        }]
        changed_tracker_candidates = [{
            "tracker_id": 99,
            "global_id": 7,
            "player_type": TeamColorAssigner.TEAM_1,
            "foot_xy": (100, 100),
        }]
        self.estimator.update((100, 100), first_candidates)
        for _ in range(3):
            result = self.estimator.update(
                (100, 100), changed_tracker_candidates
            )
        self.assertEqual(result["state"], TeamColorAssigner.TEAM_1)
        self.assertEqual(result["player_id"], 7)
        self.assertEqual(result["closest_player_id"], 7)

    def test_short_ball_detection_gap_preserves_confirmed_carrier_then_expires(self):
        for _ in range(self.estimator.confirmation_frames):
            result = self.estimator.update((100, 100), [self.candidates[0]])
        self.assertEqual(result["state"], TeamColorAssigner.TEAM_1)

        for _ in range(3):
            result = self.estimator.update(None, [])
            self.assertEqual(result["state"], TeamColorAssigner.TEAM_1)
            self.assertEqual(result["player_id"], 40)
        result = self.estimator.update(None, [])

        self.assertEqual(result["state"], "Unknown")
        self.assertIsNone(result["player_id"])
        self.assertEqual(self.estimator.valid_outfield_possession_frames, 1)
        self.assertEqual(self.estimator.unknown_frames, 7)
        self.assertEqual(
            self.estimator.team_control_percentages(),
            {"team_1": 100.0, "team_2": 0.0},
        )
        self.assertEqual(self.estimator.percentages()["team_1"], 12.5)
        self.assertEqual(self.estimator.percentages()["unknown"], 87.5)

    def test_pending_loss_of_control_is_not_counted_as_valid_possession(self):
        for _ in range(self.estimator.confirmation_frames):
            self.estimator.update((100, 100), [self.candidates[0]])

        for _ in range(2):
            result = self.estimator.update((500, 500), [self.candidates[0]])
            self.assertEqual(result["state"], TeamColorAssigner.TEAM_1)

        self.assertEqual(self.estimator.valid_outfield_possession_frames, 1)
        self.assertEqual(self.estimator.team_1_possession_frames, 1)
        self.assertEqual(self.estimator.unknown_frames, 5)
        self.assertEqual(self.estimator.team_control_percentages(), {
            "team_1": 100.0,
            "team_2": 0.0,
        })

    def test_track_style_percentages_use_only_team_assigned_frames(self):
        self.estimator.update(None, self.candidates)
        self.estimator.update((500, 500), self.candidates)
        for _ in range(4):
            self.estimator.update((100, 100), self.candidates)

        self.assertEqual(
            self.estimator.team_control_percentages(),
            {"team_1": 100.0, "team_2": 0.0},
        )
        self.assertAlmostEqual(self.estimator.percentages()["team_1"], 16.6666667)


if __name__ == "__main__":
    unittest.main()
