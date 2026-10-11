import unittest
from unittest.mock import patch

import numpy as np

from utils.team_color_utils import (
    CanonicalTeamConflictTracker,
    CanonicalTeamSmoother,
    TeamColorAssigner,
)


class TeamColorAssignerTests(unittest.TestCase):
    def test_team_evidence_uses_a_relative_color_distance_margin(self):
        evidence = TeamColorAssigner._team_evidence(
            {
                TeamColorAssigner.TEAM_1: 0.1,
                TeamColorAssigner.TEAM_2: 0.3,
            },
            minimum_margin=0.025,
        )

        self.assertEqual(evidence[0], TeamColorAssigner.TEAM_1)
        self.assertAlmostEqual(evidence[1], 0.5)
        self.assertIsNone(
            TeamColorAssigner._team_evidence(
                {
                    TeamColorAssigner.TEAM_1: 0.1,
                    TeamColorAssigner.TEAM_2: 0.11,
                },
                minimum_margin=0.025,
            )
        )

    def test_canonical_smoothing_uses_weighted_history_and_hysteresis(self):
        smoother = CanonicalTeamSmoother()
        team_1 = TeamColorAssigner.TEAM_1
        team_2 = TeamColorAssigner.TEAM_2

        self.assertIsNone(smoother.update(101, 1, (team_1, 0.6)))
        self.assertIsNone(smoother.update(101, 2, (team_1, 0.6)))
        self.assertEqual(smoother.update(101, 3, (team_1, 0.6)), team_1)
        self.assertEqual(smoother.update(101, 4, (team_2, 0.8)), team_1)

        result = team_1
        for frame_number in range(5, 11):
            result = smoother.update(101, frame_number, (team_2, 0.8))
        self.assertEqual(result, team_2)
        self.assertIsNone(smoother.update(202, 11, (team_2, 0.8)))

    def test_canonical_smoothing_rejects_out_of_order_evidence(self):
        smoother = CanonicalTeamSmoother()
        smoother.update(101, 5, (TeamColorAssigner.TEAM_1, 0.6))

        with self.assertRaises(ValueError):
            smoother.update(101, 4, (TeamColorAssigner.TEAM_1, 0.6))

    def test_canonical_smoothing_can_clear_stale_team_evidence(self):
        smoother = CanonicalTeamSmoother()
        team = TeamColorAssigner.TEAM_1
        for frame_number in range(1, 4):
            smoother.update(101, frame_number, (team, 0.8))

        smoother.clear(101)

        self.assertNotIn(101, smoother._history)
        self.assertNotIn(101, smoother._stable_teams)
        self.assertIsNone(smoother.update(101, 4, (team, 0.8)))

    def test_team_conflict_requires_sustained_evidence_and_recovers_cautiously(self):
        tracker = CanonicalTeamConflictTracker()
        team_1 = TeamColorAssigner.TEAM_1
        team_2 = TeamColorAssigner.TEAM_2

        for frame_number in range(1, tracker.CONFIRMATION_FRAMES):
            self.assertFalse(
                tracker.update(31, team_1, frame_number, (team_2, 0.8))
            )
        self.assertTrue(
            tracker.update(
                31, team_2, tracker.CONFIRMATION_FRAMES, (team_2, 0.8)
            )
        )

        result = True
        for frame_number in range(
            tracker.CONFIRMATION_FRAMES + 1,
            tracker.CONFIRMATION_FRAMES + tracker.RECOVERY_FRAMES,
        ):
            result = tracker.update(
                31, team_2, frame_number, (team_1, 0.8)
            )
        self.assertTrue(result)
        self.assertFalse(
            tracker.update(
                31,
                team_2,
                tracker.CONFIRMATION_FRAMES + tracker.RECOVERY_FRAMES,
                (team_1, 0.8),
            )
        )

    def test_weak_team_evidence_does_not_create_a_conflict(self):
        tracker = CanonicalTeamConflictTracker()
        for frame_number in range(1, 100):
            conflicted = tracker.update(
                31,
                TeamColorAssigner.TEAM_1,
                frame_number,
                (TeamColorAssigner.TEAM_2, 0.2),
            )
        self.assertFalse(conflicted)

    def test_confirmed_assignment_can_be_inherited_after_global_id_relink(self):
        assigner = TeamColorAssigner()
        assigner._provisional[42] = TeamColorAssigner.TEAM_2
        assigner._pending_assignments[42] = TeamColorAssigner.TEAM_2
        assigner._pending_counts[42] = 2

        assigner.inherit_assignment(42, TeamColorAssigner.TEAM_1)

        self.assertEqual(
            assigner._assignments[42], TeamColorAssigner.TEAM_1
        )
        self.assertNotIn(42, assigner._provisional)
        self.assertNotIn(42, assigner._pending_assignments)
        self.assertNotIn(42, assigner._pending_counts)

    @staticmethod
    def _fitted_assigner():
        assigner = TeamColorAssigner()
        assigner._team_centers_fitted = True
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.array((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.array((0.1, 0.8, 0.8)),
        }
        return assigner

    def test_kmeans_cluster_indices_map_by_center_color_not_cluster_number(self):
        assigner = TeamColorAssigner()

        mapping = assigner._map_clusters_to_teams(
            np.asarray(((0.1, 0.1), (0.1, 0.9)))
        )

        self.assertEqual(mapping[0], TeamColorAssigner.TEAM_1)
        self.assertEqual(mapping[1], TeamColorAssigner.TEAM_2)

    def test_confirmed_player_color_crop_is_refreshed_every_ten_frames(self):
        assigner = self._fitted_assigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = {
            "tracker_id": 7,
            "object_type": "player",
            "bbox": (20, 10, 40, 90),
        }
        feature = np.array((0.1, 0.1, 0.1))

        with patch.object(
            TeamColorAssigner,
            "_jersey_feature",
            return_value=feature,
        ) as feature_crop:
            for _ in range(13):
                assigner.update(
                    frame, [detection], enable_other_bucket=True
                )
            self.assertEqual(feature_crop.call_count, 4)
            self.assertEqual(
                assigner._assignments[7], TeamColorAssigner.TEAM_1
            )

            assigner.update(frame, [detection], enable_other_bucket=True)

        self.assertEqual(feature_crop.call_count, 5)

    def test_crossing_forces_a_fresh_team_color_crop_after_separation(self):
        assigner = self._fitted_assigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        feature = np.array((0.1, 0.1, 0.1))
        first = {
            "tracker_id": 7,
            "object_type": "player",
            "bbox": (10, 10, 30, 90),
        }
        crossing = {
            "tracker_id": 8,
            "object_type": "player",
            "bbox": (12, 12, 32, 88),
        }

        with patch.object(
            TeamColorAssigner,
            "_jersey_feature",
            return_value=feature,
        ) as feature_crop:
            for _ in range(assigner.PROVISIONAL_CONFIRMATION_FRAMES):
                assigner.update(frame, [first], enable_other_bucket=True)
            self.assertEqual(feature_crop.call_count, 4)

            assigner.update(
                frame, [first, crossing], enable_other_bucket=True
            )
            self.assertEqual(feature_crop.call_count, 4)
            self.assertIn(7, assigner._force_feature_refresh)

            assigner.update(frame, [first], enable_other_bucket=True)

        self.assertEqual(feature_crop.call_count, 5)
        self.assertNotIn(7, assigner._force_feature_refresh)

    def test_team_label_stays_provisional_until_four_clear_observations(self):
        assigner = TeamColorAssigner()
        assigner._team_centers_fitted = True
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.array((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.array((0.1, 0.8, 0.8)),
        }
        assigner._features[7] = np.array((0.1, 0.1, 0.1))

        for _ in range(assigner.PROVISIONAL_CONFIRMATION_FRAMES - 1):
            assigner._update_other_bucket_assignments({7})
        self.assertEqual(
            assigner._assignments.get(7, TeamColorAssigner.UNKNOWN),
            TeamColorAssigner.UNKNOWN,
        )
        self.assertTrue(assigner.is_provisional(7))

        assigner._update_other_bucket_assignments({7})
        self.assertEqual(
            assigner._assignments[7],
            TeamColorAssigner.TEAM_1,
        )
        self.assertFalse(assigner.is_provisional(7))

    def test_confirmed_team_relabel_requires_ten_clear_observations(self):
        assigner = TeamColorAssigner()
        assigner._team_centers_fitted = True
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.array((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.array((0.1, 0.8, 0.8)),
        }
        assigner._assignments[7] = TeamColorAssigner.TEAM_1
        assigner._features[7] = np.array((0.1, 0.8, 0.8))

        for _ in range(assigner.RELABEL_CONFIRMATION_FRAMES - 1):
            assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)

        assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_2)

    def test_goalkeeper_class_flicker_keeps_team_unknown(self):
        assigner = self._fitted_assigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        team_1_feature = np.array((0.1, 0.1, 0.1))
        team_2_feature = np.array((0.1, 0.8, 0.8))
        goalkeeper = {
            "tracker_id": 17,
            "object_type": "goalkeeper",
            "bbox": (20, 10, 40, 90),
        }
        mislabeled_player = {
            **goalkeeper,
            "object_type": "player",
        }
        torso_modes = []
        features = iter(
            (
                team_1_feature,
                team_2_feature,
                team_1_feature,
                team_1_feature,
                team_1_feature,
                team_1_feature,
                team_1_feature,
            )
        )

        def jersey_feature(_frame, _bbox, central_torso=False):
            torso_modes.append(central_torso)
            return next(features)

        with patch.object(
            TeamColorAssigner,
            "_jersey_feature",
            side_effect=jersey_feature,
        ), patch.object(
            TeamColorAssigner, "FEATURE_REFRESH_INTERVAL_FRAMES", 1
        ):
            for detection in (
                goalkeeper,
                mislabeled_player,
                mislabeled_player,
                mislabeled_player,
                mislabeled_player,
                mislabeled_player,
                mislabeled_player,
            ):
                result = assigner.update(frame, [detection], enable_other_bucket=True)

        self.assertEqual(
            torso_modes,
            [False, False, False, False, False, False, False],
        )
        self.assertEqual(result[17], TeamColorAssigner.UNKNOWN_GOALKEEPER)
        self.assertIn(17, assigner._goalkeeper_trackers)
        self.assertIn(17, assigner._goalkeeper_features)
        self.assertNotIn(17, assigner._features)

    def test_goalkeeper_transition_clears_prior_outfield_team_assignment(self):
        assigner = self._fitted_assigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assigner._assignments[17] = TeamColorAssigner.TEAM_2
        assigner._features[17] = np.array((0.1, 0.8, 0.8))
        assigner._feature_history[17] = [assigner._features[17]]
        ambiguous_feature = np.array((0.1, 0.45, 0.45))
        goalkeeper = {
            "tracker_id": 17,
            "object_type": "goalkeeper",
            "bbox": (20, 10, 40, 90),
        }

        with patch.object(
            TeamColorAssigner,
            "_jersey_feature",
            return_value=ambiguous_feature,
        ):
            result = assigner.update(
                frame, [goalkeeper], enable_other_bucket=True
            )

        self.assertEqual(
            assigner.last_goalkeeper_resets[17],
            TeamColorAssigner.TEAM_2,
        )
        self.assertNotIn(17, assigner._assignments)
        self.assertNotIn(17, assigner._feature_history)
        self.assertEqual(result[17], TeamColorAssigner.UNKNOWN_GOALKEEPER)

    def test_distinct_goalkeeper_kit_remains_unknown_without_team_evidence(self):
        assigner = self._fitted_assigner()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        distinct_kit = np.array((0.9, 0.9, 0.9))
        goalkeeper = {
            "tracker_id": 17,
            "object_type": "goalkeeper",
            "bbox": (20, 10, 40, 90),
        }

        with patch.object(
            TeamColorAssigner,
            "_jersey_feature",
            return_value=distinct_kit,
        ):
            for _ in range(5):
                result = assigner.update(
                    frame, [goalkeeper], enable_other_bucket=True
                )

        self.assertEqual(result[17], TeamColorAssigner.UNKNOWN_GOALKEEPER)
        self.assertNotIn(17, assigner._assignments)

    def test_roster_cap_keeps_only_eleven_least_ambiguous_assignments(self):
        assigner = TeamColorAssigner()
        assigner._team_centers_fitted = True
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.array((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.array((0.1, 0.8, 0.8)),
        }
        tracker_ids = set(range(12))
        for tracker_id in tracker_ids:
            assigner._assignments[tracker_id] = TeamColorAssigner.TEAM_1
            assigner._features[tracker_id] = np.array(
                (0.1, 0.1, 0.1 if tracker_id else 0.45)
            )

        assigner._enforce_team_roster_cap(tracker_ids)

        self.assertEqual(
            sum(
                assigner._assignments.get(tracker_id)
                == TeamColorAssigner.TEAM_1
                for tracker_id in tracker_ids
            ),
            assigner.MAX_VISIBLE_PLAYERS_PER_TEAM,
        )
        self.assertNotIn(0, assigner._assignments)


if __name__ == "__main__":
    unittest.main()
