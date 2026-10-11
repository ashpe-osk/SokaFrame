import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main import _identity_transition_events
from utils.global_identity_manager import GlobalIdentityManager


class GlobalIdentityManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = GlobalIdentityManager(
            match_threshold=0.66,
            ambiguity_margin=0.06,
        )

    @staticmethod
    def detection(
        tracker_id,
        x=100,
        y=200,
        *,
        team="team_1",
        object_type="player",
        confidence=0.95,
        feature=(1.0, 0.0, 0.0),
        eligible=True,
        size=(20, 50),
        team_confident=True,
    ):
        width, height = size
        return {
            "tracker_id": tracker_id,
            "object_type": object_type,
            "bbox": (x - width / 2, y - height, x + width / 2, y),
            "confidence": confidence,
            "team": team,
            "team_confident": team_confident,
            "reid_feature": feature,
            "player_eligible": eligible,
        }

    def test_new_player_gets_monotonic_global_id(self):
        ids = self.manager.update([self.detection(5)], 1)
        self.assertEqual(ids, {5: 1})
        self.assertEqual(self.manager.snapshot()["created_global_ids"], 1)

    def test_same_botsort_id_keeps_global_id(self):
        first = self.manager.update([self.detection(5)], 1)
        second = self.manager.update([self.detection(5, x=104)], 2)
        self.assertEqual(first[5], second[5])
        self.assertEqual(self.manager.snapshot()["created_global_ids"], 1)

    def test_mapped_global_identity_survives_conflicting_team_evidence(self):
        global_id = self.manager.update(
            [self.detection(5, team="team_1")], 1
        )[5]

        updated = self.manager.update(
            [
                self.detection(
                    5,
                    team="team_2",
                )
            ],
            2,
        )

        self.assertEqual(updated[5], global_id)
        self.assertEqual(self.manager.identities[global_id].team, "team_1")
        self.assertEqual(self.manager.tracker_to_global[5], global_id)

    def test_smoothed_team_label_can_correct_a_canonical_team(self):
        global_id = self.manager.update([self.detection(5)], 1)[5]

        self.manager.set_team_label(global_id, "team_2")

        self.assertEqual(self.manager.identities[global_id].team, "team_2")
        with self.assertRaises(ValueError):
            self.manager.set_team_label(global_id, "unknown")

    def test_smoothed_team_label_initializes_an_unknown_canonical_team(self):
        global_id = self.manager.update(
            [self.detection(5, team="unknown")], 1
        )[5]

        self.manager.set_team_label(global_id, "team_2")

        self.assertEqual(self.manager.identities[global_id].team, "team_2")

    def test_goalkeeper_classification_can_clear_a_stale_team_label(self):
        global_id = self.manager.update([self.detection(5)], 1)[5]
        self.manager.set_team_label(global_id, "team_2")

        self.manager.clear_team_label(global_id)

        self.assertEqual(self.manager.identities[global_id].team, "unknown")

    def test_mapped_player_appearance_histogram_updates_every_ten_frames(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        histogram = np.full(48, 1.0 / 48, dtype=np.float32)
        detection = self.detection(5)

        with patch(
            "utils.global_identity_manager._appearance_histogram",
            return_value=histogram,
        ) as make_histogram:
            self.manager.update([detection], 1, frame=frame)
            for frame_number in range(2, 11):
                self.manager.update(
                    [self.detection(5, x=100 + frame_number)],
                    frame_number,
                    frame=frame,
                )
            self.assertEqual(make_histogram.call_count, 1)

            self.manager.update([self.detection(5, x=111)], 11, frame=frame)

        self.assertEqual(make_histogram.call_count, 2)
        self.assertEqual(self.manager.tracker_to_global[5], 1)

    def test_lost_identity_candidates_reuse_one_current_appearance_histogram(self):
        histogram = np.full(48, 1.0 / 48, dtype=np.float32)
        detection = self.detection(29)
        with patch(
            "utils.global_identity_manager._appearance_histogram",
            return_value=histogram,
        ) as make_histogram:
            first_score = self.manager._appearance_score(
                SimpleNamespace(appearance_histogram=histogram),
                detection,
                (300, 400, 3),
            )
            second_score = self.manager._appearance_score(
                SimpleNamespace(appearance_histogram=histogram),
                detection,
                (300, 400, 3),
            )

        self.assertEqual(first_score, second_score)
        make_histogram.assert_called_once()

    def test_changed_botsort_id_reuses_matching_global_id(self):
        first = self.manager.update([self.detection(5)], 1)
        second = self.manager.update([self.detection(29, x=102)], 2)
        self.assertEqual(first[5], second[29])
        self.assertEqual(self.manager.snapshot()["created_global_ids"], 1)
        self.assertEqual(self.manager.snapshot()["reidentified_tracker_changes"], 1)
        self.assertNotIn(5, self.manager.tracker_to_global)
        accepted = next(
            row
            for row in self.manager.last_association_evidence
            if row["decision"] == "accepted"
        )
        self.assertEqual(accepted["candidate_global_id"], first[5])
        self.assertEqual(accepted["candidate_tracker_id"], 5)
        self.assertGreaterEqual(accepted["combined_score"], self.manager.match_threshold)
        self.assertGreaterEqual(accepted["appearance_reid_score"], 0.0)
        self.assertEqual(accepted["algorithm_version"], "global-identity-online-v1")

    def test_temporary_absence_then_reappearance_keeps_global_id(self):
        original = self.manager.update([self.detection(5)], 1)[5]
        self.assertEqual(self.manager.update([], 2), {})
        recovered = self.manager.update([self.detection(29, x=108)], 12)[29]
        self.assertEqual(recovered, original)

    def test_strong_reid_reconnects_after_returning_far_from_last_position(self):
        original = self.manager.update([
            self.detection(5, x=100, feature=(1, 0, 0))
        ], 1)[5]

        recovered = self.manager.update([
            self.detection(29, x=500, feature=(1, 0, 0))
        ], 21)

        self.assertEqual(recovered[29], original)

    def test_ambiguous_strong_reid_does_not_merge_distant_profiles(self):
        original = self.manager.update([
            self.detection(5, x=100, feature=(1, 0, 0)),
            self.detection(6, x=100, feature=(0, 1, 0)),
        ], 1)

        returned = self.manager.update([
            self.detection(29, x=100, feature=(1, 1, 0))
        ], 21)

        self.assertNotIn(29, returned)
        self.assertEqual(len(set(original.values())), 2)
        self.assertEqual(self.manager.snapshot()["created_global_ids"], 2)
        self.assertIn(29, self.manager.pending_tracker_ids)

        resolved = self.manager.update(
            [self.detection(29, x=100, feature=(1, 0, 0))],
            22,
        )
        self.assertEqual(resolved[29], original[5])
        self.assertNotIn(29, self.manager.pending_tracker_ids)

    def test_two_players_cross_without_sharing_global_id(self):
        first = self.manager.update([
            self.detection(5, x=100, feature=(1, 0, 0)),
            self.detection(6, x=300, feature=(0, 1, 0)),
        ], 1)
        self.assertEqual(len(set(first.values())), 2)
        crossed = self.manager.update([
            self.detection(5, x=300, feature=(1, 0, 0)),
            self.detection(6, x=100, feature=(0, 1, 0)),
        ], 2)
        self.assertEqual(crossed, first)
        self.assertEqual(len(set(crossed.values())), 2)

    def test_similar_appearance_is_disambiguated_by_position(self):
        initial = self.manager.update([
            self.detection(5, x=100, feature=(1, 0.02, 0)),
            self.detection(6, x=300, feature=(1, 0, 0.02)),
        ], 1)
        next_frame = self.manager.update([
            self.detection(5, x=105, feature=(1, 0.01, 0)),
            self.detection(6, x=295, feature=(1, 0, 0.01)),
        ], 2)
        self.assertEqual(next_frame, initial)
        new_tracks = self.manager.update([
            self.detection(20, x=108, feature=(1, 0.02, 0)),
            self.detection(21, x=292, feature=(1, 0, 0.02)),
        ], 3)
        self.assertEqual(new_tracks[20], initial[5])
        self.assertEqual(new_tracks[21], initial[6])

    def test_team_mismatch_cannot_reidentify_as_another_team(self):
        original = self.manager.update([
            self.detection(5, team="team_1")
        ], 1)[5]
        changed_team = self.manager.update([
            self.detection(29, team="team_2")
        ], 2)
        self.assertNotIn(29, changed_team)
        self.assertIn(29, self.manager.pending_tracker_ids)
        team_conflict = next(
            row
            for row in self.manager.last_association_evidence
            if row["team_conflict"]
        )
        self.assertEqual(team_conflict["candidate_global_id"], original)
        self.assertEqual(team_conflict["decision"], "unresolved")
        self.assertEqual(team_conflict["constraint"], "confident_team_conflict")

    def test_team_mismatch_candidate_does_not_break_scoring_other_candidates(self):
        initial = self.manager.update([
            self.detection(5, x=100, team="team_1"),
            self.detection(6, x=300, team="team_2"),
        ], 1)

        recovered = self.manager.update([
            self.detection(29, x=102, team="team_1"),
        ], 2)

        self.assertEqual(recovered[29], initial[5])

    def test_global_id_is_unique_among_simultaneously_visible_players(self):
        ids = self.manager.update([
            self.detection(1, x=100),
            self.detection(2, x=100),
        ], 1)
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids.values())), 2)

    def test_outside_referee_ball_and_low_confidence_are_excluded(self):
        ids = self.manager.update([
            self.detection(1, x=100, eligible=False),
            self.detection(2, object_type="referee"),
            self.detection(3, object_type="ball"),
            self.detection(4, object_type="other"),
            self.detection(5, confidence=0.2),
        ], 1)
        self.assertEqual(ids, {})
        self.assertEqual(self.manager.snapshot()["created_global_ids"], 0)

    def test_camera_compensated_positions_reconnect_across_image_pan(self):
        first = self.detection(5, x=100)
        first["camera_compensated_position"] = (100, 200)
        first_id = self.manager.update([first], 1, camera_epoch=0)[5]
        shifted = self.detection(29, x=400)
        shifted["camera_compensated_position"] = (104, 200)
        shifted_id = self.manager.update([shifted], 2, camera_epoch=0)[29]
        self.assertEqual(first_id, shifted_id)

    def test_pitch_only_contributes_when_marked_reliable(self):
        first = self.detection(5, feature=None)
        first["pitch_position"] = (10, 20)
        first["pitch_reliable"] = True
        original = self.manager.update([first], 1)[5]
        second = self.detection(29, x=500, feature=None)
        second["pitch_position"] = (10.2, 20.1)
        second["pitch_reliable"] = True
        self.assertEqual(self.manager.update([second], 2)[29], original)

    def test_registry_retains_more_than_sixty_four_identities(self):
        manager = GlobalIdentityManager()
        for frame in range(1, 81):
            manager.update(
                [
                    self.detection(
                        frame,
                        x=50 * frame,
                        team="unknown",
                        feature=tuple(
                            1.0 if index == frame - 1 else 0.0
                            for index in range(80)
                        ),
                    )
                ],
                frame,
            )
        self.assertEqual(len(manager.identities), 80)
        self.assertEqual(manager.snapshot()["retained_global_ids"], 80)
        self.assertTrue(all(len(item.history) <= 8 for item in manager.identities.values()))

    def test_legacy_capacity_remains_available_for_baseline_comparison(self):
        manager = GlobalIdentityManager(
            max_identities=2,
            max_lost_frames=100,
        )
        manager.update(
            [
                self.detection(
                    1,
                    x=100,
                    team="unknown",
                    feature=(1.0, 0.0, 0.0),
                )
            ],
            1,
        )
        manager.update(
            [
                self.detection(
                    2,
                    x=200,
                    team="unknown",
                    feature=(0.0, 1.0, 0.0),
                )
            ],
            2,
        )
        manager.update(
            [
                self.detection(
                    3,
                    x=300,
                    team="unknown",
                    feature=(0.0, 0.0, 1.0),
                )
            ],
            3,
        )

        self.assertEqual(len(manager.identities), 2)
        self.assertEqual(manager.snapshot()["registry_capacity"], 2)
        self.assertNotIn(1, manager.identities)

    def test_old_identity_prototype_remains_a_candidate_after_long_absence(self):
        original = self.manager.update(
            [self.detection(5, feature=(1, 0, 0))], 1
        )[5]
        for frame in range(2, 60001):
            self.manager.update([], frame)

        recovered = self.manager.update(
            [self.detection(29, feature=(1, 0, 0))], 60001
        )[29]

        self.assertEqual(recovered, original)

    def test_checkpoint_restores_identity_assignments_and_prototypes(self):
        original = self.manager.update(
            [self.detection(5, feature=(1, 0, 0))], 1
        )[5]
        checkpoint = self.manager.checkpoint_state()
        restored = GlobalIdentityManager(
            match_threshold=0.66,
            ambiguity_margin=0.06,
        )
        restored.restore_checkpoint_state(checkpoint)

        self.assertEqual(
            restored.update(
                [self.detection(29, feature=(1, 0, 0))], 2
            )[29],
            original,
        )

    def test_checkpoint_preserves_unresolved_track_until_it_can_be_matched(self):
        self.manager.update(
            [
                self.detection(5, x=100, feature=(1, 0, 0)),
                self.detection(6, x=100, feature=(0, 1, 0)),
            ],
            1,
        )
        self.assertNotIn(
            29,
            self.manager.update(
                [self.detection(29, x=100, feature=(1, 1, 0))],
                2,
            ),
        )
        checkpoint = self.manager.checkpoint_state()

        restored = GlobalIdentityManager(
            match_threshold=0.66,
            ambiguity_margin=0.06,
        )
        restored.restore_checkpoint_state(checkpoint)

        self.assertIn(29, restored.pending_tracker_ids)
        self.assertEqual(
            restored.update(
                [self.detection(29, x=100, feature=(1, 0, 0))],
                3,
            )[29],
            1,
        )

    def test_legacy_checkpoint_defaults_to_no_pending_tracks(self):
        checkpoint = self.manager.checkpoint_state()
        checkpoint["state"].pop("pending_tracker_ids")

        restored = GlobalIdentityManager(
            match_threshold=0.66,
            ambiguity_margin=0.06,
        )
        restored.restore_checkpoint_state(checkpoint)

        self.assertEqual(restored.pending_tracker_ids, set())

    def test_checkpoint_rejects_unknown_version_and_incompatible_settings(self):
        checkpoint = self.manager.checkpoint_state()
        with self.assertRaisesRegex(ValueError, "version"):
            self.manager.restore_checkpoint_state({"version": 99, "state": {}})

        incompatible = GlobalIdentityManager(
            match_threshold=0.70,
            ambiguity_margin=0.06,
        )
        with self.assertRaisesRegex(ValueError, "settings"):
            incompatible.restore_checkpoint_state(checkpoint)

    def test_decreasing_frame_number_is_rejected(self):
        self.manager.update([self.detection(5)], 2)
        with self.assertRaises(ValueError):
            self.manager.update([self.detection(5)], 1)

    def test_transition_events_for_global_and_tracker_identity_changes(self):
        previous_global_by_tracker = {7: 3}
        previous_tracker_by_global = {3: 7}
        current_global_by_tracker = {7: 3, 19: 3}
        events = _identity_transition_events(
            previous_global_by_tracker,
            previous_tracker_by_global,
            current_global_by_tracker,
            550,
            camera_epoch=2,
            timestamp=18.23,
        )
        self.assertIn(
            {
                "event": "global_tracker_changed",
                "frame": 550,
                "global_id": 3,
                "previous_tracker_id": 7,
                "tracker_id": 19,
                "camera_epoch": 2,
                "timestamp": 18.23,
            },
            events,
        )

        events = _identity_transition_events(
            {7: 3},
            {3: 7},
            {7: 14},
            550,
            camera_epoch=2,
            timestamp=18.23,
        )
        self.assertIn(
            {
                "event": "tracker_global_changed",
                "frame": 550,
                "tracker_id": 7,
                "previous_global_id": 3,
                "global_id": 14,
                "camera_epoch": 2,
                "timestamp": 18.23,
            },
            events,
        )


if __name__ == "__main__":
    unittest.main()
