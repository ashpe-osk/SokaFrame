import unittest

from utils.pass_detection import EvidenceBasedPassDetector
from utils.team_color_utils import TeamColorAssigner


class ReleaseBasedPassDetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = EvidenceBasedPassDetector(fps=10, image_width=1000)
        self.passer = self._player("1", TeamColorAssigner.TEAM_1, 100)
        self.teammate = self._player("2", TeamColorAssigner.TEAM_1, 300)
        self.opponent = self._player("3", TeamColorAssigner.TEAM_2, 300)

    @staticmethod
    def _player(global_id, team, x, object_type="player", visible=True):
        return {
            "global_id": global_id,
            "object_type": object_type,
            "player_type": team,
            "foot_x": x,
            "foot_y": 100,
            "is_visible": visible,
        }

    def _frame(
        self,
        frame,
        ball_x,
        players,
        carrier_id,
        *,
        state=TeamColorAssigner.TEAM_1,
        source="real",
        valid=True,
    ):
        self.detector.update(
            frame,
            frame / 10,
            (ball_x, 100) if ball_x is not None else None,
            source,
            valid,
            players,
            possession_state=state,
            possession_player_id=carrier_id,
        )

    def _prime_control(self, player=None):
        player = player or self.passer
        for frame in range(1, 4):
            self._frame(frame, 100, [self.passer, self.teammate, self.opponent], player["global_id"])

    def _release(self, frame=4):
        self._frame(
            frame,
            180,
            [self.passer, self.teammate, self.opponent],
            "1",
        )

    def _resolve_to(self, player, frame=5):
        self._frame(
            frame,
            280,
            [self.passer, self.teammate, self.opponent],
            player["global_id"],
            state=player["player_type"],
        )

    def _assert_invariant(self, detector=None):
        for team_counts in (detector or self.detector).counts_by_team().values():
            self.assertEqual(
                team_counts["attempted"],
                team_counts["completed"]
                + team_counts["incomplete"]
                + team_counts["unknown"],
            )

    def test_release_counts_attempt_once_then_teammate_possession_completes(self):
        self._prime_control()
        self._release()

        self.assertEqual(self.detector.total_count(), 1)
        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_1],
            {"attempted": 1, "completed": 0, "incomplete": 0, "unknown": 1},
        )
        self._assert_invariant()

        self._resolve_to(self.teammate)
        for frame in range(6, 10):
            self._frame(
                frame, 280, [self.passer, self.teammate, self.opponent], "2"
            )

        attempt = self.detector.finish()[0]
        self.assertEqual(attempt["passer_id"], "1")
        self.assertEqual(attempt["receiver_id"], "2")
        self.assertEqual(attempt["outcome"], "completed")
        self.assertEqual(attempt["start_x_image"], 180)
        self.assertEqual(self.detector.total_count(), 1)
        self._assert_invariant()

    def test_confirmed_opponent_possession_marks_attempt_incomplete(self):
        self._prime_control()
        self._release()
        self._resolve_to(self.opponent)

        attempt = self.detector.finish()[0]
        self.assertEqual(attempt["outcome"], "incomplete")
        self.assertEqual(attempt["team"], TeamColorAssigner.TEAM_1)
        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_1]["incomplete"],
            1,
        )
        self._assert_invariant()

    def test_unresolved_release_stays_unknown_when_analysis_finishes(self):
        self._prime_control()
        self._release()
        self._frame(
            5, None, [self.passer, self.teammate], None, state="Unknown",
            source="missing", valid=False,
        )

        attempt = self.detector.finish()[0]
        self.assertEqual(attempt["outcome"], "unknown")
        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_1]["unknown"],
            1,
        )
        self._assert_invariant()

    def test_ball_detection_gap_does_not_erase_pending_attempt(self):
        self._prime_control()
        self._release()
        self._frame(
            5, None, [self.passer, self.teammate], None, state="Unknown",
            source="missing", valid=False,
        )
        self._frame(
            6, None, [self.passer, self.teammate], None, state="Unknown",
            source="missing", valid=False,
        )
        self._resolve_to(self.teammate, frame=7)

        self.assertEqual(self.detector.total_count(), 1)
        self.assertEqual(
            self.detector.get_passes()[0]["outcome"], "completed"
        )
        self._assert_invariant()

    def test_interpolated_release_and_real_receiver_possession_are_supported(self):
        self._prime_control()
        self._frame(
            4,
            180,
            [self.passer, self.teammate],
            "1",
            source="interpolated",
        )
        self._resolve_to(self.teammate)

        self.assertEqual(self.detector.get_passes()[0]["outcome"], "completed")

    def test_dribbling_with_carrier_does_not_create_attempt(self):
        for frame, ball_x in enumerate((100, 125, 150, 175, 200, 225), 1):
            carrier = self._player("1", TeamColorAssigner.TEAM_1, ball_x)
            self._frame(
                frame,
                ball_x,
                [carrier, self.teammate],
                "1",
            )

        self.assertEqual(self.detector.finish(), [])
        self.assertEqual(self.detector.total_count(), 0)

    def test_ball_motion_without_confirmed_carrier_is_not_an_attempt(self):
        for frame, ball_x in enumerate((100, 180, 260, 340), 1):
            self._frame(
                frame,
                ball_x,
                [self.passer, self.teammate],
                None,
                state="Unknown",
            )

        self.assertEqual(self.detector.finish(), [])
        self.assertEqual(self.detector.total_count(), 0)

    def test_same_player_regaining_control_does_not_invent_outcome(self):
        self._prime_control()
        self._release()
        for frame in range(5, 8):
            self._frame(frame, 105, [self.passer, self.teammate], "1")

        attempt = self.detector.finish()[0]
        self.assertEqual(attempt["outcome"], "unknown")
        self.assertEqual(
            attempt["outcome_reason"], "original_possessor_regained_control"
        )
        self._assert_invariant()

    def test_release_attempt_is_not_duplicated_while_outcome_is_pending(self):
        self._prime_control()
        self._release()
        for frame, ball_x in ((5, 220), (6, 260), (7, 280)):
            self._frame(
                frame,
                ball_x,
                [self.passer, self.teammate],
                None,
                state="Unknown",
            )

        self.assertEqual(self.detector.total_count(), 1)
        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_1]["attempted"],
            1,
        )
        self._assert_invariant()

    def test_unresolved_attempt_times_out_as_unknown(self):
        self._prime_control()
        self._release()
        for frame in range(5, 26):
            self._frame(
                frame,
                180 + frame,
                [self.passer, self.teammate],
                None,
                state="Loose",
            )

        self.assertEqual(self.detector.get_passes()[0]["outcome"], "unknown")
        self.assertEqual(
            self.detector.get_passes()[0]["outcome_reason"], "outcome_unresolved"
        )
        self._assert_invariant()

    def test_nearby_tackle_like_ball_perturbation_is_not_a_release(self):
        opponent = self._player("3", TeamColorAssigner.TEAM_2, 130)
        players = [self.passer, self.teammate, opponent]
        for frame, ball_x in enumerate((100, 110, 120, 130), 1):
            self._frame(frame, ball_x, players, "1")

        self.assertEqual(self.detector.finish(), [])
        self.assertEqual(self.detector.total_count(), 0)

    def test_live_counts_attribute_attempt_to_the_passer_team(self):
        self._prime_control()
        self._release()
        self._resolve_to(self.opponent)

        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_1],
            {"attempted": 1, "completed": 0, "incomplete": 1, "unknown": 0},
        )
        self.assertEqual(
            self.detector.counts_by_team()[TeamColorAssigner.TEAM_2]["attempted"],
            0,
        )
        self._assert_invariant()


if __name__ == "__main__":
    unittest.main()
