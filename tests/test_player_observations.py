import unittest

from utils.player_observations import BallObservation, PlayerObservation


class PlayerObservationTests(unittest.TestCase):
    def test_player_record_uses_active_tracker_id(self):
        observation = PlayerObservation(
            frame_number=3,
            timestamp_seconds=0.1,
            tracker_id=15,
            object_type="player",
            player_type="team_1",
            image_x=100.0,
            image_y=200.0,
            foot_x=100.0,
            foot_y=200.0,
            pitch_x=None,
            pitch_y=None,
            confidence=0.9,
            is_visible=True,
            bbox=(90.0, 100.0, 110.0, 200.0),
            team_color=(0, 255, 255),
        )

        record = observation.to_dict()
        self.assertEqual(record["tracker_id"], 15)
        self.assertIsNone(record["pitch_x"])

    def test_ball_record_has_no_player_id_or_fake_pitch_position(self):
        observation = BallObservation(
            frame_number=1,
            timestamp_seconds=0.0,
            image_x=10.0,
            image_y=20.0,
            pitch_x=None,
            pitch_y=None,
            confidence=0.8,
            source="real",
            is_valid=True,
        )

        self.assertEqual(observation.position, (10.0, 20.0))
        self.assertIsNone(observation.to_dict()["pitch_x"])
        self.assertNotIn("tracker_id", observation.to_dict())


if __name__ == "__main__":
    unittest.main()
