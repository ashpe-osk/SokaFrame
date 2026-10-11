import csv
import json
import tempfile
import unittest
from pathlib import Path

from utils.match_analytics import write_match_analytics


class MatchAnalyticsTests(unittest.TestCase):
    def test_exports_summary_statistics_possession_sequences_and_tracking_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            event_source = root / "phase1_diagnostics" / "events.jsonl"
            event_source.parent.mkdir()
            event_source.write_text(
                '{"frame":2,"event":"tracker_appears"}\n',
                encoding="utf-8",
            )
            fields = (
                "frame_number",
                "timestamp_seconds",
                "object_type",
                "tracker_id",
                "global_id",
                "player_type",
                "confidence",
                "possession_state",
                "possession_player_id",
                "is_valid",
            )
            rows = [
                (1, 0.0, "ball", "", "", "", "", "Unknown", "", False),
                (1, 0.0, "goalkeeper", 7, 70, "team_2", 0.9, "Unknown", "", False),
                (1, 0.0, "player", 1, 11, "team_1", 0.8, "Unknown", "", False),
                (2, 0.5, "ball", "", "", "", "", "team_1", 11, True),
                (2, 0.5, "player", 1, 11, "team_1", 0.8, "team_1", 11, False),
                (3, 1.0, "ball", "", "", "", "", "team_1", 11, True),
                (3, 1.0, "player", 9, 11, "team_2", 0.7, "team_1", 11, False),
                (4, 1.5, "ball", "", "", "", "", "Unknown", "", False),
            ]
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)

            summary = write_match_analytics(
                observations,
                root / "run",
                fps=2.0,
                tracking_events_path=event_source,
            )

            self.assertEqual(summary["processed_frames"], 4)
            self.assertEqual(summary["unique_global_players"], 2)
            self.assertEqual(
                summary["global_ids_with_conflicting_known_teams"], ["11"]
            )
            self.assertEqual(
                summary["possession_frames"],
                {"Unknown": 2, "team_1": 2},
            )
            self.assertTrue(summary["passes"]["available"])
            self.assertFalse(summary["football_events"]["available"])
            self.assertTrue((root / "run" / "tracking_events.jsonl").is_file())
            self.assertTrue((root / "run" / "passes.csv").is_file())
            self.assertTrue((root / "run" / "passing_network.csv").is_file())

            with (root / "run" / "player_statistics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                players = {
                    row["global_id"]: row for row in csv.DictReader(stream)
                }
            self.assertEqual(players["11"]["team"], "inconsistent")
            self.assertEqual(players["11"]["visible_frames"], "3")

            with (root / "run" / "possession_sequences.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                sequences = list(csv.DictReader(stream))
            self.assertEqual(
                [
                    (
                        row["frame_start"],
                        row["frame_end"],
                        row["possession_state"],
                    )
                    for row in sequences
                ],
                [
                    ("1", "1", "Unknown"),
                    ("2", "3", "team_1"),
                    ("4", "4", "Unknown"),
                ],
            )
            self.assertEqual(
                summary["team_possession_percentages"],
                {"team_1": 100.0, "team_2": 0.0},
            )
            self.assertEqual(
                summary["valid_outfield_possession_frames"],
                {"team_1": 1, "team_2": 0},
            )

            written_summary = json.loads(
                (root / "run" / "match_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(written_summary, summary)

    def test_rejects_conflicting_possession_states_within_a_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            observations.write_text(
                "frame_number,timestamp_seconds,object_type,possession_state\n"
                "1,0,ball,Unknown\n"
                "1,0,player,team_1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Conflicting possession"):
                write_match_analytics(observations, root / "run", fps=30.0)

    def test_team_percentage_excludes_frames_without_a_valid_ball_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            observations.write_text(
                "frame_number,timestamp_seconds,object_type,global_id,"
                "player_type,is_visible,possession_state,possession_player_id,"
                "is_valid\n"
                "1,0,ball,,,,team_1,11,false\n"
                "1,0,player,11,team_1,true,team_1,11,\n",
                encoding="utf-8",
            )

            summary = write_match_analytics(
                observations, root / "run", fps=30.0
            )

            self.assertEqual(
                summary["valid_outfield_possession_frames"],
                {"team_1": 0, "team_2": 0},
            )
            self.assertEqual(
                summary["team_possession_percentages"],
                {"team_1": 0.0, "team_2": 0.0},
            )

    def test_pass_artifacts_export_only_evidence_supported_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            fields = (
                "frame_number",
                "timestamp_seconds",
                "object_type",
                "global_id",
                "player_type",
                "foot_x",
                "foot_y",
                "image_x",
                "image_y",
                "possession_state",
                "possession_player_id",
                "source",
                "is_valid",
            )
            rows = []
            for frame in range(1, 7):
                rows.append(
                    (frame, frame / 10, "ball", "", "", "", "", 100, 100,
                     "team_1", "1", "real", True)
                )
                rows.append(
                    (frame, frame / 10, "player", "1", "team_1", 100, 100,
                     "", "", "team_1", "1", "", "")
                )
            for frame, x in ((7, 170), (8, 200), (9, 230)):
                rows.append(
                    (frame, frame / 10, "ball", "", "", "", "", x, 100,
                     "Loose", "", "real", True)
                )
                rows.append(
                    (frame, frame / 10, "player", "1", "team_1", 100, 100,
                     "", "", "Loose", "", "", "")
                )
            for frame in range(10, 14):
                x = 230 + (frame - 9) * 3
                rows.append(
                    (frame, frame / 10, "ball", "", "", "", "", x, 100,
                     "team_1", "2", "real", True)
                )
                rows.append(
                    (frame, frame / 10, "player", "2", "team_1", 200, 100,
                     "", "", "team_1", "2", "", "")
                )
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)

            summary = write_match_analytics(
                observations, root / "run", fps=10.0, image_width=1280
            )

            self.assertEqual(summary["passes"]["attempted"], 1)
            self.assertEqual(summary["passes"]["completed"], 1)
            self.assertEqual(summary["pass_statistics"]["completion_percentage"], 100.0)
            self.assertEqual(summary["image_width"], 1280)
            self.assertEqual(len(summary["pass_visualizations"]), 1)
            visualization = summary["pass_visualizations"][0]
            with (root / "run" / "passes.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                exported_passes = list(csv.DictReader(stream))
            self.assertEqual(exported_passes[0]["outcome"], "completed")
            for field in (
                "start_timestamp_seconds",
                "end_timestamp_seconds",
                "start_x_image",
                "start_y_image",
                "end_x_image",
                "end_y_image",
                "team",
                "outcome",
            ):
                self.assertEqual(
                    visualization[field],
                    (
                        float(exported_passes[0][field])
                        if field in (
                            "start_timestamp_seconds",
                            "end_timestamp_seconds",
                            "start_x_image",
                            "start_y_image",
                            "end_x_image",
                            "end_y_image",
                        )
                        else exported_passes[0][field]
                    ),
                )
            self.assertEqual(exported_passes[0]["coordinate_space"], "image_pixels")
            self.assertEqual(
                exported_passes[0]["pitch_coordinates_status"], "uncalibrated"
            )
            self.assertEqual(exported_passes[0]["start_x_pitch"], "")
            with (root / "run" / "passing_network.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                network = list(csv.DictReader(stream))
            self.assertEqual(
                network,
                [{
                    "team": "team_1",
                    "passer_id": "1",
                    "receiver_id": "2",
                    "completed_passes": "1",
                }],
            )


if __name__ == "__main__":
    unittest.main()
