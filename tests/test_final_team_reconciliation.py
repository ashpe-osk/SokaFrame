import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from utils.final_team_reconciliation import reconcile_final_team_labels
from utils.dashboard import compose_dashboard


class FinalTeamReconciliationTests(unittest.TestCase):
    fields = (
        "frame_number",
        "timestamp_seconds",
        "object_type",
        "tracker_id",
        "global_id",
        "player_type",
        "image_x",
        "image_y",
        "foot_x",
        "foot_y",
        "pitch_x",
        "pitch_y",
        "confidence",
        "is_visible",
        "bbox",
        "team_color",
        "possession_state",
        "possession_player_id",
        "source",
        "is_valid",
    )

    @staticmethod
    def _write_video(path, frame_count=4):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (160, 90),
        )
        if not writer.isOpened():
            raise OSError("Could not create test source video.")
        for _ in range(frame_count):
            writer.write(np.full((90, 160, 3), 45, dtype=np.uint8))
        writer.release()

    def _write_observations(self, path, goalkeeper_team="unknown_gk"):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            for frame_number, team in enumerate(
                ("team_1", "team_2", "team_2", "team_2"), start=1
            ):
                writer.writerow(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": (frame_number - 1) / 10,
                        "object_type": "player",
                        "tracker_id": 4,
                        "global_id": 28,
                        "player_type": team,
                        "foot_x": 60,
                        "foot_y": 70,
                        "confidence": 0.9,
                        "is_visible": True,
                        "bbox": json.dumps([50, 20, 70, 70]),
                        "team_color": json.dumps([0, 255, 255]),
                    }
                )
                writer.writerow(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": (frame_number - 1) / 10,
                        "object_type": "goalkeeper",
                        "tracker_id": 17,
                        "global_id": 11,
                        "player_type": goalkeeper_team,
                        "foot_x": 20,
                        "foot_y": 70,
                        "confidence": 0.8,
                        "is_visible": True,
                        "bbox": json.dumps([10, 20, 30, 70]),
                        "team_color": json.dumps(
                            [255, 0, 255]
                            if goalkeeper_team == "team_2"
                            else [205, 205, 205]
                        ),
                    }
                )
                writer.writerow(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": (frame_number - 1) / 10,
                        "object_type": "ball",
                        "image_x": 60,
                        "image_y": 68,
                        "possession_state": "Unknown",
                        "source": "real",
                        "is_valid": True,
                    }
                )

    def test_reconciles_team_and_possession_in_csv_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            self._write_video(source)
            self._write_observations(observations)

            report = reconcile_final_team_labels(
                observations,
                source,
                output,
                fps=10.0,
                ball_gate=True,
            )

            self.assertEqual(report["processed_frames"], 4)
            self.assertEqual(report["final_teams"]["28"], "team_2")
            self.assertEqual(report["team_evidence"]["28"]["assigned_team"], "team_2")
            with observations.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            player_rows = [
                row for row in rows if row["object_type"] == "player"
            ]
            goalkeeper_rows = [
                row for row in rows if row["object_type"] == "goalkeeper"
            ]
            self.assertEqual(
                {row["player_type"] for row in player_rows}, {"team_2"}
            )
            self.assertEqual(
                {row["player_type"] for row in goalkeeper_rows},
                {"unknown_gk"},
            )
            possession_by_frame = {}
            for row in rows:
                possession_by_frame.setdefault(
                    row["frame_number"], row["possession_state"]
                )
                self.assertEqual(
                    possession_by_frame[row["frame_number"]],
                    row["possession_state"],
                )
            self.assertEqual(
                possession_by_frame,
                {"1": "Unknown", "2": "Unknown", "3": "Unknown", "4": "team_2"},
            )

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(
                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4
                )
                self.assertEqual(
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 360
                )
            finally:
                capture.release()

    def test_final_video_rebuild_keeps_pass_overlay_with_live_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (320, 240),
            )
            if not writer.isOpened():
                raise OSError("Could not create test source video.")
            for _ in range(6):
                writer.write(np.full((240, 320, 3), 45, dtype=np.uint8))
            writer.release()

            with observations.open("w", newline="", encoding="utf-8") as stream:
                csv_writer = csv.DictWriter(stream, fieldnames=self.fields)
                csv_writer.writeheader()
                for frame_number in range(1, 7):
                    ball_x = 60 if frame_number <= 4 else 60 + 40 * (frame_number - 4)
                    csv_writer.writerow(
                        {
                            "frame_number": frame_number,
                            "timestamp_seconds": (frame_number - 1) / 10,
                            "object_type": "player",
                            "tracker_id": 4,
                            "global_id": 28,
                            "player_type": "team_1",
                            "foot_x": 60,
                            "foot_y": 70,
                            "confidence": 0.9,
                            "is_visible": True,
                            "bbox": json.dumps([50, 20, 70, 70]),
                            "team_color": json.dumps([0, 255, 255]),
                        }
                    )
                    csv_writer.writerow(
                        {
                            "frame_number": frame_number,
                            "timestamp_seconds": (frame_number - 1) / 10,
                            "object_type": "ball",
                            "image_x": ball_x,
                            "image_y": 70,
                            "source": "real",
                            "is_valid": True,
                        }
                    )

            with patch(
                "utils.final_team_reconciliation.compose_dashboard",
                wraps=compose_dashboard,
            ) as compose:
                reconcile_final_team_labels(
                    observations,
                    source,
                    output,
                    fps=10.0,
                    ball_gate=True,
                )

            final_counts = compose.call_args_list[-1].kwargs["pass_counts"]
            self.assertEqual(final_counts["team_1"]["attempted"], 1)
            self.assertEqual(final_counts["team_1"]["unknown"], 1)

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                capture.set(cv2.CAP_PROP_POS_FRAMES, 5)
                success, final_frame = capture.read()
                self.assertTrue(success)
                self.assertLess(
                    float(final_frame[100, 300].mean()),
                    float(final_frame[100, 310].mean()) - 3,
                )
            finally:
                capture.release()

    def test_ambiguous_team_evidence_remains_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            self._write_video(source, frame_count=2)
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fields)
                writer.writeheader()
                for frame_number, team in ((1, "team_1"), (2, "team_2")):
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "player",
                            "tracker_id": 1,
                            "global_id": 5,
                            "player_type": team,
                            "foot_x": 60,
                            "foot_y": 70,
                            "confidence": 0.9,
                            "bbox": json.dumps([50, 20, 70, 70]),
                            "team_color": json.dumps([0, 255, 255]),
                        }
                    )
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "ball",
                            "possession_state": "Unknown",
                            "source": "missing",
                            "is_valid": False,
                        }
                    )

            report = reconcile_final_team_labels(
                observations,
                source,
                output,
                fps=10.0,
                ball_gate=True,
            )

            self.assertEqual(report["final_teams"]["5"], "unknown")
            self.assertEqual(report["unresolved_global_ids"], ["5"])
            with observations.open(newline="", encoding="utf-8") as stream:
                players = [
                    row
                    for row in csv.DictReader(stream)
                    if row["object_type"] == "player"
                ]
            self.assertEqual(
                {row["player_type"] for row in players}, {"unknown"}
            )

    def test_confident_goalkeeper_team_is_preserved_for_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            self._write_video(source)
            self._write_observations(observations, goalkeeper_team="team_2")

            report = reconcile_final_team_labels(
                observations,
                source,
                output,
                fps=10.0,
                ball_gate=True,
            )

            self.assertEqual(report["final_teams"]["11"], "team_2")
            with observations.open(newline="", encoding="utf-8") as stream:
                goalkeeper_rows = [
                    row
                    for row in csv.DictReader(stream)
                    if row["object_type"] == "goalkeeper"
                ]
            self.assertEqual(
                {row["player_type"] for row in goalkeeper_rows},
                {"team_2"},
            )

    def test_sustained_team_conflict_keeps_a_neutral_identity_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            self._write_video(source)
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fields)
                writer.writeheader()
                for frame_number in range(1, 5):
                    conflicted = frame_number > 2
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "player",
                            "tracker_id": 7,
                            "global_id": 31,
                            "player_type": (
                                "team_conflict" if conflicted else "team_1"
                            ),
                            "foot_x": 60,
                            "foot_y": 70,
                            "confidence": 0.9,
                            "is_visible": True,
                            "bbox": json.dumps([50, 20, 70, 70]),
                            "team_color": json.dumps(
                                [205, 205, 205]
                                if conflicted
                                else [0, 255, 255]
                            ),
                        }
                    )
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "ball",
                            "possession_state": "Unknown",
                            "source": "missing",
                            "is_valid": False,
                        }
                    )

            report = reconcile_final_team_labels(
                observations,
                source,
                output,
                fps=10.0,
                ball_gate=True,
            )

            self.assertEqual(report["final_teams"]["31"], "team_1")
            with observations.open(newline="", encoding="utf-8") as stream:
                players = [
                    row
                    for row in csv.DictReader(stream)
                    if row["object_type"] == "player"
                ]
            self.assertEqual(
                [row["player_type"] for row in players],
                ["team_1", "team_1", "team_conflict", "team_conflict"],
            )
            self.assertEqual(
                json.loads(players[2]["team_color"]), [205, 205, 205]
            )

    def test_confirmed_other_player_keeps_identity_without_forcing_a_team(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "tracked.mp4"
            observations = root / "observations.csv"
            self._write_video(source, frame_count=2)
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.fields)
                writer.writeheader()
                for frame_number in (1, 2):
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "player",
                            "tracker_id": 7,
                            "global_id": 37,
                            "player_type": "other",
                            "foot_x": 60,
                            "foot_y": 70,
                            "confidence": 0.9,
                            "is_visible": True,
                            "bbox": json.dumps([50, 20, 70, 70]),
                            "team_color": json.dumps([205, 205, 205]),
                        }
                    )
                    writer.writerow(
                        {
                            "frame_number": frame_number,
                            "object_type": "ball",
                            "possession_state": "Unknown",
                            "source": "missing",
                            "is_valid": False,
                        }
                    )

            report = reconcile_final_team_labels(
                observations,
                source,
                output,
                fps=10.0,
                ball_gate=True,
            )

            self.assertNotIn("37", report["final_teams"])
            with observations.open(newline="", encoding="utf-8") as stream:
                players = [
                    row
                    for row in csv.DictReader(stream)
                    if row["object_type"] == "player"
                ]
            self.assertEqual({row["global_id"] for row in players}, {"37"})
            self.assertEqual({row["player_type"] for row in players}, {"other"})


if __name__ == "__main__":
    unittest.main()
