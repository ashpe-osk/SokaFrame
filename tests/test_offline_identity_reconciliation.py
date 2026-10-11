import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from utils.offline_identity_reconciliation import reconcile_observations


class OfflineIdentityReconciliationTests(unittest.TestCase):
    @staticmethod
    def _write_observations(path):
        fields = (
            "frame_number",
            "object_type",
            "tracker_id",
            "bbox",
            "foot_x",
            "foot_y",
            "player_type",
            "confidence",
        )
        rows = [
            (1, "player", 10, "[10,10,30,60]", 20, 60, "team_1", 0.9),
            (2, "player", 10, "[11,10,31,60]", 21, 60, "team_1", 0.9),
            (2, "player", 20, "[50,10,70,60]", 60, 60, "team_1", 0.8),
            (3, "player", 20, "[51,10,71,60]", 61, 60, "team_1", 0.8),
            (5, "player", 10, "[12,10,32,60]", 22, 60, "team_1", 0.9),
            (6, "goalkeeper", 30, "[80,10,100,60]", 90, 60, "team_2", 0.9),
            (7, "ball", "", "[20,50,22,52]", "", "", "", 0.5),
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_splits_reused_raw_ids_and_never_merges_overlapping_tracklets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            self._write_observations(observations)
            before = hashlib.sha256(observations.read_bytes()).hexdigest()

            report = reconcile_observations(
                observations, root / "reconciled", fps=30
            )

            after = hashlib.sha256(observations.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertEqual(report["tracklets"], 4)
            self.assertEqual(report["automatic_merges"], 0)
            self.assertGreater(
                report["simultaneous_visibility_cannot_link_pairs"], 0
            )
            with (root / "reconciled" / "canonical_id_map.csv").open(
                newline="", encoding="utf-8-sig"
            ) as stream:
                mapping = list(csv.DictReader(stream))
            raw_10 = [row for row in mapping if row["raw_tracker_id"] == "10"]
            self.assertEqual(len(raw_10), 2)
            self.assertNotEqual(
                raw_10[0]["canonical_id"], raw_10[1]["canonical_id"]
            )
            self.assertEqual(len({row["canonical_id"] for row in mapping}), 4)

            with (
                root
                / "reconciled"
                / "identity_association_evidence.csv"
            ).open(newline="", encoding="utf-8-sig") as stream:
                evidence = list(csv.DictReader(stream))
            self.assertTrue(
                any(
                    row["constraint"]
                    == "simultaneous_visibility_cannot_link"
                    for row in evidence
                )
            )

    def test_reconciliation_is_deterministic_and_marks_missing_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            self._write_observations(observations)
            first = reconcile_observations(
                observations, root / "first", fps=30
            )
            second = reconcile_observations(
                observations, root / "second", fps=30
            )

            for name in (
                "canonical_id_map.csv",
                "identity_association_evidence.csv",
                "ambiguous_identity_matches.csv",
                "candidate_identity_merges.csv",
                "reconciliation_report.json",
            ):
                self.assertEqual(
                    (root / "first" / name).read_bytes(),
                    (root / "second" / name).read_bytes(),
                )
            self.assertTrue(
                first["automatic_merge_accuracy"].startswith("NOT EVALUATED")
            )
            self.assertEqual(first, second)

    def test_existing_map_rejects_simultaneous_identity_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observations = root / "observations.csv"
            self._write_observations(observations)
            existing = root / "old_map.csv"
            with existing.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "tracklet_id",
                        "start_frame",
                        "end_frame",
                        "canonical_id",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {
                            "tracklet_id": "10:1:2",
                            "start_frame": 1,
                            "end_frame": 2,
                            "canonical_id": 1,
                        },
                        {
                            "tracklet_id": "20:2:3",
                            "start_frame": 2,
                            "end_frame": 3,
                            "canonical_id": 1,
                        },
                    )
                )

            with self.assertRaisesRegex(ValueError, "simultaneous-visibility"):
                reconcile_observations(
                    observations,
                    root / "output",
                    existing_map_path=existing,
                )


if __name__ == "__main__":
    unittest.main()
