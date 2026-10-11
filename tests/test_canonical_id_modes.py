import unittest

from main import _parse_args


class CanonicalIdModeTests(unittest.TestCase):
    def test_default_mode_keeps_legacy_behavior(self):
        args = _parse_args([])
        self.assertEqual(args.canonical_ids, "off")

    def test_online_and_offline_modes_are_explicit_opt_ins(self):
        self.assertEqual(
            _parse_args(["--canonical-ids", "online"]).canonical_ids,
            "online",
        )
        self.assertEqual(
            _parse_args(["--canonical-ids", "offline"]).canonical_ids,
            "offline",
        )

    def test_invalid_mode_is_rejected_by_argument_parser(self):
        with self.assertRaises(SystemExit):
            _parse_args(["--canonical-ids", "automatic"])

    def test_ball_gate_requires_online_canonical_ids(self):
        with self.assertRaises(SystemExit):
            _parse_args(["--ball-gate"])
        args = _parse_args(
            ["--canonical-ids", "online", "--ball-gate"]
        )
        self.assertTrue(args.ball_gate)

    def test_team_smoothing_requires_online_canonical_ids(self):
        with self.assertRaises(SystemExit):
            _parse_args(["--team-smoothing"])
        args = _parse_args(
            ["--canonical-ids", "online", "--team-smoothing"]
        )
        self.assertTrue(args.team_smoothing)


if __name__ == "__main__":
    unittest.main()
