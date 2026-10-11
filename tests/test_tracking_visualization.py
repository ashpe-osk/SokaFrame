import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, TrackState

from utils.dashboard import PANEL_HEIGHT, compose_dashboard, output_size
from utils.camera_radar import CameraRadarRenderer
from utils.class_aware_tracker import (
    ClassAwareByteTrack,
    register_class_aware_bytetrack,
)
from utils.detection_classes import map_class_categories
from utils.pitch_region import (
    build_pitch_grass_mask,
    is_inside_playing_area,
    is_foot_point_on_grass,
    overlaps_referee_detection,
)
from utils.pitch_radar import PitchRadarProjector, SoccerPitchConfiguration
from utils.possession_utils import PossessionEstimator
from utils.referee_tracking import RefereeTrackManager
from utils.team_color_utils import TeamColorAssigner
from utils.visualization_utils import (
    draw_annotations,
    is_displayable_player,
)


class TrackingVisualizationTests(unittest.TestCase):
    def test_camera_radar_draws_normalized_frame_positions_and_ball_with_opencv(self):
        renderer = CameraRadarRenderer(output_size=(420, 245))
        markers = {
            1: (640.0, 360.0, (0, 0, 255), 1),
            2: (320.0, 180.0, (255, 0, 0), 2),
        }

        image = renderer.render(
            markers,
            frame_size=(720, 1280, 3),
            ball_position=(960.0, 540.0),
        )

        self.assertEqual(image.shape, (245, 420, 3))
        self.assertTupleEqual(tuple(image[131, 210]), (0, 0, 255))
        self.assertTupleEqual(tuple(image[80, 111]), (255, 0, 0))
        self.assertTupleEqual(
            tuple(image[183, 309]),
            TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.BALL],
        )
        self.assertEqual(renderer.TITLE, "CAMERA VIEW - NOT PITCH CALIBRATED")
        self.assertTrue(np.any(np.all(image == (205, 220, 210), axis=2)))

    def test_camera_radar_uses_shared_palette_for_ball(self):
        renderer = CameraRadarRenderer(output_size=(420, 245))
        ball_color = (3, 9, 17)
        image = renderer.render(
            {},
            frame_size=(720, 1280, 3),
            ball_position=(960.0, 540.0),
            team_colors={TeamColorAssigner.BALL: ball_color},
        )

        self.assertTupleEqual(tuple(image[183, 309]), ball_color)

    def test_pitch_region_and_duplicate_referee_filter(self):
        self.assertTrue(is_inside_playing_area((640, 360), (720, 1280, 3)))
        self.assertFalse(is_inside_playing_area((640, 100), (720, 1280, 3)))
        self.assertFalse(is_inside_playing_area((640, 710), (720, 1280, 3)))

        candidate = {"bbox": (100, 100, 140, 180)}
        same_person_referee = {
            "bbox": (101, 101, 141, 181),
            "confidence": 0.38,
        }
        nearby_referee = {
            "bbox": (145, 100, 180, 180),
            "confidence": 0.9,
        }
        self.assertTrue(
            overlaps_referee_detection(candidate, [same_person_referee])
        )
        self.assertFalse(overlaps_referee_detection(candidate, [nearby_referee]))

    def test_pitch_grass_mask_uses_largest_closed_hull_and_foot_sample(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        grass_color = cv2.cvtColor(
            np.asarray([[[60, 180, 150]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        frame[20:100, 30:210] = grass_color
        frame[0:5, 0:5] = grass_color

        mask = build_pitch_grass_mask(frame)

        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, frame.shape[:2])
        self.assertTrue(is_foot_point_on_grass(mask, (100, 80)))
        self.assertFalse(is_foot_point_on_grass(mask, (10, 10)))
        self.assertTrue(is_foot_point_on_grass(None, (10, 10)))

    def test_pitch_grass_mask_returns_none_without_grass(self):
        mask = build_pitch_grass_mask(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertIsNone(mask)

    def test_edge_foot_requires_local_grass_support(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        mask = np.full((60, 60), 255, dtype=np.uint8)

        self.assertFalse(is_foot_point_on_grass(mask, (5, 30), frame=frame))
        self.assertTrue(is_foot_point_on_grass(mask, (46, 30), frame=frame))

        grass_color = cv2.cvtColor(
            np.asarray([[[60, 180, 150]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        frame[20:30, 0:10] = grass_color
        self.assertTrue(is_foot_point_on_grass(mask, (5, 30), frame=frame))
        self.assertTrue(is_foot_point_on_grass(mask, (30, 30), frame=frame))
        self.assertTrue(is_foot_point_on_grass(None, (5, 30), frame=frame))

    def test_bottom_edge_foot_requires_strong_local_grass_support(self):
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        mask = np.full((60, 60), 255, dtype=np.uint8)
        self.assertFalse(
            is_foot_point_on_grass(mask, (30, 57), frame=frame)
        )

        grass_color = cv2.cvtColor(
            np.asarray([[[60, 180, 150]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        frame[47:56, 26:35] = grass_color
        self.assertTrue(
            is_foot_point_on_grass(mask, (30, 57), frame=frame)
        )

    def test_other_bucket_uses_30_label_majority(self):
        assigner = TeamColorAssigner()
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.asarray((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.asarray((0.9, 0.9, 0.9)),
        }
        assigner._team_centers_fitted = True
        assigner._features[7] = np.asarray((0.1, 0.1, 0.1))
        for _ in range(30):
            assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)

        assigner._features[7] = np.asarray((0.9, 0.9, 0.9))
        assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)
        self.assertEqual(len(assigner._other_label_history[7]), 30)

    def test_other_bucket_uses_full_lab_distance_for_team_labels(self):
        assigner = TeamColorAssigner()
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.asarray(
                (139.0, 124.5, 134.5)
            )
            / 255.0,
            TeamColorAssigner.TEAM_2: np.asarray(
                (190.5, 124.0, 149.0)
            )
            / 255.0,
        }
        assigner._team_centers_fitted = True
        feature = np.asarray((170.0, 126.0, 140.0)) / 255.0
        assigner._features[3] = feature

        chroma_distances = assigner._team_chroma_distances(feature)
        lab_distances = assigner._team_distances(feature)
        self.assertEqual(
            min(chroma_distances, key=chroma_distances.get),
            TeamColorAssigner.TEAM_1,
        )
        self.assertEqual(
            min(lab_distances, key=lab_distances.get),
            TeamColorAssigner.TEAM_2,
        )

        for _ in range(assigner.OTHER_LABEL_HISTORY_SIZE):
            assigner._update_other_bucket_assignments({3})

        self.assertEqual(
            assigner._assignments[3], TeamColorAssigner.TEAM_2
        )

    def test_other_bucket_does_not_flip_confirmed_team_on_opposite_color_votes(self):
        assigner = TeamColorAssigner()
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.asarray((0.1, 0.1, 0.1)),
            TeamColorAssigner.TEAM_2: np.asarray((0.9, 0.9, 0.9)),
        }
        assigner._team_centers_fitted = True
        assigner._features[7] = np.asarray((0.1, 0.1, 0.1))
        for _ in range(30):
            assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)

        assigner._features[7] = np.asarray((0.9, 0.9, 0.9))
        for _ in range(8):
            assigner._update_other_bucket_assignments({7})

        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)
        self.assertEqual(assigner.team_label_change_count, 0)

        for _ in range(
            assigner.RELABEL_CONFIRMATION_FRAMES - 8 - 1
        ):
            assigner._update_other_bucket_assignments({7})

        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)
        assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_2)
        self.assertEqual(assigner.team_label_change_count, 1)

        assigner._features[7] = np.asarray((0.5, 0.5, 0.5))
        for _ in range(30):
            assigner._update_other_bucket_assignments({7})

        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_2)
        self.assertEqual(assigner.team_label_change_count, 1)

    def test_other_bucket_does_not_reclassify_confirmed_other_as_a_team(self):
        assigner = TeamColorAssigner()
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.asarray((0.0, 0.0, 0.0)),
            TeamColorAssigner.TEAM_2: np.asarray((1.0, 1.0, 1.0)),
        }
        assigner._team_centers_fitted = True
        assigner._features[7] = np.asarray((0.5, 0.5, 0.5))
        for _ in range(assigner.OTHER_LABEL_HISTORY_SIZE):
            assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.OTHER)

        assigner._features[7] = np.asarray((0.0, 0.0, 0.0))
        for _ in range(29):
            assigner._update_other_bucket_assignments({7})

        self.assertEqual(assigner._assignments[7], TeamColorAssigner.OTHER)
        assigner._update_other_bucket_assignments({7})
        self.assertEqual(assigner._assignments[7], TeamColorAssigner.TEAM_1)
        self.assertEqual(assigner.team_label_change_count, 0)

    def test_other_bucket_recognizes_referee_and_far_colors(self):
        assigner = TeamColorAssigner()
        assigner._team_centers = {
            TeamColorAssigner.TEAM_1: np.asarray((0.0, 0.0, 0.0)),
            TeamColorAssigner.TEAM_2: np.asarray((1.0, 1.0, 1.0)),
        }
        assigner._team_centers_fitted = True
        assigner._referee_color_center = np.asarray((0.5, 0.5, 0.5))
        assigner._features[7] = np.asarray((0.5, 0.5, 0.5))
        assigner._features[8] = np.asarray((0.2, 0.5, 0.8))

        for _ in range(assigner.OTHER_LABEL_HISTORY_SIZE):
            assigner._update_other_bucket_assignments({7, 8})

        self.assertEqual(assigner._assignments[7], TeamColorAssigner.REFEREE)
        self.assertEqual(assigner._assignments[8], TeamColorAssigner.OTHER)

    def test_referee_color_uses_confident_detection_and_torso_crop(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[20:55, 25:75] = (0, 128, 255)
        assigner = TeamColorAssigner()

        assigner.observe_referee_detections(
            frame,
            [
                {"bbox": (0, 0, 100, 100), "confidence": 0.29},
                {"bbox": (0, 0, 100, 100), "confidence": 0.30},
            ],
        )

        self.assertEqual(len(assigner._referee_color_history), 1)
        np.testing.assert_array_equal(
            assigner._referee_color_center,
            TeamColorAssigner._torso_lab_feature(
                frame, (0, 0, 100, 100)
            ),
        )

    @staticmethod
    def _white_keypoint_frame(image_points):
        points = np.rint(image_points).astype(int)
        width = max(1400, int(points[:, 0].max()) + 20)
        height = max(1200, int(points[:, 1].max()) + 20)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for x, y in points:
            cv2.line(
                frame,
                (x - 7, y),
                (x + 7, y),
                (255, 255, 255),
                2,
            )
        return frame

    def test_semantic_class_mapping_uses_model_names(self):
        self.assertEqual(
            map_class_categories(
                {0: "Soccer Ball", 1: "Goal Keeper", 2: "Players", 3: "Official"}
            ),
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        )
        self.assertEqual(
            map_class_categories(
                {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
            ),
            {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        )
        self.assertEqual(map_class_categories({4: "unrecognized"}), {4: "other"})

    def test_class_aware_tracker_is_registered_with_ultralytics(self):
        from ultralytics.trackers.track import TRACKER_MAP

        class_categories = {0: "player", 1: "ball"}
        register_class_aware_bytetrack(class_categories)

        self.assertIs(TRACKER_MAP["classbytetrack"], ClassAwareByteTrack)
        self.assertEqual(ClassAwareByteTrack.class_categories, class_categories)

    def test_tracker_gates_cross_class_matches_but_groups_players_and_goalkeepers(
        self,
    ):
        ClassAwareByteTrack.set_class_categories(
            {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
        )
        tracks = [
            SimpleNamespace(cls=0),
            SimpleNamespace(cls=2),
        ]
        detections = [
            SimpleNamespace(cls=1),
            SimpleNamespace(cls=2),
            SimpleNamespace(cls=3),
        ]
        costs = ClassAwareByteTrack._gate_class_mismatches(
            np.zeros((2, 3), dtype=np.float32), tracks, detections
        )

        np.testing.assert_array_equal(
            costs,
            [
                [
                    0.0,
                    ClassAwareByteTrack.CLASS_MISMATCH_COST,
                    ClassAwareByteTrack.CLASS_MISMATCH_COST,
                ],
                [
                    ClassAwareByteTrack.CLASS_MISMATCH_COST,
                    0.0,
                    ClassAwareByteTrack.CLASS_MISMATCH_COST,
                ],
            ],
        )

    def test_tracker_appearance_cost_prevents_crossing_identity_swap(self):
        ClassAwareByteTrack.set_class_categories({0: "player"})
        tracker = ClassAwareByteTrack(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.10,
                track_buffer=30,
                match_thresh=0.80,
                fuse_score=False,
            )
        )
        team_one_feature = np.asarray((0.55, 0.48, 0.50))
        team_two_feature = np.asarray((0.75, 0.49, 0.58))
        tracker._appearance_history = {
            1: [team_one_feature] * tracker.APPEARANCE_MIN_SAMPLES,
            2: [team_two_feature] * tracker.APPEARANCE_MIN_SAMPLES,
        }
        tracker._current_appearance_features = {
            0: team_two_feature,
            1: team_one_feature,
        }
        tracks = [
            SimpleNamespace(track_id=1, cls=0),
            SimpleNamespace(track_id=2, cls=0),
        ]
        detections = [
            SimpleNamespace(idx=0, cls=0),
            SimpleNamespace(idx=1, cls=0),
        ]
        spatial_costs = np.asarray(((0.10, 0.30), (0.30, 0.10)))

        appearance_costs = tracker._appearance_aware_costs(
            spatial_costs, tracks, detections
        )

        self.assertLess(
            appearance_costs[0, 1] + appearance_costs[1, 0],
            appearance_costs[0, 0] + appearance_costs[1, 1],
        )

    def test_lost_track_spatial_gate_rejects_far_matches_only_when_enabled(self):
        ClassAwareByteTrack.set_class_categories({2: "player"})
        tracker = ClassAwareByteTrack(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.10,
                track_buffer=30,
                match_thresh=0.80,
                fuse_score=False,
            )
        )
        tracker.frame_id = 12
        tracker._last_observed_boxes = {1: ((0.0, 0.0, 20.0, 20.0), 10)}
        lost_track = SimpleNamespace(
            track_id=1,
            state=TrackState.Lost,
            cls=2,
        )
        tracked_track = SimpleNamespace(
            track_id=2,
            state=TrackState.Tracked,
            cls=2,
        )
        detection = SimpleNamespace(
            xyxy=np.asarray((100.0, 0.0, 120.0, 20.0)),
            cls=2,
            idx=0,
        )
        costs = np.zeros((2, 1), dtype=np.float32)

        gated = tracker._gate_lost_track_spatial_costs(
            costs, [lost_track, tracked_track], [detection]
        )

        self.assertEqual(
            gated[0, 0], ClassAwareByteTrack.CLASS_MISMATCH_COST
        )
        self.assertEqual(gated[1, 0], costs[1, 0])
        with patch.object(BYTETracker, "get_dists", return_value=costs):
            gated_result = tracker.get_dists(
                [lost_track, tracked_track], [detection]
            )
        self.assertEqual(
            gated_result[0, 0], ClassAwareByteTrack.CLASS_MISMATCH_COST
        )
        self.assertEqual(gated_result[1, 0], costs[1, 0])
        nearby_detection = SimpleNamespace(
            xyxy=np.asarray((50.0, 0.0, 70.0, 20.0)),
        )
        nearby = tracker._gate_lost_track_spatial_costs(
            costs[:1], [lost_track], [nearby_detection]
        )
        self.assertEqual(nearby[0, 0], costs[0, 0])

        tracker.ENABLE_LOST_TRACK_SPATIAL_GATE = False
        with patch.object(BYTETracker, "get_dists", return_value=costs):
            disabled = tracker.get_dists(
                [lost_track, tracked_track], [detection]
            )
        np.testing.assert_array_equal(disabled, costs)

    def test_tracked_track_spatial_gate_is_opt_in_and_has_no_gap_allowance(self):
            ClassAwareByteTrack.set_class_categories({2: "player"})
            tracker = ClassAwareByteTrack(
                SimpleNamespace(
                    track_high_thresh=0.25,
                    track_low_thresh=0.10,
                    new_track_thresh=0.10,
                    track_buffer=30,
                    match_thresh=0.80,
                    fuse_score=False,
                )
            )
            tracker._last_observed_boxes = {
                1: ((0.0, 0.0, 20.0, 20.0), 12)
            }
            tracked_track = SimpleNamespace(
                track_id=1,
                state=TrackState.Tracked,
                cls=2,
            )
            lost_track = SimpleNamespace(
                track_id=2,
                state=TrackState.Lost,
                cls=2,
            )
            detection = SimpleNamespace(
                xyxy=np.asarray((100.0, 0.0, 120.0, 20.0)),
                cls=2,
                idx=0,
            )
            costs = np.zeros((2, 1), dtype=np.float32)

            self.assertFalse(tracker.ENABLE_TRACKED_TRACK_SPATIAL_GATE)
            with patch.object(BYTETracker, "get_dists", return_value=costs):
                disabled = tracker.get_dists(
                    [tracked_track, lost_track], [detection]
                )
            np.testing.assert_array_equal(disabled, costs)

            tracker.ENABLE_TRACKED_TRACK_SPATIAL_GATE = True
            gated = tracker._gate_tracked_track_spatial_costs(
                costs, [tracked_track, lost_track], [detection]
            )
            self.assertEqual(
                gated[0, 0], ClassAwareByteTrack.CLASS_MISMATCH_COST
            )
            self.assertEqual(gated[1, 0], costs[1, 0])

            with patch.object(BYTETracker, "get_dists", return_value=costs):
                enabled = tracker.get_dists(
                    [tracked_track, lost_track], [detection]
                )
            self.assertEqual(
                enabled[0, 0], ClassAwareByteTrack.CLASS_MISMATCH_COST
            )
            self.assertEqual(enabled[1, 0], costs[1, 0])

    def test_overlapping_referee_detection_gets_a_separate_tracker_id(self):
        ClassAwareByteTrack.set_class_categories(
            {0: "player", 1: "referee"}
        )
        tracker = ClassAwareByteTrack(
            SimpleNamespace(
                track_high_thresh=0.25,
                track_low_thresh=0.10,
                new_track_thresh=0.10,
                track_buffer=30,
                match_thresh=0.80,
                fuse_score=False,
            )
        )
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        player = Boxes(
            np.asarray([[10, 10, 40, 90, 0.9, 0]], dtype=np.float32),
            orig_shape=frame.shape[:2],
        )
        referee = Boxes(
            np.asarray([[10, 10, 40, 90, 0.9, 1]], dtype=np.float32),
            orig_shape=frame.shape[:2],
        )
        first_tracks = tracker.update(player, frame)
        tracker.update(referee, frame)
        second_tracks = tracker.update(referee, frame)

        self.assertEqual(len(first_tracks), 1)
        self.assertEqual(len(second_tracks), 1)
        self.assertNotEqual(first_tracks[0, 4], second_tracks[0, 4])

    def test_ellipse_matches_compact_track_py_geometry_and_team_color(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        color = TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.TEAM_1]
        detection = {
            "tracker_id": 7,
            "global_id": 7,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.TEAM_1,
            "team_color": color,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse", wraps=cv2.ellipse
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.rectangle", wraps=cv2.rectangle
        ) as draw_rectangle, patch(
            "utils.visualization_utils.cv2.putText", wraps=cv2.putText
        ) as put_text:
            draw_annotations(frame, [detection])

        self.assertEqual(draw_ellipse.call_count, 1)
        ellipse = draw_ellipse.call_args.kwargs
        self.assertEqual(ellipse["center"], (110, 140))
        self.assertEqual(ellipse["axes"], (40, 14))
        self.assertEqual(ellipse["angle"], 0.0)
        self.assertEqual(ellipse["startAngle"], -45)
        self.assertEqual(ellipse["endAngle"], 235)
        self.assertEqual(ellipse["color"], color)
        self.assertEqual(ellipse["thickness"], 2)
        self.assertEqual(ellipse["lineType"], cv2.LINE_4)
        self.assertEqual(draw_rectangle.call_args.args[1], (90, 144))
        self.assertEqual(draw_rectangle.call_args.args[2], (130, 164))
        self.assertEqual(put_text.call_args.args[1], "7")

    def test_unknown_goalkeeper_is_displayed_in_goalkeeper_color(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        goalkeeper = {
            "tracker_id": 7,
            "global_id": 11,
            "bbox": (90, 40, 130, 140),
            "object_type": "goalkeeper",
            "player_type": TeamColorAssigner.UNKNOWN_GOALKEEPER,
            "team_color": TeamColorAssigner.NEUTRAL_COLOR,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse"
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.rectangle"
        ) as draw_rectangle, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [goalkeeper])

        self.assertEqual(
            draw_ellipse.call_args.kwargs["color"],
            TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.GOALKEEPER],
        )
        self.assertEqual(
            draw_rectangle.call_args.args[3],
            TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.GOALKEEPER],
        )
        self.assertEqual(put_text.call_args.args[1], "11")

    def test_unresolved_identity_does_not_display_raw_tracker_id(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = {
            "tracker_id": 42,
            "global_id": None,
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.UNKNOWN,
            "team_color": TeamColorAssigner.NEUTRAL_COLOR,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse", wraps=cv2.ellipse
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.rectangle"
        ) as draw_rectangle, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [detection], identity_resolved=True)

        draw_ellipse.assert_not_called()
        draw_rectangle.assert_not_called()
        put_text.assert_not_called()

    def test_unknown_team_is_not_displayed_even_with_a_global_id(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = {
            "tracker_id": 42,
            "global_id": 11,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.UNKNOWN,
            "team_color": TeamColorAssigner.NEUTRAL_COLOR,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse"
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [detection])

        draw_ellipse.assert_not_called()
        put_text.assert_not_called()

    def test_resolved_players_and_confirmed_other_tracks_are_displayable(self):
        known_player = {
            "object_type": "player",
            "global_id": 7,
            "player_type": TeamColorAssigner.TEAM_1,
        }
        self.assertTrue(is_displayable_player(known_player))
        unresolved_team_player = {
            "object_type": "player",
            "global_id": 12,
            "player_type": TeamColorAssigner.OTHER,
        }
        self.assertTrue(is_displayable_player(unresolved_team_player))
        known_goalkeeper = {
            "object_type": "goalkeeper",
            "global_id": 11,
            "player_type": TeamColorAssigner.TEAM_2,
        }
        self.assertTrue(is_displayable_player(known_goalkeeper))
        for detection in (
            {
                "object_type": "player",
                "global_id": None,
                "player_type": TeamColorAssigner.TEAM_1,
            },
            {
                "object_type": "player",
                "global_id": 7,
                "player_type": TeamColorAssigner.UNKNOWN,
            },
            {
                "object_type": "player",
                "global_id": None,
                "player_type": TeamColorAssigner.OTHER,
            },
            {
                "object_type": "goalkeeper",
                "global_id": None,
                "player_type": TeamColorAssigner.UNKNOWN_GOALKEEPER,
            },
            {
                "object_type": "referee",
                "global_id": None,
                "player_type": TeamColorAssigner.REFEREE,
            },
        ):
            with self.subTest(detection=detection):
                self.assertFalse(is_displayable_player(detection))
        self.assertTrue(
            is_displayable_player(
                {
                    "object_type": "goalkeeper",
                    "global_id": 11,
                    "player_type": TeamColorAssigner.UNKNOWN_GOALKEEPER,
                }
            )
        )

    def test_goalkeeper_with_confident_team_is_drawn_in_team_color(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        goalkeeper = {
            "tracker_id": 17,
            "global_id": 11,
            "bbox": (90, 40, 130, 140),
            "object_type": "goalkeeper",
            "player_type": TeamColorAssigner.TEAM_2,
            "team_color": TeamColorAssigner.TEAM_COLORS[
                TeamColorAssigner.TEAM_2
            ],
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse"
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [goalkeeper])

        self.assertEqual(
            draw_ellipse.call_args.kwargs["color"],
            TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.TEAM_2],
        )
        self.assertEqual(put_text.call_args.args[1], "11")

    def test_confirmed_other_player_is_drawn_with_neutral_id(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        player = {
            "global_id": 14,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.OTHER,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse"
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [player])

        self.assertEqual(
            draw_ellipse.call_args.kwargs["color"],
            TeamColorAssigner.NEUTRAL_COLOR,
        )
        self.assertEqual(put_text.call_args.args[1], "14")

    def test_conflicted_team_player_is_drawn_with_neutral_id(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        player = {
            "global_id": 31,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.TEAM_CONFLICT,
        }
        with patch(
            "utils.visualization_utils.cv2.ellipse"
        ) as draw_ellipse, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            draw_annotations(frame, [player])

        self.assertEqual(
            draw_ellipse.call_args.kwargs["color"],
            TeamColorAssigner.NEUTRAL_COLOR,
        )
        self.assertEqual(put_text.call_args.args[1], "31")

    def test_global_id_is_rendered_instead_of_temporary_tracker_id(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = {
            "tracker_id": 42,
            "global_id": 7,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.TEAM_1,
            "team_color": TeamColorAssigner.TEAM_COLORS[
                TeamColorAssigner.TEAM_1
            ],
        }
        with patch("utils.visualization_utils.cv2.putText", wraps=cv2.putText) as put_text:
            draw_annotations(frame, [detection])
        self.assertEqual(put_text.call_args.args[1], "7")

    def test_referee_has_no_ellipse_or_id_overlay(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        referee = {
            "tracker_id": 9,
            "bbox": (30, 10, 50, 80),
            "object_type": "referee",
            "player_type": TeamColorAssigner.REFEREE,
        }
        with patch("utils.visualization_utils.cv2.ellipse") as draw_ellipse, patch(
            "utils.visualization_utils.cv2.putText"
        ) as put_text:
            annotated = draw_annotations(frame, [referee])
        draw_ellipse.assert_not_called()
        put_text.assert_not_called()
        np.testing.assert_array_equal(annotated, frame)

    def test_annotation_can_draw_into_frame_when_caller_owns_buffer(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        detection = {
            "tracker_id": 7,
            "global_id": 7,
            "object_type": "player",
            "bbox": (90, 40, 130, 140),
            "player_type": TeamColorAssigner.TEAM_1,
            "team_color": TeamColorAssigner.TEAM_COLORS[
                TeamColorAssigner.TEAM_1
            ],
        }

        annotated = draw_annotations(
            frame, [detection], copy_frame=False
        )

        self.assertIs(annotated, frame)
        self.assertTrue(np.any(frame))

    def test_radar_static_background_is_cached_and_not_mutated(self):
        renderer = CameraRadarRenderer(output_size=(420, 245))
        background_before = renderer._background.copy()

        first = renderer.render(
            {1: (640.0, 360.0, (0, 0, 255), 1)},
            frame_size=(720, 1280, 3),
        )
        second = renderer.render(
            {2: (320.0, 180.0, (255, 0, 0), 2)},
            frame_size=(720, 1280, 3),
        )

        self.assertFalse(np.array_equal(first, second))
        np.testing.assert_array_equal(renderer._background, background_before)

    def test_valid_nonstandard_team_hues_are_not_misclassified_as_referees(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        team_hues = (31, 81, 31, 81, 31)
        for index, hue in enumerate(team_hues):
            kit_hsv = np.asarray([[[hue, 210, 210]]], dtype=np.uint8)
            kit_bgr = cv2.cvtColor(kit_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            frame[20:80, index * 30 : (index + 1) * 30] = kit_bgr
        detections = [
            {"tracker_id": 1, "object_type": "player", "bbox": (0, 10, 30, 90)},
            {"tracker_id": 2, "object_type": "player", "bbox": (30, 10, 60, 90)},
            {"tracker_id": 3, "object_type": "player", "bbox": (60, 10, 90, 90)},
            {"tracker_id": 4, "object_type": "player", "bbox": (90, 10, 120, 90)},
            {"tracker_id": 5, "object_type": "player", "bbox": (120, 10, 150, 90)},
        ]
        assigner = TeamColorAssigner()
        for _ in range(12):
            assignments = assigner.update(frame, detections)

        self.assertEqual(assignments[1], assignments[3])
        self.assertEqual(assignments[2], assignments[4])
        self.assertEqual(assignments[1], assignments[5])
        self.assertNotEqual(assignments[1], assignments[2])
        self.assertFalse(assigner.last_referee_candidates)
        self.assertNotIn(TeamColorAssigner.REFEREE, assignments.values())

    def test_kit_color_sampling_ignores_pitch_around_player_box(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:] = (0, 128, 0)
        boxes = [(10, 10, 50, 90), (70, 10, 110, 90)]
        colors = [(0, 0, 255), (255, 0, 0)]
        features = []
        for (x1, y1, x2, y2), color in zip(boxes, colors):
            center_x = (x1 + x2) // 2
            frame[20:70, center_x - 5 : center_x + 5] = color
            features.append(TeamColorAssigner._jersey_feature(frame, (x1, y1, x2, y2)))

        self.assertGreater(np.linalg.norm(features[0] - features[1]), 0.20)

    def test_kit_color_sampling_ignores_pitch_without_kit_color_evidence(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[:] = (0, 128, 0)
        bbox = (40, 10, 80, 90)
        self.assertIsNone(TeamColorAssigner._jersey_feature(frame, bbox))

        pitch_hsv = np.full((1, 1, 3), (42, 180, 200), dtype=np.uint8)
        pitch_color = cv2.cvtColor(pitch_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        frame[:] = pitch_color
        self.assertIsNone(TeamColorAssigner._jersey_feature(frame, bbox))

        frame[30:35, 52:68] = (0, 255, 255)
        self.assertIsNone(TeamColorAssigner._jersey_feature(frame, bbox))

        frame[30:60, 52:68] = (0, 255, 255)
        kit_feature = TeamColorAssigner._jersey_feature(frame, bbox)
        self.assertIsNotNone(kit_feature)
        expected = cv2.cvtColor(
            np.asarray([[[0, 255, 255]]], dtype=np.uint8),
            cv2.COLOR_BGR2LAB,
        )[0, 0].astype(np.float64) / 255.0
        self.assertLess(np.linalg.norm(kit_feature - expected), 0.08)

    def test_team_clustering_does_not_reclassify_kit_outliers_as_referees(self):
        frame = np.zeros((120, 300, 3), dtype=np.uint8)
        hsv_colors = [(70, 210, 210)] * 4 + [(140, 210, 210)] * 4 + [
            (105, 210, 210)
        ]
        detections = []
        for tracker_id, hsv_color in enumerate(hsv_colors, start=1):
            kit_bgr = cv2.cvtColor(
                np.asarray([[hsv_color]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )[0, 0]
            x1 = (tracker_id - 1) * 30
            frame[20:80, x1 : x1 + 30] = kit_bgr
            detections.append(
                {
                    "tracker_id": tracker_id,
                    "object_type": "player",
                    "bbox": (x1, 10, x1 + 30, 90),
                }
            )

        assigner = TeamColorAssigner()
        for _ in range(12):
            assignments = assigner.update(frame, detections)

        self.assertEqual(assignments[1], assignments[4])
        self.assertEqual(assignments[5], assignments[8])
        self.assertNotEqual(assignments[1], assignments[5])
        self.assertIn(assignments[9], (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2))
        self.assertFalse(assigner.last_referee_candidates)
        self.assertEqual(
            set(assigner.team_colors),
            {
                TeamColorAssigner.TEAM_1,
                TeamColorAssigner.TEAM_2,
                TeamColorAssigner.GOALKEEPER,
                TeamColorAssigner.BALL,
            },
        )

    def test_unknown_goalkeeper_remains_neutral_without_clear_team_match(self):
        frame = np.zeros((100, 150, 3), dtype=np.uint8)
        frame[20:80, 0:30] = (0, 0, 255)
        frame[20:80, 30:60] = (255, 0, 0)
        frame[20:80, 60:90] = (0, 0, 255)
        frame[20:80, 90:120] = (255, 0, 0)
        frame[20:80, 120:150] = (0, 255, 255)
        detections = [
            {"tracker_id": 1, "object_type": "player", "bbox": (0, 10, 30, 90)},
            {"tracker_id": 2, "object_type": "player", "bbox": (30, 10, 60, 90)},
            {"tracker_id": 3, "object_type": "player", "bbox": (60, 10, 90, 90)},
            {"tracker_id": 4, "object_type": "player", "bbox": (90, 10, 120, 90)},
            {"tracker_id": 5, "object_type": "goalkeeper", "bbox": (120, 10, 150, 90)},
        ]
        assigner = TeamColorAssigner()
        assignments = assigner.update(frame, detections)
        self.assertEqual(assignments[5], TeamColorAssigner.UNKNOWN_GOALKEEPER)

    def test_goalkeeper_stays_unknown_even_when_kit_matches_a_team(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[20:80, 0:30] = (0, 0, 255)
        frame[20:80, 30:60] = (255, 0, 0)
        frame[20:80, 60:90] = (0, 0, 255)
        frame[20:80, 90:120] = (255, 0, 0)
        frame[20:80, 120:150] = (0, 0, 255)
        detections = [
            {"tracker_id": 1, "object_type": "player", "bbox": (0, 10, 30, 90)},
            {"tracker_id": 2, "object_type": "player", "bbox": (30, 10, 60, 90)},
            {"tracker_id": 3, "object_type": "player", "bbox": (60, 10, 90, 90)},
            {"tracker_id": 4, "object_type": "player", "bbox": (90, 10, 120, 90)},
            {"tracker_id": 5, "object_type": "goalkeeper", "bbox": (120, 10, 150, 90)},
        ]
        assigner = TeamColorAssigner()
        assignments = {}
        for _ in range(12):
            assignments = assigner.update(frame, detections)
        self.assertEqual(
            assignments[5], TeamColorAssigner.UNKNOWN_GOALKEEPER
        )
        self.assertNotIn(5, assigner._assignments)
        self.assertLessEqual(
            min(assigner.last_goalkeeper_distances[5].values()),
            TeamColorAssigner.MAX_GOALKEEPER_COLOR_DISTANCE,
        )

    def test_goalkeeper_stays_unknown_with_clear_nearby_hue(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        hsv_colors = (0, 120, 0, 120, 18)
        detections = []
        for tracker_id, hue in enumerate(hsv_colors, start=1):
            color = cv2.cvtColor(
                np.asarray([[[hue, 220, 220]]], dtype=np.uint8),
                cv2.COLOR_HSV2BGR,
            )[0, 0]
            x1 = (tracker_id - 1) * 30
            frame[20:80, x1 : x1 + 30] = color
            detections.append(
                {
                    "tracker_id": tracker_id,
                    "object_type": (
                        "goalkeeper" if tracker_id == 5 else "player"
                    ),
                    "bbox": (x1, 10, x1 + 30, 90),
                }
            )

        assigner = TeamColorAssigner()
        assignments = {}
        for _ in range(12):
            assignments = assigner.update(frame, detections)

        self.assertEqual(
            assignments[5], TeamColorAssigner.UNKNOWN_GOALKEEPER
        )
        self.assertNotIn(5, assigner._assignments)
        self.assertLess(
            min(assigner.last_goalkeeper_distances[5].values()),
            TeamColorAssigner.MAX_GOALKEEPER_COLOR_DISTANCE,
        )

    def test_team_assignment_uses_kit_color_not_pitch_half_or_nearby_players(self):
        frame = np.zeros((100, 240, 3), dtype=np.uint8)
        colors = [
            (0, 0, 255),
            (255, 0, 0),
            (0, 0, 255),
            (255, 0, 0),
        ]
        detections = []
        for tracker_id, color in enumerate(colors, start=1):
            x1 = (tracker_id - 1) * 60
            frame[20:80, x1 : x1 + 30] = color
            detections.append(
                {
                    "tracker_id": tracker_id,
                    "object_type": "player",
                    "bbox": (x1, 10, x1 + 30, 90),
                }
            )
        assigner = TeamColorAssigner()
        for _ in range(12):
            assignments = assigner.update(frame, detections)
        self.assertEqual(assignments[1], assignments[3])
        self.assertEqual(assignments[2], assignments[4])
        self.assertNotEqual(assignments[1], assignments[2])

    def test_green_kit_with_white_stripes_is_stable_against_cream_team(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        green = cv2.cvtColor(
            np.asarray([[[147, 120, 132]]], dtype=np.uint8),
            cv2.COLOR_LAB2BGR,
        )[0, 0]
        cream = cv2.cvtColor(
            np.asarray([[[189, 123, 149]]], dtype=np.uint8),
            cv2.COLOR_LAB2BGR,
        )[0, 0]
        detections = []
        for tracker_id in range(1, 7):
            x1 = (tracker_id - 1) * 40
            is_green = tracker_id in (1, 2, 3)
            frame[20:85, x1 : x1 + 32] = green if is_green else cream
            if is_green:
                for stripe_x in range(x1 + 4, x1 + 32, 8):
                    frame[28:65, stripe_x : stripe_x + 3] = (245, 245, 245)
            detections.append(
                {
                    "tracker_id": tracker_id,
                    "object_type": "player",
                    "bbox": (x1, 10, x1 + 32, 95),
                }
            )

        assigner = TeamColorAssigner()
        for _ in range(12):
            assignments = assigner.update(frame, detections)

        self.assertEqual(assignments[1], assignments[2])
        self.assertEqual(assignments[2], assignments[3])
        self.assertEqual(assignments[4], assignments[5])
        self.assertEqual(assignments[5], assignments[6])
        self.assertNotEqual(assignments[1], assignments[4])
        self.assertTrue(assigner._team_centers_fitted)
        self.assertEqual(assignments[1], TeamColorAssigner.TEAM_1)
        self.assertEqual(assignments[4], TeamColorAssigner.TEAM_2)

    def test_team_torso_feature_keeps_dark_green_pixels(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        grass = cv2.cvtColor(
            np.asarray([[[60, 150, 170]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        dark_green = cv2.cvtColor(
            np.asarray([[[60, 180, 70]]], dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        frame[:] = grass
        frame[32:48, 40:60] = dark_green

        feature = TeamColorAssigner._team_torso_lab_feature(
            frame, (30, 10, 70, 90)
        )
        referee_feature = TeamColorAssigner._torso_lab_feature(
            frame, (30, 10, 70, 90)
        )
        dark_green_lab = cv2.cvtColor(
            np.asarray([[dark_green]], dtype=np.uint8), cv2.COLOR_BGR2LAB
        )[0, 0]

        self.assertIsNotNone(feature)
        np.testing.assert_allclose(feature * 255.0, dark_green_lab, atol=1.0)
        self.assertIsNone(referee_feature)

    def test_team_colors_fit_once_and_one_bad_crop_cannot_flip_a_track(self):
        frame = np.zeros((100, 240, 3), dtype=np.uint8)
        colors = [(0, 0, 255), (255, 0, 0), (0, 0, 255), (255, 0, 0)]
        detections = []
        for tracker_id, color in enumerate(colors, start=1):
            x1 = (tracker_id - 1) * 60
            frame[20:80, x1 : x1 + 30] = color
            detections.append(
                {
                    "tracker_id": tracker_id,
                    "object_type": "player",
                    "bbox": (x1, 10, x1 + 30, 90),
                }
            )

        assigner = TeamColorAssigner()
        for _ in range(12):
            assignments = assigner.update(frame, detections)
        frozen_centers = {
            team: center.copy() for team, center in assigner._team_centers.items()
        }
        original_team = assignments[1]
        frame[20:80, 0:30] = colors[1]
        for _ in range(1):
            assignments = assigner.update(frame, detections)

        self.assertEqual(assignments[1], original_team)
        self.assertEqual(assigner.team_label_change_count, 0)
        for team, center in frozen_centers.items():
            np.testing.assert_array_equal(assigner._team_centers[team], center)

    def test_referee_class_history_locks_track_and_hides_new_tracks(self):
        manager = RefereeTrackManager()
        detections = [{"tracker_id": 7, "object_type": "player"}]
        for _ in range(2):
            visible_ids, _ = manager.update(detections)
            self.assertNotIn(7, visible_ids)
        visible_ids, _ = manager.update(detections)
        self.assertIn(7, visible_ids)

        referee_detection = [{"tracker_id": 7, "object_type": "referee"}]
        _, counts = manager.update(referee_detection)
        self.assertEqual(counts["detected_by_class"], 1)
        manager.update(detections)
        _, counts = manager.update(referee_detection)
        self.assertEqual(counts["detected_by_class"], 1)
        visible_ids, _ = manager.update(detections)
        self.assertIn(7, visible_ids)
        _, counts = manager.update(referee_detection)
        self.assertEqual(counts["detected_by_class"], 1)
        visible_ids, counts = manager.update(detections)
        self.assertEqual(
            manager.referee_reason[7], "reclassified_by_class_history"
        )
        self.assertNotIn(7, visible_ids)
        self.assertEqual(counts["reclassified_by_class_history"], 1)
        manager.mark_referee(7, "rejected_by_kit_color")
        self.assertEqual(
            manager.referee_reason[7], "reclassified_by_class_history"
        )

    def test_transient_referee_label_does_not_lock_player_track(self):
        manager = RefereeTrackManager()
        for observation in range(8):
            object_type = "referee" if observation == 3 else "player"
            visible_ids, _ = manager.update(
                [{"tracker_id": 12, "object_type": object_type}]
            )

        self.assertNotIn(12, manager.referee_reason)
        self.assertIn(12, visible_ids)
        self.assertEqual(manager.track_age[12], 8)

    def test_pitch_homography_projects_keypoint_and_player(self):
        config = SoccerPitchConfiguration()
        image_points = config.vertices / 10.0 + np.array([100.0, 50.0])
        projector = PitchRadarProjector()
        matrix, loo_error_m, inliers, ratio = projector._estimate(
            image_points,
            np.ones(32, dtype=np.float32),
            self._white_keypoint_frame(image_points),
        )

        self.assertIsNotNone(matrix)
        self.assertLess(loo_error_m, 0.01)
        self.assertGreaterEqual(inliers, projector.MIN_INLIERS)
        self.assertGreaterEqual(ratio, projector.MIN_INLIER_RATIO)
        projected = projector.project_points(matrix, [(400, 200)])[0]
        np.testing.assert_allclose(projected, [3000, 1500], atol=5)

    def test_pitch_homography_accepts_five_consistent_inliers(self):
        config = SoccerPitchConfiguration()
        indices = np.array([0, 5, 13, 16, 24, 29])
        image_points = config.vertices / 10.0 + np.array([100.0, 50.0])
        image_points[indices[1]] += (300.0, 250.0)
        confidence = np.zeros(32, dtype=np.float32)
        confidence[indices] = 1.0

        matrix, loo_error_m, inliers, ratio = PitchRadarProjector()._estimate(
            image_points,
            confidence,
            self._white_keypoint_frame(image_points),
        )

        self.assertIsNotNone(matrix)
        self.assertEqual(inliers, 5)
        self.assertEqual(ratio, 5 / 6)
        self.assertIsNotNone(loo_error_m)

    def test_pitch_homography_accepts_four_verified_non_collinear_points(self):
        config = SoccerPitchConfiguration()
        indices = np.asarray([0, 5, 24, 29])
        image_points = config.vertices / 10.0 + np.asarray([100.0, 50.0])
        confidence = np.zeros(32, dtype=np.float32)
        confidence[indices] = 1.0
        projector = PitchRadarProjector()

        matrix, loo_error_m, inliers, _ = projector._estimate(
            image_points,
            confidence,
            self._white_keypoint_frame(image_points),
        )

        self.assertIsNotNone(matrix)
        self.assertEqual(inliers, 4)
        self.assertIsNone(loo_error_m)
        self.assertIsNotNone(projector.last_calibration_metrics["chamfer_px"])

    def test_pitch_homography_rejects_collinear_verified_points(self):
        projector = PitchRadarProjector()
        image_points = np.zeros((32, 2), dtype=np.float32)
        indices = np.asarray([0, 13, 16, 29])
        image_points[indices] = np.column_stack(
            (np.arange(4, dtype=np.float32) * 20 + 100, np.full(4, 200))
        )
        confidence = np.zeros(32, dtype=np.float32)
        confidence[indices] = 1.0

        matrix, _, inliers, _ = projector._estimate(
            image_points,
            confidence,
            self._white_keypoint_frame(image_points[indices]),
        )

        self.assertIsNone(matrix)
        self.assertEqual(inliers, 0)

    def test_high_confidence_keypoint_on_grass_is_rejected(self):
        projector = PitchRadarProjector()
        points = np.asarray([[50, 50], [100, 100]], dtype=np.float32)
        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (47, 47), (53, 53), (255, 255, 255), cv2.FILLED)
        cv2.line(frame, (93, 100), (107, 100), (255, 255, 255), 2)

        verified = projector._verify_keypoints(
            frame, points, np.ones(2, dtype=np.float32)
        )

        np.testing.assert_array_equal(verified, [False, True])
        self.assertEqual(projector.last_keypoint_diagnostics["dropped"], 1)

    def test_pitch_radar_preserves_pitch_aspect_ratio_and_shows_no_reference(self):
        radar = PitchRadarProjector().draw_radar(
            {},
            output_size=(480, 240),
            has_pitch_reference=False,
        )
        self.assertEqual(radar.shape, (240, 480, 3))
        self.assertTrue(np.any(radar != np.full_like(radar, (55, 117, 74))))

    def test_pitch_radar_draws_notebook_pitch_geometry_and_pitch_coordinates(self):
        team_color = TeamColorAssigner.TEAM_COLORS[TeamColorAssigner.TEAM_1]
        radar = PitchRadarProjector().draw_radar(
            {1: (3000, 2000, team_color, 1)},
            output_size=(480, 280),
        )

        self.assertTupleEqual(tuple(radar[80, 120]), team_color)
        self.assertFalse(np.array_equal(radar[140, 277], (55, 117, 74)))
        self.assertFalse(np.array_equal(radar[58, 81], (55, 117, 74)))

    def test_debug_overlay_projects_pitch_lines_and_circle_back_to_video(self):
        frame = np.zeros((800, 1400, 3), dtype=np.uint8)
        image_to_pitch = np.asarray(
            [[10.0, 0.0, -1000.0], [0.0, 10.0, -500.0], [0.0, 0.0, 1.0]]
        )
        overlay = PitchRadarProjector().draw_debug_overlay(
            frame, image_to_pitch
        )
        self.assertTupleEqual(
            tuple(overlay[400, 700]),
            (255, 0, 255),
        )
        self.assertTrue(np.any(np.all(overlay == (255, 255, 0), axis=2)))

    def test_pitch_radar_renders_unknown_player_in_neutral_color(self):
        projector = PitchRadarProjector()
        radar = projector.draw_radar(
            {
                17: (
                    6000,
                    3500,
                    TeamColorAssigner.NEUTRAL_COLOR,
                    17,
                )
            },
            output_size=(420, 245),
        )
        self.assertTrue(
            np.any(np.all(radar == TeamColorAssigner.NEUTRAL_COLOR, axis=2))
        )

    def test_dashboard_writer_size_includes_bottom_panel(self):
        self.assertEqual(output_size(1280, 720), (1280, 720 + PANEL_HEIGHT))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((230, 460, 3), dtype=np.uint8)
        from utils.possession_utils import PossessionEstimator

        dashboard = compose_dashboard(
            frame, radar, PossessionEstimator(), layout="panel"
        )
        self.assertEqual(dashboard.shape, (720 + PANEL_HEIGHT, 1280, 3))

    def test_dashboard_possession_legend_does_not_label_the_ball(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((245, 420, 3), dtype=np.uint8)
        from utils.possession_utils import PossessionEstimator

        with patch("utils.dashboard.cv2.putText", wraps=cv2.putText) as draw_text:
            compose_dashboard(
                frame, radar, PossessionEstimator(), layout="panel"
            )

        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("Team 1", labels)
        self.assertIn("Team 2", labels)
        self.assertIn("Goalkeeper", labels)
        self.assertNotIn("Ball", labels)
        self.assertFalse(any("Loose" in label or "Unknown" in label for label in labels))

    def test_dashboard_shows_live_team_pass_counts(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((245, 420, 3), dtype=np.uint8)
        from utils.possession_utils import PossessionEstimator

        pass_counts = {
            TeamColorAssigner.TEAM_1: {
                "attempted": 3,
                "completed": 2,
                "incomplete": 1,
                "unknown": 0,
            },
            TeamColorAssigner.TEAM_2: {
                "attempted": 1,
                "completed": 1,
                "incomplete": 0,
                "unknown": 0,
            },
        }
        with patch("utils.dashboard.cv2.putText", wraps=cv2.putText) as draw_text:
            dashboard = compose_dashboard(
                frame,
                radar,
                PossessionEstimator(),
                layout="panel",
                pass_counts=pass_counts,
            )

        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertNotIn("LIVE PASSES", labels)
        self.assertIn("PASSES", labels)
        self.assertIn("Attempted", labels)
        self.assertIn("Completed", labels)
        self.assertIn("Incomplete", labels)
        self.assertFalse(
            any("Loose" in label or "Unknown" in label for label in labels)
        )
        self.assertIn("3", labels)
        self.assertIn("2", labels)
        self.assertIn("1", labels)
        pass_heading = next(
            call for call in draw_text.call_args_list if call.args[1] == "PASSES"
        )
        self.assertGreater(pass_heading.args[2][0], frame.shape[1] - 340)
        self.assertLess(pass_heading.args[2][0], frame.shape[1])
        self.assertEqual(pass_heading.args[2][1], 46)
        panel_background = dashboard[20, frame.shape[1] - 327]
        self.assertTrue(np.all(panel_background > 0))
        self.assertTrue(np.all(panel_background < 30))
        self.assertTrue(
            np.array_equal(
                dashboard[:720, : frame.shape[1] - 340],
                frame[:, : frame.shape[1] - 340],
            )
        )

    def test_overlay_dashboard_shows_live_team_pass_counts(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((245, 420, 3), dtype=np.uint8)
        from utils.possession_utils import PossessionEstimator

        pass_counts = {
            TeamColorAssigner.TEAM_1: {
                "attempted": 2,
                "completed": 1,
                "incomplete": 1,
            },
            TeamColorAssigner.TEAM_2: {
                "attempted": 1,
                "completed": 1,
                "incomplete": 0,
            },
        }
        with patch("utils.dashboard.cv2.putText", wraps=cv2.putText) as draw_text:
            overlay = compose_dashboard(
                frame,
                radar,
                PossessionEstimator(),
                layout="overlay",
                pass_counts=pass_counts,
            )

        labels = [call.args[1] for call in draw_text.call_args_list]
        self.assertNotIn("LIVE PASSES", labels)
        self.assertIn("PASSES", labels)
        self.assertIn("Attempted", labels)
        self.assertIn("Completed", labels)
        self.assertIn("Incomplete", labels)
        self.assertIn("2", labels)
        self.assertIn("1", labels)
        self.assertFalse(
            any("Loose" in label or "Unknown" in label for label in labels)
        )
        self.assertEqual(overlay.shape, frame.shape)
        pass_heading = next(
            call for call in draw_text.call_args_list if call.args[1] == "PASSES"
        )
        self.assertGreater(pass_heading.args[2][0], frame.shape[1] - 340)
        self.assertLess(pass_heading.args[2][0], frame.shape[1])

    def test_dashboard_overlay_keeps_frame_size_and_uses_team_control_colors(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        radar = np.zeros((245, 420, 3), dtype=np.uint8)
        colors = {
            TeamColorAssigner.TEAM_1: (0, 255, 0),
            TeamColorAssigner.TEAM_2: (255, 0, 255),
        }
        estimator = PossessionEstimator()
        for _ in range(4):
            estimator.update(
                (100, 100),
                [{
                    "tracker_id": 1,
                    "player_type": TeamColorAssigner.TEAM_1,
                    "foot_xy": (100, 100),
                }],
            )

        overlay = compose_dashboard(
            frame, radar, estimator, layout="overlay", team_colors=colors
        )

        self.assertEqual(overlay.shape, frame.shape)
        self.assertTrue(np.any(np.all(overlay == colors[TeamColorAssigner.TEAM_1], axis=2)))
        np.testing.assert_array_equal(overlay[100:500, 100:900], frame[100:500, 100:900])


if __name__ == "__main__":
    unittest.main()
