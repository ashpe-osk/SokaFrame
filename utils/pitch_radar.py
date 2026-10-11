"""Keypoint homography validation and radar rendering in pitch coordinates."""

from collections import defaultdict, deque
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

@dataclass(frozen=True)
class SoccerPitchConfiguration:
    width: int = 7000
    length: int = 12000
    penalty_box_width: int = 4100
    penalty_box_length: int = 2015
    goal_box_width: int = 1832
    goal_box_length: int = 550
    centre_circle_radius: int = 915
    penalty_spot_distance: int = 1100

    @property
    def vertices(self):
        w, length = self.width, self.length
        pbw, pbl = self.penalty_box_width, self.penalty_box_length
        gbw, gbl = self.goal_box_width, self.goal_box_length
        spot, radius = self.penalty_spot_distance, self.centre_circle_radius
        return np.asarray(
            [
                (0, 0), (0, (w - pbw) / 2), (0, (w - gbw) / 2),
                (0, (w + gbw) / 2), (0, (w + pbw) / 2), (0, w),
                (gbl, (w - gbw) / 2), (gbl, (w + gbw) / 2),
                (spot, w / 2), (pbl, (w - pbw) / 2),
                (pbl, (w - gbw) / 2), (pbl, (w + gbw) / 2),
                (pbl, (w + pbw) / 2), (length / 2, 0),
                (length / 2, w / 2 - radius),
                (length / 2, w / 2 + radius), (length / 2, w),
                (length - pbl, (w - pbw) / 2),
                (length - pbl, (w - gbw) / 2),
                (length - pbl, (w + gbw) / 2),
                (length - pbl, (w + pbw) / 2),
                (length - spot, w / 2),
                (length - gbl, (w - gbw) / 2),
                (length - gbl, (w + gbw) / 2),
                (length, 0), (length, (w - pbw) / 2),
                (length, (w - gbw) / 2), (length, (w + gbw) / 2),
                (length, (w + pbw) / 2), (length, w),
                (length / 2 - radius, w / 2),
                (length / 2 + radius, w / 2),
            ],
            dtype=np.float32,
        )

    @property
    def edges(self):
        return np.asarray(
            [
                (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
                (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
                (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
                (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
                (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
                (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
            ],
            dtype=np.int32,
        ) - 1


class PitchRadarProjector:
    MIN_KEYPOINTS = 4
    MIN_INLIERS = 4
    MIN_INLIER_RATIO = 0.35
    RANSAC_REPROJECTION_CM = 50.0
    MAX_REUSE_FRAMES = 10
    MAX_TRACK_JUMP_CM = 2500.0
    SMOOTHING_WINDOW = 5
    KEYPOINT_CONFIDENCE = 0.5
    WHITE_LINE_WINDOW = 15
    MIN_WHITE_PIXELS = 8
    WHITE_LINE_MAX_SATURATION = 80
    WHITE_LINE_MIN_VALUE = 160
    MAX_KEYPOINT_TO_LINE_PX = 5.0
    MAX_LINE_PIXEL_DEVIATION_PX = 1.8
    MIN_LINE_SUPPORT_SPAN_PX = 8.0

    def __init__(self, configuration=None, *, profile_timings=False):
        self.configuration = configuration or SoccerPitchConfiguration()
        self.profile_timings = profile_timings
        self.last_timings = {}
        self.homography = None
        self.last_good_frame = -1
        self.frame_index = -1
        self.previous_histogram = None
        self.previous_positions = {}
        self.smoothed_positions = defaultdict(
            lambda: deque(maxlen=self.SMOOTHING_WINDOW)
        )
        self.accepted_frames = 0
        self.rejected_frames = 0
        self.stale_frames = 0
        self.sanity_passed = 0
        self.sanity_failed = 0
        self.sanity_unavailable = 0
        self.last_keypoint_diagnostics = {
            "confidence_candidates": 0,
            "verified": 0,
            "dropped": 0,
        }
        self.last_verified_keypoints = None
        self.last_calibration_metrics = {
            "loo_error_m": None,
            "chamfer_px": None,
        }

    def _record_timing(self, name, started):
        if started is not None:
            self.last_timings[name] = (
                self.last_timings.get(name, 0.0)
                + perf_counter() - started
            )

    def update(self, keypoint_xy, keypoint_confidence, tracked_points, frame):
        self.last_timings = {}
        started = perf_counter() if self.profile_timings else None
        self.frame_index += 1
        scene_cut = self._is_scene_cut(frame)
        self._record_timing("scene_cut_detection", started)
        if scene_cut:
            self.homography = None
            self.last_good_frame = -1
            self.previous_positions.clear()
            self.smoothed_positions.clear()

        self.last_verified_keypoints = None
        matrix, _, inlier_count, inlier_ratio = self._estimate(
            keypoint_xy, keypoint_confidence, frame
        )
        started = perf_counter() if self.profile_timings else None
        if matrix is not None and not self._most_players_project_to_pitch(
            matrix, tracked_points
        ):
            matrix = None
        accepted = matrix is not None
        if accepted and not self._positions_are_stable(matrix, tracked_points):
            matrix = None
            accepted = False
        self._record_timing("projected_player_validation", started)

        if accepted:
            self.homography = matrix
            self.last_good_frame = self.frame_index
            self.accepted_frames += 1
            status = "accepted"
        elif (
            self.homography is not None
            and self.frame_index - self.last_good_frame <= self.MAX_REUSE_FRAMES
        ):
            matrix = self.homography
            self.stale_frames += 1
            status = "stale"
        else:
            self.rejected_frames += 1
            status = "rejected"

        if (
            status == "stale"
            and self.last_calibration_metrics["chamfer_px"] is None
        ):
            started = perf_counter() if self.profile_timings else None
            self.last_calibration_metrics["chamfer_px"] = (
                self._template_chamfer_px(frame, matrix)
            )
            self._record_timing("pitch_line_agreement", started)

        projected = {}
        if matrix is not None:
            started = perf_counter() if self.profile_timings else None
            for track_id, point in tracked_points.items():
                destination = self.project_points(matrix, [point])[0]
                if self._inside_pitch(destination):
                    self.smoothed_positions[track_id].append(destination)
                    projected[track_id] = np.mean(
                        self.smoothed_positions[track_id], axis=0
                    )
            self.previous_positions = {
                track_id: np.asarray(point, dtype=np.float32)
                for track_id, point in projected.items()
            }
            self._record_timing("player_projection", started)

        started = perf_counter() if self.profile_timings else None
        sanity = self._check_halfway_sanity(
            keypoint_xy, keypoint_confidence, tracked_points, projected
        )
        self._record_timing("halfway_sanity", started)
        if sanity is True:
            self.sanity_passed += 1
        elif sanity is False:
            self.sanity_failed += 1
        else:
            self.sanity_unavailable += 1
        return projected, {
            "status": status,
            "loo_error_m": self.last_calibration_metrics["loo_error_m"],
            "chamfer_px": self.last_calibration_metrics["chamfer_px"],
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "scene_cut": scene_cut,
            "halfway_sanity": sanity,
            "verified_keypoints": self.last_keypoint_diagnostics["verified"],
            "dropped_keypoints": self.last_keypoint_diagnostics["dropped"],
        }

    @classmethod
    def _white_line_mask(cls, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return (
            (hsv[:, :, 1] <= cls.WHITE_LINE_MAX_SATURATION)
            & (hsv[:, :, 2] >= cls.WHITE_LINE_MIN_VALUE)
        )

    def _verify_keypoints(self, frame, image_points, confidence):
        points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(confidence, dtype=np.float32).reshape(-1)
        verified = np.zeros(len(points), dtype=bool)
        confidence_mask = (
            (scores > self.KEYPOINT_CONFIDENCE)
            & np.isfinite(points).all(axis=1)
            & np.any(points != 0.0, axis=1)
        )
        white_mask = self._white_line_mask(frame)
        half_window = self.WHITE_LINE_WINDOW // 2
        height, width = white_mask.shape
        for index in np.flatnonzero(confidence_mask):
            x, y = np.rint(points[index]).astype(int)
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            x1, x2 = max(0, x - half_window), min(width, x + half_window + 1)
            y1, y2 = max(0, y - half_window), min(height, y + half_window + 1)
            local_y, local_x = np.nonzero(white_mask[y1:y2, x1:x2])
            if len(local_x) < self.MIN_WHITE_PIXELS:
                continue
            line = cv2.fitLine(
                np.column_stack((local_x + x1, local_y + y1)).astype(
                    np.float32
                ),
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            ).reshape(-1)
            direction = line[:2]
            origin = line[2:]
            offsets = np.column_stack((local_x + x1, local_y + y1)) - origin
            along = offsets @ direction
            perpendicular = (
                offsets[:, 0] * direction[1]
                - offsets[:, 1] * direction[0]
            )
            center_offset = np.asarray((x, y), dtype=np.float32) - origin
            center_distance = abs(
                float(
                    center_offset[0] * direction[1]
                    - center_offset[1] * direction[0]
                )
            )
            support_span = float(np.percentile(along, 95) - np.percentile(along, 5))
            line_like = (
                float(np.median(np.abs(perpendicular)))
                <= self.MAX_LINE_PIXEL_DEVIATION_PX
                and float(np.mean(np.abs(perpendicular) <= 2.0)) >= 0.7
                and support_span >= self.MIN_LINE_SUPPORT_SPAN_PX
                and center_distance <= self.MAX_KEYPOINT_TO_LINE_PX
                and float(np.min(along)) - 1.5
                <= float(center_offset @ direction)
                <= float(np.max(along)) + 1.5
            )
            verified[index] = line_like
        self.last_keypoint_diagnostics = {
            "confidence_candidates": int(confidence_mask.sum()),
            "verified": int(verified.sum()),
            "dropped": int(confidence_mask.sum() - verified.sum()),
        }
        self.last_verified_keypoints = verified.copy()
        return verified

    def _leave_one_out_error_m(self, image_points, pitch_points):
        if len(image_points) < self.MIN_KEYPOINTS + 1:
            return None
        errors_cm = []
        for held_out_index in range(len(image_points)):
            training_mask = np.arange(len(image_points)) != held_out_index
            training_image = image_points[training_mask]
            training_pitch = pitch_points[training_mask]
            if (
                np.linalg.matrix_rank(
                    training_image - training_image.mean(axis=0)
                ) < 2
                or np.linalg.matrix_rank(
                    training_pitch - training_pitch.mean(axis=0)
                ) < 2
            ):
                continue
            matrix, _ = cv2.findHomography(
                training_image, training_pitch, method=0
            )
            if matrix is None or not np.isfinite(matrix).all():
                continue
            projected = self.project_points(
                matrix, [image_points[held_out_index]]
            )[0]
            error = float(
                np.linalg.norm(projected - pitch_points[held_out_index])
            )
            if np.isfinite(error):
                errors_cm.append(error)
        if not errors_cm:
            return None
        return float(np.median(errors_cm) / 100.0)

    def _template_chamfer_px(self, frame, image_to_pitch):
        white_mask = self._white_line_mask(frame)
        if not np.any(white_mask):
            return None
        height, width = white_mask.shape
        pitch_to_image = np.linalg.inv(
            np.asarray(image_to_pitch, dtype=np.float64)
        )
        if not np.isfinite(pitch_to_image).all():
            return None
        edges = self.configuration.edges
        vertices = self.configuration.vertices
        pitch_segments = [
            (vertices[start], vertices[end]) for start, end in edges
        ]
        center_x = self.configuration.length / 2
        center_y = self.configuration.width / 2
        radius = self.configuration.centre_circle_radius
        circle = np.asarray(
            [
                (
                    center_x + radius * np.cos(angle),
                    center_y + radius * np.sin(angle),
                )
                for angle in np.linspace(0.0, 2 * np.pi, 193)
            ],
            dtype=np.float32,
        )
        pitch_segments.extend(zip(circle, circle[1:]))

        projected_samples = []
        for start, end in pitch_segments:
            projected_ends = self.project_points(
                pitch_to_image, (start, end)
            )
            if not np.isfinite(projected_ends).all():
                continue
            pixel_length = float(
                np.linalg.norm(projected_ends[1] - projected_ends[0])
            )
            if pixel_length > 100_000:
                continue
            sample_count = max(2, int(np.ceil(pixel_length)) + 1)
            fractions = np.linspace(0.0, 1.0, sample_count)[:, None]
            projected_samples.append(
                projected_ends[0]
                + fractions * (projected_ends[1] - projected_ends[0])
            )
        if not projected_samples:
            return None

        samples = np.vstack(projected_samples)
        pixels = np.rint(samples).astype(np.int32)
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < height)
        )
        pixels = pixels[inside]
        if len(pixels) == 0:
            return None
        distance_to_white = cv2.distanceTransform(
            (~white_mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        distances = distance_to_white[pixels[:, 1], pixels[:, 0]]
        return float(np.mean(distances))

    def _estimate(self, keypoint_xy, keypoint_confidence, frame):
        self.last_keypoint_diagnostics = {
            "confidence_candidates": 0,
            "verified": 0,
            "dropped": 0,
        }
        self.last_calibration_metrics = {
            "loo_error_m": None,
            "chamfer_px": None,
        }
        if keypoint_xy is None or keypoint_confidence is None:
            self.last_keypoint_diagnostics = {
                "confidence_candidates": 0,
                "verified": 0,
                "dropped": 0,
            }
            return None, None, 0, 0.0
        image_points = np.asarray(keypoint_xy, dtype=np.float32).reshape(-1, 2)
        confidence = np.asarray(keypoint_confidence, dtype=np.float32).reshape(-1)
        if len(image_points) != len(self.configuration.vertices):
            return None, None, 0, 0.0
        if len(confidence) != len(image_points):
            return None, None, 0, 0.0
        started = perf_counter() if self.profile_timings else None
        valid = self._verify_keypoints(frame, image_points, confidence)
        self._record_timing("keypoint_line_validation", started)
        image_points = image_points[valid]
        pitch_points = self.configuration.vertices[valid]
        if len(image_points) < self.MIN_KEYPOINTS:
            return None, None, 0, 0.0
        if (
            np.linalg.matrix_rank(image_points - image_points.mean(axis=0)) < 2
            or np.linalg.matrix_rank(pitch_points - pitch_points.mean(axis=0)) < 2
        ):
            return None, None, 0, 0.0

        started = perf_counter() if self.profile_timings else None
        matrix, inliers = cv2.findHomography(
            image_points,
            pitch_points,
            cv2.RANSAC,
            self.RANSAC_REPROJECTION_CM,
        )
        self._record_timing("homography_ransac", started)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return None, None, 0, 0.0
        inlier_mask = inliers.ravel().astype(bool)
        inlier_count = int(inlier_mask.sum())
        inlier_ratio = inlier_count / len(image_points)
        if (
            inlier_count < self.MIN_INLIERS
            or inlier_ratio < self.MIN_INLIER_RATIO
            or abs(float(np.linalg.det(matrix))) < 1e-12
            or np.linalg.cond(matrix) > 1e12
        ):
            return None, None, inlier_count, inlier_ratio

        affine_scale = float(np.linalg.norm(matrix[:2, :2], ord=2))
        if not 0.01 <= affine_scale <= 1000.0:
            return None, None, inlier_count, inlier_ratio
        started = perf_counter() if self.profile_timings else None
        loo_error_m = self._leave_one_out_error_m(image_points, pitch_points)
        self._record_timing("leave_one_out_error", started)
        started = perf_counter() if self.profile_timings else None
        chamfer_px = self._template_chamfer_px(frame, matrix)
        self._record_timing("pitch_line_agreement", started)
        self.last_calibration_metrics = {
            "loo_error_m": loo_error_m,
            "chamfer_px": chamfer_px,
        }
        return (
            matrix,
            self.last_calibration_metrics["loo_error_m"],
            inlier_count,
            inlier_ratio,
        )

    def _most_players_project_to_pitch(self, matrix, tracked_points):
        player_points = [
            point
            for track_id, point in tracked_points.items()
            if track_id != "__ball__"
        ]
        if not player_points:
            return True
        projected = self.project_points(matrix, player_points)
        inside = [self._inside_pitch(point) for point in projected]
        return sum(inside) / len(inside) >= 0.5

    def _positions_are_stable(self, matrix, tracked_points):
        if len(self.previous_positions) < 3:
            return True
        movements = []
        for track_id, point in tracked_points.items():
            previous = self.previous_positions.get(track_id)
            if previous is None:
                continue
            current = self.project_points(matrix, [point])[0]
            movements.append(float(np.linalg.norm(current - previous)))
        return (
            not movements
            or float(np.median(movements)) <= self.MAX_TRACK_JUMP_CM
        )

    def _is_scene_cut(self, frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        cut = (
            self.previous_histogram is not None
            and cv2.compareHist(
                self.previous_histogram, histogram, cv2.HISTCMP_BHATTACHARYYA
            )
            > 0.60
        )
        self.previous_histogram = histogram
        return bool(cut)

    def _check_halfway_sanity(
        self, keypoint_xy, keypoint_confidence, tracked_points, projected
    ):
        if keypoint_xy is None or keypoint_confidence is None:
            return None
        points = np.asarray(keypoint_xy, dtype=np.float32).reshape(-1, 2)
        confidence = np.asarray(keypoint_confidence, dtype=np.float32).reshape(-1)
        if len(points) != 32 or min(confidence[13], confidence[16]) <= 0.5:
            return None
        left, right = points[13], points[16]
        valid_results = []
        for track_id, point in tracked_points.items():
            projected_point = projected.get(track_id)
            if projected_point is None:
                continue
            low_y, high_y = sorted((float(left[1]), float(right[1])))
            if not low_y <= float(point[1]) <= high_y:
                continue
            if abs(float(right[1] - left[1])) < 1e-3:
                center_x = (float(left[0]) + float(right[0])) / 2.0
            else:
                fraction = (float(point[1]) - float(left[1])) / (
                    float(right[1]) - float(left[1])
                )
                center_x = float(left[0] + fraction * (right[0] - left[0]))
            image_left = float(point[0]) < center_x
            pitch_left = float(projected_point[0]) < self.configuration.length / 2
            valid_results.append(image_left == pitch_left)
        if len(valid_results) < 3:
            return None
        return sum(valid_results) / len(valid_results) >= 0.8

    @staticmethod
    def project_points(matrix, points):
        points_array = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points_array, matrix).reshape(-1, 2)

    def _inside_pitch(self, point):
        return (
            -1000 <= float(point[0]) <= self.configuration.length + 1000
            and -1000 <= float(point[1]) <= self.configuration.width + 1000
        )

    def draw_radar(
        self,
        positions,
        output_size=(420, 245),
        ball_position=None,
        has_pitch_reference=True,
    ):
        width, height = output_size
        radar = np.full((height, width, 3), (55, 117, 74), dtype=np.uint8)
        pitch = self.configuration
        scale = min(width / pitch.length, height / pitch.width)
        pitch_width = int(round(pitch.length * scale))
        pitch_height = int(round(pitch.width * scale))
        offset_x = (width - pitch_width) // 2
        offset_y = (height - pitch_height) // 2

        def to_pixel(point):
            return (
                int(round(offset_x + float(point[0]) * scale)),
                int(round(offset_y + float(point[1]) * scale)),
            )

        line_color = (235, 240, 235)
        cv2.rectangle(
            radar,
            to_pixel((0, 0)),
            to_pixel((pitch.length, pitch.width)),
            line_color,
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            radar,
            to_pixel((pitch.length / 2, 0)),
            to_pixel((pitch.length / 2, pitch.width)),
            line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            radar,
            to_pixel((pitch.length / 2, pitch.width / 2)),
            max(1, int(round(pitch.centre_circle_radius * scale))),
            line_color,
            1,
            cv2.LINE_AA,
        )
        cv2.circle(
            radar,
            to_pixel((pitch.length / 2, pitch.width / 2)),
            2,
            line_color,
            cv2.FILLED,
            cv2.LINE_AA,
        )

        penalty_y1 = (pitch.width - pitch.penalty_box_width) / 2
        penalty_y2 = (pitch.width + pitch.penalty_box_width) / 2
        goal_y1 = (pitch.width - pitch.goal_box_width) / 2
        goal_y2 = (pitch.width + pitch.goal_box_width) / 2
        for x1, x2 in (
            (0, pitch.penalty_box_length),
            (pitch.length - pitch.penalty_box_length, pitch.length),
        ):
            cv2.rectangle(
                radar,
                to_pixel((x1, penalty_y1)),
                to_pixel((x2, penalty_y2)),
                line_color,
                1,
                cv2.LINE_AA,
            )
        for x1, x2 in (
            (0, pitch.goal_box_length),
            (pitch.length - pitch.goal_box_length, pitch.length),
        ):
            cv2.rectangle(
                radar,
                to_pixel((x1, goal_y1)),
                to_pixel((x2, goal_y2)),
                line_color,
                1,
                cv2.LINE_AA,
            )
        for x in (
            pitch.penalty_spot_distance,
            pitch.length - pitch.penalty_spot_distance,
        ):
            cv2.circle(
                radar,
                to_pixel((x, pitch.width / 2)),
                2,
                line_color,
                cv2.FILLED,
                cv2.LINE_AA,
            )

        for _, point in positions.items():
            x, y, color, label = point
            if not (0 <= x <= pitch.length and 0 <= y <= pitch.width):
                continue
            pixel = to_pixel((x, y))
            cv2.circle(radar, pixel, 5, color, cv2.FILLED, cv2.LINE_AA)
            if label is not None:
                cv2.putText(
                    radar,
                    str(label),
                    (pixel[0] + 5, pixel[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
        if ball_position is not None:
            ball_x, ball_y = ball_position
            if 0 <= ball_x <= pitch.length and 0 <= ball_y <= pitch.width:
                cv2.circle(
                    radar,
                    to_pixel(ball_position),
                    5,
                    (40, 240, 255),
                    cv2.FILLED,
                    cv2.LINE_AA,
                )
        if not has_pitch_reference:
            message = "TODO: CALIBRATE PITCH"
            text_size = cv2.getTextSize(
                message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )[0]
            text_origin = (
                max(4, (width - text_size[0]) // 2),
                max(22, height // 2),
            )
            cv2.putText(
                radar,
                message,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
        return radar

    def draw_debug_overlay(self, frame, image_to_pitch):
        overlay = frame.copy()
        pitch_to_image = np.linalg.inv(
            np.asarray(image_to_pitch, dtype=np.float64)
        )
        if abs(float(pitch_to_image[2, 2])) > 1e-12:
            pitch_to_image /= pitch_to_image[2, 2]

        def draw_projected_segment(start, end, color):
            image_points = self.project_points(
                pitch_to_image, (start, end)
            )
            if (
                not np.isfinite(image_points).all()
                or np.max(np.abs(image_points)) > 1_000_000
            ):
                return
            cv2.line(
                overlay,
                tuple(np.rint(image_points[0]).astype(int)),
                tuple(np.rint(image_points[1]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )

        for start_index, end_index in self.configuration.edges:
            draw_projected_segment(
                self.configuration.vertices[start_index],
                self.configuration.vertices[end_index],
                (255, 0, 255),
            )

        center_x = self.configuration.length / 2
        center_y = self.configuration.width / 2
        radius = self.configuration.centre_circle_radius
        circle = [
            (
                center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle),
            )
            for angle in np.linspace(0.0, 2 * np.pi, 97)
        ]
        for start, end in zip(circle, circle[1:]):
            draw_projected_segment(start, end, (255, 255, 0))
        return overlay

    @staticmethod
    def draw_debug_keypoints(
        frame, keypoint_xy, keypoint_confidence, loo_error_m, chamfer_px
    ):
        if keypoint_xy is None or keypoint_confidence is None:
            return frame
        points = np.asarray(keypoint_xy).reshape(-1, 2)
        confidence = np.asarray(keypoint_confidence).reshape(-1)
        for point, score in zip(points, confidence):
            if score > PitchRadarProjector.KEYPOINT_CONFIDENCE and np.any(point != 0):
                cv2.circle(
                    frame,
                    tuple(np.round(point).astype(int)),
                    4,
                    (0, 255, 255),
                    cv2.FILLED,
                    cv2.LINE_AA,
                )
        loo_label = (
            f"LOO: {loo_error_m:.2f} m"
            if loo_error_m is not None
            else "LOO: unavailable"
        )
        chamfer_label = (
            f"chamfer: {chamfer_px:.1f} px"
            if chamfer_px is not None
            else "chamfer: unavailable"
        )
        label = f"Pitch {loo_label}; {chamfer_label}"
        cv2.putText(
            frame,
            label,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame
