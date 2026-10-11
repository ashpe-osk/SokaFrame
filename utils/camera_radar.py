"""OpenCV scatter radar for image-space player and ball positions."""

import cv2
import numpy as np

from utils.team_color_utils import TeamColorAssigner


class CameraRadarRenderer:
    TITLE = "CAMERA VIEW - NOT PITCH CALIBRATED"

    def __init__(self, output_size=(420, 245)):
        self.width, self.height = output_size
        if self.width < 120 or self.height < 100:
            raise ValueError("Camera radar dimensions are too small")
        self._background = np.full(
            (self.height, self.width, 3), (36, 65, 50), dtype=np.uint8
        )
        self._draw_static_background()

    def _draw_static_background(self):
        radar = self._background
        cv2.putText(
            radar,
            self.TITLE,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        left, top = 12, 30
        right, bottom = self.width - 12, self.height - 12
        cv2.rectangle(radar, (left, top), (right, bottom), (205, 220, 210), 1)
        midpoint_x = (left + right) // 2
        cv2.line(
            radar,
            (midpoint_x, top),
            (midpoint_x, bottom),
            (145, 170, 155),
            1,
            cv2.LINE_AA,
        )

    def render(self, positions, frame_size, ball_position=None, team_colors=None):
        frame_height, frame_width = frame_size[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Camera radar requires positive frame dimensions")

        colors = {**TeamColorAssigner.TEAM_COLORS, **(team_colors or {})}
        radar = self._background.copy()
        left, top = 12, 30
        right, bottom = self.width - 12, self.height - 12

        for point in positions.values():
            x, y, color, tracker_id = point
            radar_x = left + round(float(x) / frame_width * (right - left))
            radar_y = top + round(float(y) / frame_height * (bottom - top))
            radar_x = int(np.clip(radar_x, left, right))
            radar_y = int(np.clip(radar_y, top, bottom))
            cv2.circle(radar, (radar_x, radar_y), 6, (16, 24, 20), 3, cv2.LINE_AA)
            cv2.circle(radar, (radar_x, radar_y), 4, tuple(map(int, color)), -1, cv2.LINE_AA)
            if tracker_id is not None:
                cv2.putText(
                    radar,
                    str(tracker_id),
                    (radar_x + 6, radar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if ball_position is not None:
            ball_x = left + round(float(ball_position[0]) / frame_width * (right - left))
            ball_y = top + round(float(ball_position[1]) / frame_height * (bottom - top))
            cv2.circle(radar, (ball_x, ball_y), 5, (16, 24, 20), 3, cv2.LINE_AA)
            cv2.circle(
                radar,
                (ball_x, ball_y),
                3,
                colors[TeamColorAssigner.BALL],
                -1,
                cv2.LINE_AA,
            )

        # TODO: Add pitch coordinates with a night-trained keypoint model or manual anchors plus camera-motion tracking.
        return radar
