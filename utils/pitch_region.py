"""Conservative multi-view image-space playing-area filter."""

import cv2
import numpy as np


_PLAYING_AREA = np.asarray(
    (
        (0.0, 0.20),
        (1.0, 0.19),
        (1.0, 0.96),
        (0.0, 0.99),
    ),
    dtype=np.float32,
)
_FRAME_EDGE_GRASS_MARGIN = 12
_EDGE_GRASS_SAMPLE_RADIUS = 4
_MIN_EDGE_GRASS_SUPPORT = 0.05
_BOTTOM_FRAME_GRASS_MARGIN = 0.05
_MIN_BOTTOM_FRAME_GRASS_SUPPORT = 0.65


def is_inside_playing_area(point, frame_shape):
    """Return whether an image point lies in the conservative pitch ROI."""
    height, width = frame_shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Playing-area checks require positive frame dimensions")
    x, y = map(float, point)
    normalized_point = (x / width, y / height)
    return cv2.pointPolygonTest(_PLAYING_AREA, normalized_point, False) >= 0


def overlaps_referee_detection(
    candidate,
    referee_detections,
    minimum_confidence=0.30,
    minimum_iou=0.50,
):
    """Detect a player-class box duplicated over a confident referee box."""
    x1, y1, x2, y2 = map(float, candidate["bbox"])
    candidate_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if candidate_area == 0:
        return False

    for referee in referee_detections:
        if referee.get("confidence", 0.0) < minimum_confidence:
            continue
        rx1, ry1, rx2, ry2 = map(float, referee["bbox"])
        intersection_width = max(0.0, min(x2, rx2) - max(x1, rx1))
        intersection_height = max(0.0, min(y2, ry2) - max(y1, ry1))
        intersection = intersection_width * intersection_height
        referee_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        union = candidate_area + referee_area - intersection
        if union > 0 and intersection / union >= minimum_iou:
            return True
    return False


def build_pitch_grass_mask(frame):
    """Build a filled hull mask for the largest connected grass region."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(
        hsv,
        np.asarray((30, 40, 40), dtype=np.uint8),
        np.asarray((90, 255, 255), dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    if component_count <= 1:
        return None

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    mask = np.zeros_like(grass)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def is_foot_point_on_grass(mask, foot_point, offset=6, frame=None):
    """Return whether the sample pixel above a foot point falls inside mask."""
    if mask is None:
        return True
    height, width = mask.shape[:2]
    x, y = map(int, foot_point)
    near_bottom_edge = y >= height * (1.0 - _BOTTOM_FRAME_GRASS_MARGIN)
    y -= int(offset)
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    if not mask[y, x]:
        return False
    near_side_edge = min(x, width - 1 - x) < _FRAME_EDGE_GRASS_MARGIN
    if frame is None or not (near_side_edge or near_bottom_edge):
        return True

    support = local_grass_support(frame, foot_point, offset=offset)
    minimum_support = (
        _MIN_BOTTOM_FRAME_GRASS_SUPPORT
        if near_bottom_edge
        else _MIN_EDGE_GRASS_SUPPORT
    )
    return support >= minimum_support


def local_grass_support(frame, foot_point, offset=6):
    """Measure the share of grass-colored pixels around the foot sample."""
    height, width = frame.shape[:2]
    x, y = map(int, foot_point)
    y -= int(offset)
    if x < 0 or x >= width or y < 0 or y >= height:
        return 0.0
    radius = _EDGE_GRASS_SAMPLE_RADIUS
    y1, y2 = max(0, y - radius), min(height, y + radius + 1)
    x1, x2 = max(0, x - radius), min(width, x + radius + 1)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(
        hsv,
        np.asarray((30, 40, 40), dtype=np.uint8),
        np.asarray((90, 255, 255), dtype=np.uint8),
    )
    return cv2.countNonZero(grass) / grass.size
