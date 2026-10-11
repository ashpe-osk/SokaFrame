"""Sparse optical-flow camera-motion estimation and reference coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraMotion:
    transform: object
    translation: tuple
    rotation_degrees: float
    scale: float
    confidence: float
    inliers: int
    tracked_features: int
    scene_cut: bool
    available: bool

    def to_dict(self):
        return {
            "motion": [
                float(self.translation[0]),
                float(self.translation[1]),
                float(self.rotation_degrees),
                float(self.scale),
            ],
            "translation_px": [
                float(self.translation[0]),
                float(self.translation[1]),
            ],
            "rotation_degrees": float(self.rotation_degrees),
            "scale": float(self.scale),
            "confidence": float(self.confidence),
            "inliers": int(self.inliers),
            "tracked_features": int(self.tracked_features),
            "scene_cut": bool(self.scene_cut),
            "available": bool(self.available),
            "transform_previous_to_current": (
                np.asarray(self.transform).tolist()
                if self.transform is not None
                else None
            ),
        }


class CameraMotionEstimator:
    def __init__(self, max_features=500, scene_cut_threshold=0.60):
        self.max_features = max_features
        self.scene_cut_threshold = scene_cut_threshold
        self.previous_gray = None
        self.previous_histogram = None

    @staticmethod
    def _histogram(gray):
        resized = cv2.resize(gray, (64, 36))
        histogram = cv2.calcHist([resized], [0], None, [32], [0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        return histogram

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = self._histogram(gray)
        scene_cut = (
            self.previous_histogram is not None
            and cv2.compareHist(
                self.previous_histogram,
                histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
            > self.scene_cut_threshold
        )
        if self.previous_gray is None or scene_cut:
            self.previous_gray = gray
            self.previous_histogram = histogram
            return CameraMotion(
                None, (0.0, 0.0), 0.0, 1.0, 0.0, 0, 0, scene_cut, False
            )

        points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points is None or len(points) < 4:
            self.previous_gray = gray
            self.previous_histogram = histogram
            return CameraMotion(
                None, (0.0, 0.0), 0.0, 1.0, 0.0, 0, 0, False, False
            )

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        valid = status.ravel().astype(bool)
        source = points.reshape(-1, 2)[valid]
        destination = next_points.reshape(-1, 2)[valid]
        transform = None
        inliers = 0
        if len(source) >= 4:
            transform, inlier_mask = cv2.estimateAffinePartial2D(
                source,
                destination,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if transform is not None and inlier_mask is not None:
                inliers = int(inlier_mask.sum())
            else:
                transform = None

        tracked_count = int(len(source))
        confidence = inliers / tracked_count if tracked_count else 0.0
        if transform is None or inliers < 4:
            motion = CameraMotion(
                None,
                (0.0, 0.0),
                0.0,
                1.0,
                0.0,
                inliers,
                tracked_count,
                False,
                False,
            )
        else:
            a, b = float(transform[0, 0]), float(transform[1, 0])
            scale = float(np.sqrt(a * a + b * b))
            rotation = float(np.degrees(np.arctan2(b, a)))
            motion = CameraMotion(
                transform.astype(np.float64),
                (float(transform[0, 2]), float(transform[1, 2])),
                rotation,
                scale,
                confidence,
                inliers,
                tracked_count,
                False,
                True,
            )
        self.previous_gray = gray
        self.previous_histogram = histogram
        return motion


class CameraReference:
    """Map current-frame points to a camera-stabilized reference coordinate."""

    def __init__(self):
        self.reference_to_current = np.eye(3, dtype=np.float64)
        self.epoch = 0

    def update(self, motion):
        if motion.scene_cut:
            self.reference_to_current = np.eye(3, dtype=np.float64)
            self.epoch += 1
        elif (
            motion.available
            and motion.transform is not None
            and motion.confidence >= 0.35
        ):
            matrix = np.asarray(motion.transform, dtype=np.float64)
            if (
                matrix.shape == (2, 3)
                and np.isfinite(matrix).all()
                and abs(float(np.linalg.det(matrix[:, :2]))) > 0.1
            ):
                affine = np.vstack((matrix, (0.0, 0.0, 1.0)))
                self.reference_to_current = affine @ self.reference_to_current

    def to_reference(self, point):
        current_to_reference = np.linalg.inv(self.reference_to_current)
        x, y = map(float, point)
        transformed = current_to_reference @ np.asarray((x, y, 1.0))
        if abs(float(transformed[2])) <= 1e-12:
            return float(x), float(y)
        return (
            float(transformed[0] / transformed[2]),
            float(transformed[1] / transformed[2]),
        )
