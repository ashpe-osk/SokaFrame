"""Filter detector results to eligible players before they reach BoT-SORT."""

from collections import Counter

import numpy as np

from utils.pitch_region import (
    build_pitch_grass_mask,
    is_foot_point_on_grass,
    is_inside_playing_area,
    local_grass_support,
    overlaps_referee_detection,
)


class PlayerTrackingInputFilter:
    def __init__(
        self,
        class_categories,
        referee_duplicate_confidence,
        referee_duplicate_iou,
        enable_pitch_mask=True,
        diagnostics_enabled=False,
    ):
        self.class_categories = class_categories
        self.referee_duplicate_confidence = referee_duplicate_confidence
        self.referee_duplicate_iou = referee_duplicate_iou
        self.enable_pitch_mask = enable_pitch_mask
        self.diagnostics_enabled = diagnostics_enabled
        self.detections = []
        self.selected_indices = []
        self.result_detection_offsets = []
        self.filter_counts = Counter()
        self.valid_detection_counts = Counter()
        self.total_detection_count = 0
        self.rejected_detections = []
        self.frame_number = 0

    def __call__(self, predictor):
        self.detections = []
        self.selected_indices = []
        self.result_detection_offsets = []
        self.filter_counts = Counter()
        self.rejected_detections = []

        for result_index, result in enumerate(predictor.results):
            self.result_detection_offsets.append(len(self.detections))
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                self.selected_indices.append([])
                continue

            box_data = boxes.data
            if hasattr(box_data, "cpu"):
                box_data = box_data.cpu().numpy()
            else:
                box_data = np.asarray(box_data)
            frame = result.orig_img
            grass_mask = (
                build_pitch_grass_mask(frame)
                if self.enable_pitch_mask
                else None
            )
            referee_detections = []
            for row in box_data:
                x1, y1, x2, y2, confidence, class_id = row[:6]
                if self.class_categories.get(int(class_id), "other") == "referee":
                    referee_detections.append(
                        {
                            "bbox": (float(x1), float(y1), float(x2), float(y2)),
                            "confidence": float(confidence),
                        }
                    )

            selected = []
            result_detections = []
            for detection_index, row in enumerate(box_data):
                x1, y1, x2, y2, confidence, class_id = row[:6]
                object_type = self.class_categories.get(
                    int(class_id), "other"
                )
                detection = {
                    "tracker_id": None,
                    "class_id": int(class_id),
                    "object_type": object_type,
                    "bbox": (float(x1), float(y1), float(x2), float(y2)),
                    "confidence": float(confidence),
                    "pretracker_selected": False,
                    "pretracker_rejection_reason": None,
                }
                if self.diagnostics_enabled and object_type in (
                    "player",
                    "goalkeeper",
                ):
                    foot_point = (
                        (detection["bbox"][0] + detection["bbox"][2]) / 2.0,
                        detection["bbox"][3],
                    )
                    sample_x, sample_y = map(int, foot_point)
                    sample_y -= 6
                    detection["pretracker_spatial_evidence"] = {
                        "foot_point": foot_point,
                        "inside_playing_area": is_inside_playing_area(
                            foot_point, frame.shape
                        ),
                        "grass_mask_contains_sample": (
                            grass_mask is None
                            or (
                                0 <= sample_x < frame.shape[1]
                                and 0 <= sample_y < frame.shape[0]
                                and bool(grass_mask[sample_y, sample_x])
                            )
                        ),
                        "local_grass_support": (
                            local_grass_support(frame, foot_point)
                            if self.enable_pitch_mask
                            else None
                        ),
                    }
                result_detections.append(detection)

                if object_type not in ("player", "goalkeeper"):
                    continue

                foot_point = ((x1 + x2) / 2.0, y2)
                if not is_inside_playing_area(foot_point, frame.shape):
                    self._record_rejection(detection, "outside_pitch")
                    continue
                if overlaps_referee_detection(
                    detection,
                    referee_detections,
                    minimum_confidence=self.referee_duplicate_confidence,
                    minimum_iou=self.referee_duplicate_iou,
                ):
                    self._record_rejection(
                        detection, "overlapping_referee"
                    )
                    continue
                if self.enable_pitch_mask and not is_foot_point_on_grass(
                    grass_mask, foot_point, frame=frame
                ):
                    self._record_rejection(detection, "pitch_mask")
                    continue

                selected.append(detection_index)
                detection["pretracker_selected"] = True
                self.valid_detection_counts[object_type] += 1

            self.detections.extend(result_detections)
            self.selected_indices.append(selected)
            self.total_detection_count += len(result_detections)
            predictor.results[result_index] = result[selected]

    def _record_rejection(self, detection, reason):
        detection["pretracker_rejection_reason"] = reason
        self.filter_counts[reason] += 1
        self.rejected_detections.append(
            {
                "frame_number": self.frame_number,
                "class_id": detection["class_id"],
                "object_type": detection["object_type"],
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "rejection_reason": reason,
            }
        )

    def attach_tracker_ids(self, tracker):
        current_track_ids = {
            int(track.idx): int(track.track_id)
            for track in tracker.tracked_stracks
            if track.frame_id == tracker.frame_id and track.is_activated
        }
        for result_index, selected in enumerate(self.selected_indices):
            result_offset = self.result_detection_offsets[result_index]
            for tracker_input_index, detection_index in enumerate(selected):
                tracker_id = current_track_ids.get(tracker_input_index)
                if tracker_id is not None:
                    self.detections[result_offset + detection_index][
                        "tracker_id"
                    ] = tracker_id
        return self.detections
