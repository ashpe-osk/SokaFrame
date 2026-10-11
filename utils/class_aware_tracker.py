"""ByteTrack association that prevents detections from different classes matching."""

from collections import deque

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker, TrackState
from ultralytics.trackers.utils import matching
from ultralytics.trackers.utils.stracks import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from utils.team_color_utils import TeamColorAssigner


class ClassAwareByteTrack(BYTETracker):
    class_categories = {}
    CLASS_MISMATCH_COST = 1_000_000.0
    ENABLE_LOST_TRACK_SPATIAL_GATE = True
    ENABLE_TRACKED_TRACK_SPATIAL_GATE = False
    LOST_TRACK_MAX_SPEED_PER_FRAME = 10.0
    LOST_TRACK_MAX_MISSED_FRAME_ALLOWANCE = 300.0
    APPEARANCE_HISTORY_SIZE = 15
    APPEARANCE_MIN_SAMPLES = 3
    APPEARANCE_DISTANCE_SCALE = 25.0
    APPEARANCE_COST_WEIGHT = 0.60

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._appearance_history = {}
        self._current_appearance_features = {}
        self._last_observed_boxes = {}

    @classmethod
    def set_class_categories(cls, class_categories):
        cls.class_categories = dict(class_categories)

    @classmethod
    def _class_category(cls, class_id):
        try:
            class_id = int(class_id)
        except (TypeError, ValueError, OverflowError):
            return class_id
        category = cls.class_categories.get(class_id, class_id)
        if category == "goalkeeper":
            return "player"
        return class_id if category == "other" else category

    @classmethod
    def _gate_class_mismatches(cls, costs, tracks, detections):
        track_categories = [cls._class_category(track.cls) for track in tracks]
        detection_categories = [
            cls._class_category(detection.cls) for detection in detections
        ]
        compatible = np.asarray(
            [
                [
                    track_category == detection_category
                    for detection_category in detection_categories
                ]
                for track_category in track_categories
            ],
            dtype=bool,
        ).reshape(costs.shape)
        gated_costs = costs.copy()
        gated_costs[~compatible] = cls.CLASS_MISMATCH_COST
        return gated_costs

    def update(self, results, img=None, feats=None, **kwargs):
        self._current_appearance_features = self._extract_detection_appearances(
            results, img
        )
        self.frame_id += 1
        activated = []
        refind = []
        lost = []
        removed = []

        results_high, results_low, mask_high, mask_low = self._split_detections(
            results
        )
        detections = self.init_track(
            results_high, self._input_for(img, feats, mask_high)
        )
        detections_second = self.init_track(
            results_low, self._input_for(img, feats, mask_low)
        )
        for tracks, mask in (
            (detections, mask_high),
            (detections_second, mask_low),
        ):
            for track, index in zip(tracks, np.flatnonzero(mask)):
                track.idx = index

        unconfirmed, tracked = self._split_tracked()
        track_pool = joint_stracks(tracked, self.lost_stracks)
        self.multi_predict(track_pool)
        self._pre_first_associate(track_pool, unconfirmed, img, results_high)

        unmatched_tracks, unmatched_detections = self._first_association(
            track_pool, detections, activated, refind
        )
        unmatched_tracks, unmatched_detections = self._post_first_association(
            track_pool,
            detections,
            unmatched_tracks,
            unmatched_detections,
            activated,
            refind,
        )
        self._second_association(
            track_pool,
            unmatched_tracks,
            detections_second,
            activated,
            refind,
            lost,
        )
        unmatched_detections, detections = self._unconfirmed_association(
            unconfirmed,
            unmatched_detections,
            detections,
            activated,
            removed,
        )
        self._init_new_tracks(
            unmatched_detections, detections, activated, refind
        )
        self._remove_stale_lost(removed)
        self._merge_track_pools(activated, refind, lost, removed)
        self._update_appearance_profiles(img)
        return self._format_output()

    def _merge_track_pools(self, activated, refind, lost, removed):
        self.tracked_stracks = [
            track
            for track in self.tracked_stracks
            if track.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)

        categories = {
            self._class_category(track.cls)
            for track in self.tracked_stracks + self.lost_stracks
        }
        tracked_by_category = []
        lost_by_category = []
        for category in categories:
            tracked_category, lost_category = remove_duplicate_stracks(
                [
                    track
                    for track in self.tracked_stracks
                    if self._class_category(track.cls) == category
                ],
                [
                    track
                    for track in self.lost_stracks
                    if self._class_category(track.cls) == category
                ],
            )
            tracked_by_category.extend(tracked_category)
            lost_by_category.extend(lost_category)
        self.tracked_stracks = tracked_by_category
        self.lost_stracks = lost_by_category
        self.removed_stracks_frame = removed
        self.removed_stracks.extend(removed)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]
        live_track_ids = {
            track.track_id for track in self.tracked_stracks + self.lost_stracks
        }
        for track_id in list(self._appearance_history):
            if track_id not in live_track_ids:
                self._appearance_history.pop(track_id, None)
        for track_id in list(self._last_observed_boxes):
            if track_id not in live_track_ids:
                self._last_observed_boxes.pop(track_id, None)

    def get_dists(self, tracks, detections):
        costs = super().get_dists(tracks, detections)
        if self.ENABLE_LOST_TRACK_SPATIAL_GATE:
            costs = self._gate_lost_track_spatial_costs(
                costs, tracks, detections
            )
        costs = self._appearance_aware_costs(costs, tracks, detections)
        if self.ENABLE_TRACKED_TRACK_SPATIAL_GATE:
            costs = self._gate_tracked_track_spatial_costs(
                costs, tracks, detections
            )
        return self._gate_class_mismatches(costs, tracks, detections)

    def _gate_tracked_track_spatial_costs(self, costs, tracks, detections):
        if costs.size == 0:
            return costs

        gated_costs = costs.copy()
        for track_index, track in enumerate(tracks):
            if track.state != TrackState.Tracked:
                continue
            last_observation = self._last_observed_boxes.get(track.track_id)
            if last_observation is None:
                continue
            previous_box, _ = last_observation
            previous_height = previous_box[3] - previous_box[1]

            for detection_index, detection in enumerate(detections):
                current_box = np.asarray(detection.xyxy, dtype=np.float64)
                current_height = current_box[3] - current_box[1]
                max_displacement = max(
                    60.0, 2.0 * max(previous_height, current_height)
                )
                previous_foot = (
                    (previous_box[0] + previous_box[2]) / 2.0,
                    previous_box[3],
                )
                current_foot = (
                    (current_box[0] + current_box[2]) / 2.0,
                    current_box[3],
                )
                if (
                    np.linalg.norm(
                        np.asarray(current_foot) - np.asarray(previous_foot)
                    )
                    > max_displacement
                ):
                    gated_costs[
                        track_index, detection_index
                    ] = self.CLASS_MISMATCH_COST
        return gated_costs

    def _gate_lost_track_spatial_costs(self, costs, tracks, detections):
        if costs.size == 0:
            return costs

        gated_costs = costs.copy()
        for track_index, track in enumerate(tracks):
            if track.state != TrackState.Lost:
                continue
            last_observation = self._last_observed_boxes.get(track.track_id)
            if last_observation is None:
                continue
            previous_box, last_frame = last_observation
            previous_height = previous_box[3] - previous_box[1]
            missed_frames = max(0, self.frame_id - last_frame - 1)
            frame_allowance = min(
                self.LOST_TRACK_MAX_SPEED_PER_FRAME * missed_frames,
                self.LOST_TRACK_MAX_MISSED_FRAME_ALLOWANCE,
            )

            for detection_index, detection in enumerate(detections):
                current_box = np.asarray(detection.xyxy, dtype=np.float64)
                current_height = current_box[3] - current_box[1]
                max_height = max(previous_height, current_height)
                max_displacement = max(60.0, 2.0 * max_height) + frame_allowance
                previous_foot = (
                    (previous_box[0] + previous_box[2]) / 2.0,
                    previous_box[3],
                )
                current_foot = (
                    (current_box[0] + current_box[2]) / 2.0,
                    current_box[3],
                )
                if (
                    np.linalg.norm(
                        np.asarray(current_foot) - np.asarray(previous_foot)
                    )
                    > max_displacement
                ):
                    gated_costs[
                        track_index, detection_index
                    ] = self.CLASS_MISMATCH_COST
        return gated_costs

    def _second_association(
        self,
        strack_pool,
        u_track,
        detections_second,
        activated,
        refind,
        lost,
    ):
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        if r_tracked_stracks and detections_second:
            costs = matching.iou_distance(
                r_tracked_stracks, detections_second
            )
            costs = self._appearance_aware_costs(
                costs, r_tracked_stracks, detections_second
            )
            if self.ENABLE_TRACKED_TRACK_SPATIAL_GATE:
                costs = self._gate_tracked_track_spatial_costs(
                    costs, r_tracked_stracks, detections_second
                )
            costs = self._gate_class_mismatches(
                costs, r_tracked_stracks, detections_second
            )
            matches, u_track, _ = matching.linear_assignment(costs, thresh=0.5)
            self._apply_matches(
                matches, r_tracked_stracks, detections_second, activated, refind
            )
        else:
            u_track = list(range(len(r_tracked_stracks)))

        for index in u_track:
            track = r_tracked_stracks[index]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

    def _extract_detection_appearances(self, results, frame):
        if frame is None or not len(results):
            return {}
        appearances = {}
        boxes = results.xyxy
        for index, (bbox, class_id) in enumerate(zip(boxes, results.cls)):
            if self.class_categories.get(int(class_id)) != "player":
                continue
            feature = TeamColorAssigner._team_torso_lab_feature(frame, bbox)
            if feature is not None:
                appearances[index] = feature
        return appearances

    def _appearance_aware_costs(self, costs, tracks, detections):
        if costs.size == 0:
            return costs
        adjusted = costs.copy()
        for track_index, track in enumerate(tracks):
            history = self._appearance_history.get(track.track_id)
            if (
                history is None
                or len(history) < self.APPEARANCE_MIN_SAMPLES
                or self.class_categories.get(int(track.cls)) != "player"
            ):
                continue
            track_feature = np.median(np.stack(tuple(history)), axis=0)
            for detection_index, detection in enumerate(detections):
                detection_feature = self._current_appearance_features.get(
                    int(detection.idx)
                )
                if detection_feature is None:
                    continue
                color_distance = float(
                    np.linalg.norm(track_feature[1:] - detection_feature[1:])
                    * 255.0
                )
                appearance_cost = min(
                    color_distance / self.APPEARANCE_DISTANCE_SCALE, 1.0
                )
                adjusted[track_index, detection_index] = (
                    (1.0 - self.APPEARANCE_COST_WEIGHT)
                    * adjusted[track_index, detection_index]
                    + self.APPEARANCE_COST_WEIGHT * appearance_cost
                )
        return adjusted

    def _update_appearance_profiles(self, frame):
        for track in self.tracked_stracks:
            if (
                track.state != TrackState.Tracked
                or track.frame_id != self.frame_id
            ):
                continue
            self._last_observed_boxes[track.track_id] = (
                tuple(float(value) for value in track.xyxy),
                self.frame_id,
            )
            if (
                frame is None
                or self.class_categories.get(int(track.cls)) != "player"
            ):
                continue
            feature = self._current_appearance_features.get(int(track.idx))
            if feature is None:
                continue
            history = self._appearance_history.setdefault(
                track.track_id,
                deque(maxlen=self.APPEARANCE_HISTORY_SIZE),
            )
            history.append(feature)

    def reset(self):
        super().reset()
        self._appearance_history.clear()
        self._current_appearance_features.clear()
        self._last_observed_boxes.clear()


def register_class_aware_bytetrack(class_categories):
    """Register the configured tracker type used by Ultralytics model.track."""
    from ultralytics.trackers.track import TRACKER_MAP

    ClassAwareByteTrack.set_class_categories(class_categories)
    TRACKER_MAP["classbytetrack"] = ClassAwareByteTrack
