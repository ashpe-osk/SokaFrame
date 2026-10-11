"""Persistent referee identification and short-track visibility control."""

from collections import Counter, deque


class RefereeTrackManager:
    HISTORY_SIZE = 30
    REFEREE_RATIO = 0.30
    MIN_HISTORY_SAMPLES = 8
    MIN_VISIBLE_AGE = 3
    STATE_EXPIRY_FRAMES = 120

    def __init__(self):
        self.class_history = {}
        self.track_age = Counter()
        self.last_seen_frame = {}
        self.frame_index = 0
        self.referee_reason = {}
        self.current_class_referee_ids = set()
        self.class_detected_track_ids = set()

    def update(self, detections):
        self.frame_index += 1
        current_ids = set()
        self.current_class_referee_ids = set()
        counts = {
            "detected_by_class": 0,
            "reclassified_by_class_history": 0,
            "rejected_by_kit_color": 0,
        }
        for detection in detections:
            tracker_id = detection.get("tracker_id")
            if tracker_id is None:
                continue
            current_ids.add(tracker_id)
            self.last_seen_frame[tracker_id] = self.frame_index
            self.track_age[tracker_id] += 1
            history = self.class_history.setdefault(
                tracker_id, deque(maxlen=self.HISTORY_SIZE)
            )
            object_type = detection.get("object_type", "other")
            history.append(object_type)
            if object_type == "referee":
                self.current_class_referee_ids.add(tracker_id)
                self.class_detected_track_ids.add(tracker_id)
                counts["detected_by_class"] += 1
            if (
                self.referee_reason.get(tracker_id) is None
                and len(history) >= self.MIN_HISTORY_SAMPLES
                and sum(label == "referee" for label in history)
                >= int(len(history) * self.REFEREE_RATIO + 0.999)
            ):
                self.mark_referee(tracker_id, "reclassified_by_class_history")

            reason = self.referee_reason.get(tracker_id)
            if reason == "reclassified_by_class_history" and object_type != "referee":
                counts["reclassified_by_class_history"] += 1
            elif reason == "rejected_by_kit_color":
                counts["rejected_by_kit_color"] += 1

        self._expire_stale_tracks()
        visible_ids = {
            tracker_id
            for tracker_id in current_ids
            if tracker_id not in self.referee_reason
            and tracker_id not in self.current_class_referee_ids
            and self.track_age[tracker_id] >= self.MIN_VISIBLE_AGE
        }
        return visible_ids, counts

    def _expire_stale_tracks(self):
        stale_ids = [
            tracker_id
            for tracker_id, last_seen in self.last_seen_frame.items()
            if self.frame_index - last_seen > self.STATE_EXPIRY_FRAMES
        ]
        for tracker_id in stale_ids:
            self.last_seen_frame.pop(tracker_id, None)
            self.class_history.pop(tracker_id, None)
            self.track_age.pop(tracker_id, None)
            self.referee_reason.pop(tracker_id, None)
            self.class_detected_track_ids.discard(tracker_id)

    def mark_referee(self, tracker_id, reason):
        if tracker_id not in self.referee_reason:
            self.referee_reason[tracker_id] = reason
        if self.referee_reason[tracker_id] == "reclassified_by_class_history":
            return
        elif reason == "reclassified_by_class_history":
            self.referee_reason[tracker_id] = reason
