"""Bounded, online identity reconciliation for raw BoT-SORT player tracks."""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math

import cv2
import numpy as np


PLAYER_CLASSES = frozenset({"player", "goalkeeper"})
UNKNOWN_TEAMS = frozenset({None, "", "unknown", "unknown_gk"})
IDENTITY_CHECKPOINT_VERSION = 1


def _unit_vector(values):
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else None


def _histogram_distance(first, second):
    if first is None or second is None or first.shape != second.shape:
        return None
    value = float(
        cv2.compareHist(
            first.astype(np.float32).reshape(-1, 1),
            second.astype(np.float32).reshape(-1, 1),
            cv2.HISTCMP_BHATTACHARYYA,
        )
    )
    return value if math.isfinite(value) else None


@dataclass
class _PlayerIdentity:
    global_id: int
    tracker_id: int
    team: str | None
    object_type: str
    position: np.ndarray
    bbox_size: np.ndarray
    last_seen_frame: int
    reid_feature: np.ndarray | None = None
    appearance_histogram: np.ndarray | None = None
    pitch_position: np.ndarray | None = None
    pitch_reliable: bool = False
    camera_epoch: int = 0
    last_appearance_frame: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=8))

    def update(
        self,
        detection,
        frame_number,
        frame_shape,
        camera_epoch,
        update_appearance=True,
    ):
        if self.camera_epoch != camera_epoch:
            self.history.clear()
            self.camera_epoch = camera_epoch
        self.position = _position(detection)
        self.bbox_size = _bbox_size(detection["bbox"])
        self.team = _resolve_team(self.team, detection.get("team"))
        self.object_type = str(detection.get("object_type", self.object_type))
        self.last_seen_frame = frame_number
        reid = _unit_vector(detection.get("reid_feature"))
        if reid is not None:
            self.reid_feature = _ema(self.reid_feature, reid)
        if update_appearance:
            histogram = _detection_appearance(detection, frame_shape)
            if histogram is not None:
                self.appearance_histogram = _histogram_ema(
                    self.appearance_histogram, histogram
                )
                self.last_appearance_frame = frame_number
        pitch = detection.get("pitch_position")
        self.pitch_reliable = bool(detection.get("pitch_reliable")) and pitch is not None
        self.pitch_position = (
            np.asarray(pitch, dtype=np.float32).reshape(2)
            if self.pitch_reliable
            else None
        )
        self.history.append((frame_number, self.position.copy()))


def _ema(previous, current, alpha=0.20):
    if previous is None or previous.shape != current.shape:
        return current.copy()
    return _unit_vector((1.0 - alpha) * previous + alpha * current)


def _histogram_ema(previous, current, alpha=0.20):
    if previous is None or previous.shape != current.shape:
        return current.copy()
    updated = (1.0 - alpha) * previous + alpha * current
    total = float(updated.sum())
    return updated / total if total > 1e-8 else current.copy()


def _resolve_team(previous, current):
    if current in UNKNOWN_TEAMS:
        return previous
    if previous in UNKNOWN_TEAMS:
        return current
    return previous


def _position(detection):
    compensated = detection.get("camera_compensated_position")
    if compensated is not None:
        return np.asarray(compensated, dtype=np.float32).reshape(2)
    x1, _, x2, y2 = map(float, detection["bbox"])
    return np.asarray(((x1 + x2) / 2.0, y2), dtype=np.float32)


def _bbox_size(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    return np.asarray((max(1.0, x2 - x1), max(1.0, y2 - y1)), dtype=np.float32)


def _appearance_histogram(detection, frame_shape, existing=None):
    if existing is not None:
        supplied = np.asarray(existing, dtype=np.float32).reshape(-1)
        if supplied.size and np.isfinite(supplied).all():
            total = float(supplied.sum())
            if total > 1e-8:
                return supplied / total
    frame = detection.get("_frame")
    if frame is None or frame_shape is None:
        return None
    x1, y1, x2, y2 = map(float, detection["bbox"])
    width, height = x2 - x1, y2 - y1
    if width < 2 or height < 4:
        return None
    left = max(0, int(round(x1 + width * 0.20)))
    right = min(frame.shape[1], int(round(x2 - width * 0.20)))
    top = max(0, int(round(y1 + height * 0.15)))
    bottom = min(frame.shape[0], int(round(y1 + height * 0.62)))
    if right - left < 2 or bottom - top < 2:
        return None
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
    return histogram.reshape(-1).astype(np.float32)


def _detection_appearance(detection, frame_shape):
    if not detection.get("_appearance_computed", False):
        detection["appearance"] = _appearance_histogram(
            detection,
            frame_shape,
            existing=detection.get("appearance"),
        )
        detection["_appearance_computed"] = True
    return detection.get("appearance")


class GlobalIdentityManager:
    """Assign persistent player IDs without changing BoT-SORT tracker IDs.

    Registry size and per-player trajectory history are bounded. Re-identifying
    a lost player only remaps the raw BoT-SORT ID; IDs are never fed back to the
    tracker. Inputs must be player detections already approved by the
    authoritative pre-tracker filter and post-tracker visibility checks.
    """

    APPEARANCE_REFRESH_INTERVAL_FRAMES = 10
    REID_FEATURE_REFRESH_INTERVAL_FRAMES = 10

    def __init__(
        self,
        match_threshold=0.70,
        ambiguity_margin=0.08,
        minimum_confidence=0.60,
        max_identities=None,
        max_lost_frames=None,
    ):
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        if ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be non-negative")
        self.match_threshold = float(match_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.minimum_confidence = float(minimum_confidence)
        if max_identities is not None and max_identities <= 0:
            raise ValueError("max_identities must be positive or None")
        if max_lost_frames is not None and max_lost_frames < 0:
            raise ValueError("max_lost_frames must be non-negative or None")
        self.max_identities = (
            int(max_identities) if max_identities is not None else None
        )
        self.max_lost_frames = (
            int(max_lost_frames) if max_lost_frames is not None else None
        )
        self.next_global_id = 1
        self.identities = {}
        self.tracker_to_global = {}
        self.pending_tracker_ids = set()
        self.last_frame = 0
        self.camera_epoch = 0
        self.match_events = 0
        self.created_ids = 0
        self.last_association_evidence = []

    def update(self, detections, frame_number, frame=None, camera_epoch=None):
        """Return ``tracker_id -> global_id`` for valid visible players."""
        self.last_association_evidence = []
        frame_number = int(frame_number)
        if frame_number < self.last_frame:
            raise ValueError("frame_number must be non-decreasing")
        self.last_frame = frame_number
        if camera_epoch is not None:
            self.camera_epoch = int(camera_epoch)
        frame_shape = frame.shape if frame is not None else None
        valid = self._valid_detections(detections, frame)
        if not valid:
            self._cleanup_stale_tracker_mappings()
            return {}

        # BoT-SORT normally gives one detection per active player track. If
        # multiple rows carry one raw ID, retain the strongest detection.
        by_tracker = {}
        for detection in valid:
            tracker_id = int(detection["tracker_id"])
            previous = by_tracker.get(tracker_id)
            if previous is None or detection["confidence"] > previous["confidence"]:
                by_tracker[tracker_id] = detection

        assignments = {}
        reserved_global_ids = set()
        reidentified_tracker_ids = set()
        candidate_evidence = {}
        accepted_tracker_ids = set()
        for tracker_id, detection in sorted(
            by_tracker.items(),
            key=lambda item: item[1]["confidence"],
            reverse=True,
        ):
            global_id = self.tracker_to_global.get(tracker_id)
            identity = self.identities.get(global_id)
            if (
                global_id is not None
                and identity is not None
                and global_id not in reserved_global_ids
            ):
                assignments[tracker_id] = global_id
                reserved_global_ids.add(global_id)
                self.pending_tracker_ids.discard(tracker_id)

        unassigned = [
            (tracker_id, detection)
            for tracker_id, detection in by_tracker.items()
            if tracker_id not in assignments
        ]
        remaining = {tracker_id for tracker_id, _ in unassigned}
        while remaining:
            pair_options = []
            for tracker_id in remaining:
                scored = []
                candidate_reid_scores = []
                rejected_team_candidates = []
                for global_id, identity in self.identities.items():
                    gap = frame_number - identity.last_seen_frame
                    if (
                        global_id in reserved_global_ids
                        or gap <= 0
                        or (
                            self.max_lost_frames is not None
                            and gap > self.max_lost_frames
                        )
                    ):
                        continue
                    evidence = self._match_evidence(
                        identity,
                        by_tracker[tracker_id],
                        gap,
                        frame_shape,
                    )
                    if evidence["team_conflict"]:
                        rejected_team_candidates.append(
                            (evidence["appearance_reid_score"], global_id, evidence)
                        )
                    elif evidence["combined_score"] is not None:
                        scored.append(
                            (evidence["combined_score"], global_id, evidence)
                        )
                    reid_score = self._reid_score(
                        identity, by_tracker[tracker_id]
                    )
                    if reid_score is not None:
                        candidate_reid_scores.append((reid_score, global_id))
                candidate_reid_scores.sort(reverse=True)
                if candidate_reid_scores:
                    best_reid, best_reid_id = candidate_reid_scores[0]
                    second_reid = (
                        candidate_reid_scores[1][0]
                        if len(candidate_reid_scores) > 1
                        else None
                    )
                    if (
                        best_reid >= 0.90
                        and (
                            second_reid is None
                            or best_reid - second_reid >= 0.05
                        )
                    ):
                        identity = self.identities[best_reid_id]
                        gap = frame_number - identity.last_seen_frame
                        fallback_evidence = self._match_evidence(
                            identity,
                            by_tracker[tracker_id],
                            gap,
                            frame_shape,
                            reid_fallback=True,
                        )
                        fallback_score = fallback_evidence["combined_score"]
                        if fallback_score is not None:
                            scored = [
                                candidate
                                for candidate in scored
                                if candidate[1] != best_reid_id
                            ]
                            scored.append(
                                (fallback_score, best_reid_id, fallback_evidence)
                            )
                scored.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
                team_conflict_candidate = (
                    max(
                        rejected_team_candidates,
                        key=lambda candidate: (
                            candidate[0] if candidate[0] is not None else -1,
                            candidate[1],
                        ),
                    )
                    if rejected_team_candidates
                    else None
                )
                candidate_evidence[tracker_id] = {
                    "scored": scored[:2],
                    "team_conflict": team_conflict_candidate,
                }
                if not scored:
                    continue
                best_score, global_id, _ = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else None
                if best_score < self.match_threshold:
                    continue
                if (
                    second_score is not None
                    and best_score - second_score < self.ambiguity_margin
                ):
                    continue
                pair_options.append((best_score, tracker_id, global_id))
            if not pair_options:
                break
            _, tracker_id, global_id = max(pair_options)
            self._remap_tracker(global_id, tracker_id)
            self.pending_tracker_ids.discard(tracker_id)
            reidentified_tracker_ids.add(tracker_id)
            assignments[tracker_id] = global_id
            reserved_global_ids.add(global_id)
            remaining.remove(tracker_id)
            accepted_tracker_ids.add(tracker_id)
            self.match_events += 1

        self.last_association_evidence = []
        for tracker_id, detection in by_tracker.items():
            candidates = candidate_evidence.get(
                tracker_id, {"scored": [], "team_conflict": None}
            )
            scored = candidates["scored"]
            best = scored[0] if scored else None
            runner_up = scored[1] if len(scored) > 1 else None
            margin = (
                best[0] - runner_up[0]
                if best is not None and runner_up is not None
                else None
            )
            if tracker_id in accepted_tracker_ids:
                decision = "accepted"
                reason = "best candidate passed score and ambiguity thresholds"
            elif best is not None and best[0] < self.match_threshold:
                decision = "rejected"
                reason = "best candidate below acceptance threshold"
            elif best is not None and margin is not None and margin < self.ambiguity_margin:
                decision = "unresolved"
                reason = "best-versus-runner-up margin below threshold"
            elif candidates["team_conflict"] is not None:
                conflict_score = candidates["team_conflict"][0]
                if (
                    tracker_id in self.pending_tracker_ids
                    or (
                        conflict_score is not None
                        and conflict_score >= self.match_threshold
                    )
                ):
                    decision = "unresolved"
                    reason = "plausible identity has conflicting team evidence"
                else:
                    decision = "rejected"
                    reason = "confidently conflicting team evidence"
            elif tracker_id in self.pending_tracker_ids:
                decision = "unresolved"
                reason = "waiting for confident historical identity match"
            else:
                decision = "new_identity"
                reason = "no eligible historical identity candidate"
            if decision == "unresolved":
                self.pending_tracker_ids.add(tracker_id)
            if best is None and candidates["team_conflict"] is not None:
                conflict_score, conflict_id, conflict_evidence = candidates["team_conflict"]
                self.last_association_evidence.append(
                    self._evidence_row(
                        frame_number,
                        tracker_id,
                        self.identities[conflict_id],
                        conflict_evidence,
                        decision=decision,
                        reason=reason,
                        best_score=None,
                        runner_up_score=None,
                        margin=None,
                    )
                )
            elif best is not None:
                self.last_association_evidence.append(
                    self._evidence_row(
                        frame_number,
                        tracker_id,
                        self.identities[best[1]],
                        best[2],
                        decision=decision,
                        reason=reason,
                        best_score=best[0],
                        runner_up_score=runner_up[0] if runner_up else None,
                        margin=margin,
                    )
                )
            else:
                self.last_association_evidence.append(
                    {
                        "frame_number": frame_number,
                        "current_tracker_id": tracker_id,
                        "current_tracklet_ref": f"{tracker_id}@{frame_number}",
                        "candidate_global_id": "",
                        "candidate_tracker_id": "",
                        "candidate_tracklet_ref": "",
                        "candidate_last_seen_frame": "",
                        "gap_frames": "",
                        "motion_score": "",
                        "appearance_reid_score": "",
                        "team_agreement": "",
                        "team_conflict": False,
                        "scale_consistency_score": "",
                        "time_gap_score": "",
                        "pitch_space_score": "",
                        "distance_score": "",
                        "combined_score": "",
                        "best_candidate_score": "",
                        "runner_up_score": "",
                        "score_margin": "",
                        "decision": decision,
                        "reason": reason,
                        "constraint": "",
                        "algorithm_version": "global-identity-online-v1",
                    }
                )
            if (
                best is not None
                and candidates["team_conflict"] is not None
                and candidates["team_conflict"][1]
                not in {best[1], runner_up[1] if runner_up else None}
            ):
                _, conflict_id, conflict_evidence = candidates["team_conflict"]
                self.last_association_evidence.append(
                    self._evidence_row(
                        frame_number,
                        tracker_id,
                        self.identities[conflict_id],
                        conflict_evidence,
                        decision="rejected",
                        reason="confidently conflicting team evidence",
                        best_score=best[0],
                        runner_up_score=runner_up[0] if runner_up else None,
                        margin=margin,
                    )
                )
            if runner_up is not None:
                self.last_association_evidence.append(
                    self._evidence_row(
                        frame_number,
                        tracker_id,
                        self.identities[runner_up[1]],
                        runner_up[2],
                        decision="rejected",
                        reason="runner-up candidate; included in margin calculation",
                        best_score=best[0],
                        runner_up_score=runner_up[0],
                        margin=margin,
                    )
                )

        for tracker_id, detection in by_tracker.items():
            global_id = assignments.get(tracker_id)
            if global_id is None:
                if tracker_id in self.pending_tracker_ids:
                    continue
                global_id = self._create_identity(
                    tracker_id,
                    detection,
                    frame_number,
                    frame_shape,
                    protected=reserved_global_ids,
                )
                assignments[tracker_id] = global_id
                reserved_global_ids.add(global_id)
            else:
                identity = self.identities[global_id]
                update_appearance = (
                    tracker_id in reidentified_tracker_ids
                    or identity.appearance_histogram is None
                    or frame_number - identity.last_appearance_frame
                    >= self.APPEARANCE_REFRESH_INTERVAL_FRAMES
                )
                identity.update(
                    detection,
                    frame_number,
                    frame_shape,
                    self.camera_epoch,
                    update_appearance=update_appearance,
                )

        self._cleanup_stale_tracker_mappings()
        if len(assignments.values()) != len(set(assignments.values())):
            raise AssertionError("A global player ID was assigned to multiple visible tracks.")
        return assignments

    def set_team_label(self, global_id, team):
        """Store the current consensus team on an existing identity."""
        if team not in {"team_1", "team_2"}:
            raise ValueError("team must be 'team_1' or 'team_2'")
        identity = self.identities.get(int(global_id))
        if identity is None:
            raise KeyError(f"Unknown global identity: {global_id}")
        identity.team = team

    def clear_team_label(self, global_id):
        """Clear stale team evidence when an identity is recognized as a goalkeeper."""
        identity = self.identities.get(int(global_id))
        if identity is None:
            raise KeyError(f"Unknown global identity: {global_id}")
        identity.team = "unknown"

    def _valid_detections(self, detections, frame):
        valid = []
        for source in detections:
            detection = dict(source)
            object_type = str(detection.get("object_type", "")).lower()
            if object_type not in PLAYER_CLASSES:
                continue
            if detection.get("player_eligible") is False:
                continue
            if detection.get("tracker_id") is None or "bbox" not in detection:
                continue
            try:
                confidence = float(detection.get("confidence", 0.0))
                bbox = tuple(map(float, detection["bbox"]))
                if (
                    len(bbox) != 4
                    or not np.isfinite(bbox).all()
                    or not math.isfinite(confidence)
                    or confidence < self.minimum_confidence
                ):
                    continue
            except (TypeError, ValueError):
                continue
            if frame is not None:
                detection["_frame"] = frame
            detection["confidence"] = confidence
            valid.append(detection)
        return valid

    def _match_score(
        self, identity, detection, gap, frame_shape, reid_fallback=False
    ):
        return self._match_evidence(
            identity, detection, gap, frame_shape, reid_fallback
        )["combined_score"]

    @staticmethod
    def _has_confident_team_conflict(identity, detection):
        if not detection.get("team_confident", True):
            return False
        team = detection.get("team")
        return (
            identity.team not in UNKNOWN_TEAMS
            and team not in UNKNOWN_TEAMS
            and identity.team != team
        )

    def _match_evidence(
        self, identity, detection, gap, frame_shape, reid_fallback=False
    ):
        """Return normalized [0, 1] similarity components; combined is not a probability."""
        team = detection.get("team")
        team_conflict = self._has_confident_team_conflict(identity, detection)

        position = _position(detection)
        size = _bbox_size(detection["bbox"])
        seconds = gap / 30.0
        scale = min(260.0, max(45.0, 30.0 + 20.0 * math.sqrt(seconds)))
        same_camera_epoch = int(detection.get("camera_epoch", self.camera_epoch)) == identity.camera_epoch
        if same_camera_epoch:
            position_error = float(np.linalg.norm(position - identity.position))
            spatial = math.exp(-position_error / scale)
            velocity = self._velocity(identity)
            displacement = velocity * gap
            displacement_length = float(np.linalg.norm(displacement))
            if displacement_length > 220.0:
                displacement *= 220.0 / displacement_length
            predicted = identity.position + displacement
            predicted_error = float(np.linalg.norm(position - predicted))
            prediction_scale = min(
                320.0, max(55.0, 40.0 + 24.0 * math.sqrt(seconds))
            )
            motion = math.exp(-predicted_error / prediction_scale)
        else:
            spatial = None
            motion = None
        pitch_score = self._pitch_score(identity, detection)
        if pitch_score is not None:
            spatial = None
            motion = None

        reid_score = self._reid_score(identity, detection)
        appearance = self._appearance_score(
            identity, detection, frame_shape, reid_score
        )
        if reid_fallback:
            spatial = None
            motion = None
        team_score = (
            None
            if team in UNKNOWN_TEAMS or identity.team in UNKNOWN_TEAMS
            else 1.0
        )
        size_ratio = np.minimum(identity.bbox_size, size) / np.maximum(
            identity.bbox_size, size
        )
        size_score = float(np.sqrt(np.prod(size_ratio)))
        temporal = math.exp(-gap / (30.0 * 180.0))

        components = [
            (0.23, spatial),
            (0.27, motion),
            (0.30, appearance),
            (0.12, team_score),
            (0.08, size_score),
            (0.05, temporal),
            (0.30, pitch_score),
        ]
        present = [(weight, value) for weight, value in components if value is not None]
        combined = (
            sum(weight * value for weight, value in present)
            / sum(weight for weight, _ in present)
            if present and not team_conflict
            else None
        )
        return {
            "candidate_tracker_id": identity.tracker_id,
            "candidate_last_seen_frame": identity.last_seen_frame,
            "motion_score": motion,
            "appearance_reid_score": appearance,
            "team_agreement": (
                None
                if team in UNKNOWN_TEAMS or identity.team in UNKNOWN_TEAMS
                else not team_conflict
            ),
            "team_conflict": team_conflict,
            "scale_consistency_score": size_score,
            "time_gap_score": temporal,
            "pitch_space_score": pitch_score,
            "distance_score": max(
                (value for value in (spatial, motion) if value is not None),
                default=None,
            ),
            "combined_score": combined,
            "constraint": (
                "confident_team_conflict" if team_conflict else ""
            ),
        }

    @staticmethod
    def _evidence_row(
        frame_number,
        tracker_id,
        identity,
        evidence,
        *,
        decision,
        reason,
        best_score,
        runner_up_score,
        margin,
    ):
        return {
            "frame_number": frame_number,
            "current_tracker_id": tracker_id,
            "current_tracklet_ref": f"{tracker_id}@{frame_number}",
            "candidate_global_id": identity.global_id,
            "candidate_tracker_id": evidence["candidate_tracker_id"],
            "candidate_tracklet_ref": (
                f"{evidence['candidate_tracker_id']}@"
                f"{evidence['candidate_last_seen_frame']}"
            ),
            "candidate_last_seen_frame": evidence[
                "candidate_last_seen_frame"
            ],
            "gap_frames": frame_number
            - evidence["candidate_last_seen_frame"],
            "motion_score": evidence["motion_score"],
            "appearance_reid_score": evidence["appearance_reid_score"],
            "team_agreement": evidence["team_agreement"],
            "team_conflict": evidence["team_conflict"],
            "scale_consistency_score": evidence["scale_consistency_score"],
            "time_gap_score": evidence["time_gap_score"],
            "pitch_space_score": evidence["pitch_space_score"],
            "distance_score": evidence["distance_score"],
            "combined_score": evidence["combined_score"],
            "best_candidate_score": best_score,
            "runner_up_score": runner_up_score,
            "score_margin": margin,
            "decision": decision,
            "reason": reason,
            "constraint": evidence["constraint"],
            "algorithm_version": "global-identity-online-v1",
        }

    @staticmethod
    def _velocity(identity):
        points = list(identity.history)[-5:]
        if len(points) < 2:
            return np.zeros(2, dtype=np.float32)
        start_frame, start_position = points[0]
        end_frame, end_position = points[-1]
        elapsed = end_frame - start_frame
        if elapsed <= 0:
            return np.zeros(2, dtype=np.float32)
        velocity = (end_position - start_position) / elapsed
        speed = float(np.linalg.norm(velocity))
        if speed > 20.0:
            velocity *= 20.0 / speed
        return velocity

    @staticmethod
    def _reid_score(identity, detection):
        current_reid = _unit_vector(detection.get("reid_feature"))
        if (
            current_reid is not None
            and identity.reid_feature is not None
            and current_reid.shape == identity.reid_feature.shape
        ):
            cosine = float(np.clip(np.dot(current_reid, identity.reid_feature), -1, 1))
            return (cosine + 1.0) / 2.0
        return None

    @staticmethod
    def _appearance_score(identity, detection, frame_shape, reid_score=None):
        if reid_score is not None:
            return reid_score
        current_histogram = _detection_appearance(detection, frame_shape)
        distance = _histogram_distance(
            identity.appearance_histogram, current_histogram
        )
        return None if distance is None else max(0.0, min(1.0, 1.0 - distance))

    @staticmethod
    def _pitch_score(identity, detection):
        if not identity.pitch_reliable or not detection.get("pitch_reliable"):
            return None
        pitch = detection.get("pitch_position")
        if pitch is None or identity.pitch_position is None:
            return None
        distance = float(
            np.linalg.norm(np.asarray(pitch, dtype=np.float32) - identity.pitch_position)
        )
        return math.exp(-distance / 3.0)

    def _create_identity(
        self, tracker_id, detection, frame_number, frame_shape, protected
    ):
        if self.max_identities is not None:
            self._make_room(frame_number, protected)
        global_id = self.next_global_id
        self.next_global_id += 1
        identity = _PlayerIdentity(
            global_id=global_id,
            tracker_id=tracker_id,
            team=detection.get("team"),
            object_type=str(detection["object_type"]),
            position=_position(detection),
            bbox_size=_bbox_size(detection["bbox"]),
            last_seen_frame=frame_number,
            reid_feature=_unit_vector(detection.get("reid_feature")),
            appearance_histogram=_detection_appearance(
                detection, frame_shape
            ),
            camera_epoch=int(detection.get("camera_epoch", self.camera_epoch)),
            last_appearance_frame=frame_number,
            history=deque(maxlen=8),
        )
        pitch = detection.get("pitch_position")
        identity.pitch_reliable = bool(detection.get("pitch_reliable")) and pitch is not None
        identity.pitch_position = (
            np.asarray(pitch, dtype=np.float32).reshape(2)
            if identity.pitch_reliable
            else None
        )
        identity.history.append((frame_number, identity.position.copy()))
        self.identities[global_id] = identity
        self.tracker_to_global[tracker_id] = global_id
        self.created_ids += 1
        return global_id

    def _remap_tracker(self, global_id, tracker_id):
        identity = self.identities[global_id]
        old_tracker_id = identity.tracker_id
        if self.tracker_to_global.get(old_tracker_id) == global_id:
            del self.tracker_to_global[old_tracker_id]
        previous_global_id = self.tracker_to_global.get(tracker_id)
        if previous_global_id is not None and previous_global_id != global_id:
            previous_identity = self.identities.get(previous_global_id)
            if previous_identity is not None:
                previous_identity.tracker_id = -1
        identity.tracker_id = tracker_id
        self.tracker_to_global[tracker_id] = global_id

    def _cleanup_stale_tracker_mappings(self):
        expired_mappings = [
            tracker_id
            for tracker_id, global_id in self.tracker_to_global.items()
            if global_id not in self.identities
        ]
        for tracker_id in expired_mappings:
            del self.tracker_to_global[tracker_id]

    def _make_room(self, frame_number, protected):
        while len(self.identities) >= self.max_identities:
            inactive = [
                (identity.last_seen_frame, global_id)
                for global_id, identity in self.identities.items()
                if identity.last_seen_frame < frame_number
                and global_id not in protected
            ]
            if not inactive:
                raise RuntimeError(
                    "Global identity registry is full and all identities are visible."
                )
            _, global_id = min(inactive)
            identity = self.identities.pop(global_id)
            if self.tracker_to_global.get(identity.tracker_id) == global_id:
                del self.tracker_to_global[identity.tracker_id]

    def checkpoint_state(self):
        """Return a versioned snapshot containing all historical identities."""
        state = deepcopy(self.__dict__)
        return {"version": IDENTITY_CHECKPOINT_VERSION, "state": state}

    def restore_checkpoint_state(self, checkpoint):
        """Validate and restore a complete identity-manager checkpoint."""
        if not isinstance(checkpoint, dict):
            raise ValueError("Global identity checkpoint has invalid state.")
        if "version" in checkpoint:
            if checkpoint.get("version") != IDENTITY_CHECKPOINT_VERSION:
                raise ValueError(
                    "Unsupported global identity checkpoint version."
                )
            state = checkpoint.get("state")
        else:
            # Older pipeline checkpoints stored the manager __dict__ directly.
            state = checkpoint
        if state is None:
            raise ValueError(
                "Global identity checkpoint has invalid state."
            )
        required = {
            "match_threshold",
            "ambiguity_margin",
            "minimum_confidence",
            "next_global_id",
            "identities",
            "tracker_to_global",
            "last_frame",
            "camera_epoch",
            "match_events",
            "created_ids",
            "pending_tracker_ids",
        }
        if isinstance(state, dict):
            state.setdefault("pending_tracker_ids", set())
        if not isinstance(state, dict) or not required.issubset(state):
            raise ValueError("Global identity checkpoint has invalid state.")
        for setting in (
            "match_threshold",
            "ambiguity_margin",
            "minimum_confidence",
            "max_identities",
            "max_lost_frames",
        ):
            saved_setting = state.get(setting, getattr(self, setting))
            if saved_setting != getattr(self, setting):
                raise ValueError(
                    "Global identity checkpoint settings do not match."
                )
            state[setting] = saved_setting
        identities = state["identities"]
        tracker_to_global = state["tracker_to_global"]
        if (
            not isinstance(identities, dict)
            or not isinstance(tracker_to_global, dict)
            or any(
                not isinstance(global_id, int)
                or not isinstance(identity, _PlayerIdentity)
                or identity.global_id != global_id
                for global_id, identity in identities.items()
            )
            or any(
                not isinstance(tracker_id, int)
                or global_id not in identities
                for tracker_id, global_id in tracker_to_global.items()
            )
            or not isinstance(state["pending_tracker_ids"], set)
            or any(
                not isinstance(tracker_id, int)
                for tracker_id in state["pending_tracker_ids"]
            )
        ):
            raise ValueError("Global identity checkpoint has invalid identities.")
        next_global_id = state["next_global_id"]
        if (
            not isinstance(next_global_id, int)
            or next_global_id <= max(identities, default=0)
            or not isinstance(state["last_frame"], int)
            or state["last_frame"] < 0
            or not isinstance(state["camera_epoch"], int)
            or not isinstance(state["match_events"], int)
            or not isinstance(state["created_ids"], int)
        ):
            raise ValueError("Global identity checkpoint has invalid counters.")
        state = {key: state[key] for key in required}
        state["max_identities"] = self.max_identities
        state["max_lost_frames"] = self.max_lost_frames
        self.__dict__.clear()
        self.__dict__.update(deepcopy(state))

    def snapshot(self):
        """Small diagnostics surface used by tests and end-of-run reporting."""
        return {
            "next_global_id": self.next_global_id,
            "retained_global_ids": len(self.identities),
            "tracker_mappings": len(self.tracker_to_global),
            "unresolved_tracker_ids": len(self.pending_tracker_ids),
            "registry_capacity": self.max_identities,
            "max_lost_frames": self.max_lost_frames,
            "created_global_ids": self.created_ids,
            "reidentified_tracker_changes": self.match_events,
        }
