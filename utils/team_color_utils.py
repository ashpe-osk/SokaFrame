"""Track-stable team assignment from torso-only kit colors."""

from collections import Counter, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans


def _box_iou(first, second):
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TeamColorAssigner:
    TEAM_1 = "team_1"
    TEAM_2 = "team_2"
    GOALKEEPER = "goalkeeper"
    BALL = "ball"
    REFEREE = "referee"
    OTHER = "other"
    UNKNOWN = "unknown"
    UNKNOWN_GOALKEEPER = "unknown_gk"
    TEAM_CONFLICT = "team_conflict"

    FEATURE_HISTORY_SIZE = 15
    FEATURE_REFRESH_INTERVAL_FRAMES = 10
    MIN_TEAM_CLUSTER_TRACKS = 2
    MIN_TRACKS_TO_FIT = 4
    MIN_TRACK_OBSERVATIONS_TO_FIT = 3
    TEAM_VOTE_HISTORY_SIZE = 15
    MIN_TEAM_VOTE_OBSERVATIONS = 9
    MIN_TEAM_VOTE_SHARE = 0.75
    PROVISIONAL_CONFIRMATION_FRAMES = 4
    RELABEL_CONFIRMATION_FRAMES = 10
    MIN_TEAM_DISTANCE_MARGIN = 0.025
    MAX_GOALKEEPER_COLOR_DISTANCE = 0.32
    MIN_GOALKEEPER_COLOR_MARGIN = 0.08
    REFEREE_COLOR_CONFIDENCE = 0.30
    OTHER_BUCKET_MAX_LAB_DISTANCE = 45.0
    OTHER_LABEL_HISTORY_SIZE = 30
    MIN_OTHER_LABEL_OBSERVATIONS = 5
    STRONG_TEAM_CORRECTION_SHARE = 1.0
    MAX_VISIBLE_PLAYERS_PER_TEAM = 11
    TEAM_SMOOTHING_MIN_OBSERVATIONS = 3
    TEAM_SMOOTHING_MIN_SHARE = 0.75

    TEAM_COLORS = {
        TEAM_1: (0, 255, 255),
        TEAM_2: (255, 0, 255),
        GOALKEEPER: (0, 165, 255),
        BALL: (255, 255, 255),
    }
    NEUTRAL_COLOR = (205, 205, 205)

    def __init__(self):
        self._features = {}
        self._feature_history = {}
        self._goalkeeper_features = {}
        self._goalkeeper_history = {}
        self._goalkeeper_trackers = set()
        self.last_goalkeeper_resets = {}
        self._assignments = {}
        self._team_votes = {}
        self._provisional = {}
        self._pending_assignments = {}
        self._pending_counts = {}
        self._goalkeeper_pending = {}
        self._goalkeeper_pending_counts = {}
        self._other_label_history = {}
        self._last_feature_update = {}
        self._force_feature_refresh = set()
        self._update_frame = 0
        self._referee_color_history = deque(maxlen=self.OTHER_LABEL_HISTORY_SIZE)
        self._team_centers = {}
        self._referee_color_center = None
        self._team_centers_fitted = False
        self.team_colors = dict(self.TEAM_COLORS)
        self.team_label_change_count = 0
        self.team_label_changed_ids = set()
        self.last_referee_candidates = set()
        self.last_color_distances = {}
        self.last_goalkeeper_distances = {}
        self.last_team_evidence = {}
        self.last_observed_team_evidence = {}
        self.prefit_sample_count = 0
        self.prefit_cluster_counts = ()
        self.prefit_rejected_reason = None
        self._prefit_features = []

    def update(
        self,
        frame,
        detections,
        referee_ids=(),
        enable_other_bucket=False,
        overlap_iou=0.1,
    ):
        self.last_referee_candidates = set()
        self.last_color_distances = {}
        self.last_goalkeeper_distances = {}
        self.last_observed_team_evidence = {}
        self.last_goalkeeper_resets = {}
        self._update_frame += 1
        known_referees = set(referee_ids)
        observed_outfield_ids = set()
        observed_goalkeeper_ids = set()
        sampled_outfield_ids = set()
        sampled_goalkeeper_ids = set()
        contaminated_ids = self._overlapping_player_ids(
            detections, minimum_iou=overlap_iou
        )

        for detection in detections:
            tracker_id = detection.get("tracker_id")
            if tracker_id is None:
                continue
            object_type = detection.get("object_type", "player")
            if tracker_id in known_referees or object_type == "referee":
                self._mark_referee(tracker_id)
                continue
            if object_type in ("ball", "other"):
                continue
            if (
                object_type == "goalkeeper"
                and tracker_id not in self._goalkeeper_trackers
            ):
                self._goalkeeper_trackers.add(tracker_id)
                previous_assignment = self._assignments.get(tracker_id)
                self.last_goalkeeper_resets[tracker_id] = (
                    previous_assignment
                )
                self.clear_goalkeeper_assignment(tracker_id)
            is_goalkeeper_track = (
                object_type == "goalkeeper"
                or tracker_id in self._goalkeeper_trackers
            )

            if tracker_id in contaminated_ids:
                if tracker_id in self._features:
                    observed_outfield_ids.add(tracker_id)
                    self._force_feature_refresh.add(tracker_id)
                if tracker_id in self._goalkeeper_features:
                    observed_goalkeeper_ids.add(tracker_id)
                    self._force_feature_refresh.add(tracker_id)
                continue

            last_feature_update = self._last_feature_update.get(tracker_id)
            has_stable_assignment = self._assignments.get(tracker_id) in (
                self.TEAM_1,
                self.TEAM_2,
                self.OTHER,
            )
            cached_feature = (
                self._goalkeeper_features.get(tracker_id)
                if is_goalkeeper_track
                else self._features.get(tracker_id)
            )
            if (
                has_stable_assignment
                and cached_feature is not None
                and tracker_id not in self._force_feature_refresh
                and last_feature_update is not None
                and self._update_frame - last_feature_update
                < self.FEATURE_REFRESH_INTERVAL_FRAMES
            ):
                if is_goalkeeper_track:
                    observed_goalkeeper_ids.add(tracker_id)
                else:
                    observed_outfield_ids.add(tracker_id)
                continue

            feature = self._jersey_feature(
                frame,
                detection["bbox"],
                central_torso=enable_other_bucket and not is_goalkeeper_track,
            )
            if feature is None:
                continue
            self._last_feature_update[tracker_id] = self._update_frame
            self._force_feature_refresh.discard(tracker_id)
            if is_goalkeeper_track:
                self._update_history(
                    self._goalkeeper_history, tracker_id, feature
                )
                self._goalkeeper_features[tracker_id] = self._average_features(
                    self._goalkeeper_history[tracker_id]
                )
                observed_goalkeeper_ids.add(tracker_id)
                sampled_goalkeeper_ids.add(tracker_id)
                self._features.pop(tracker_id, None)
                self._feature_history.pop(tracker_id, None)
                continue

            self._update_history(self._feature_history, tracker_id, feature)
            self._features[tracker_id] = self._average_features(
                self._feature_history[tracker_id]
            )
            observed_outfield_ids.add(tracker_id)
            sampled_outfield_ids.add(tracker_id)

        for tracker_id in known_referees:
            self._mark_referee(tracker_id)

        self._fit_team_centers_once()
        self.last_color_distances = {
            tracker_id: self._team_chroma_distances(
                self._features[tracker_id]
            )
            for tracker_id in observed_outfield_ids
        }
        self.last_goalkeeper_distances = {
            tracker_id: self._team_distances(
                self._goalkeeper_features[tracker_id]
            )
            for tracker_id in observed_goalkeeper_ids
        }
        if enable_other_bucket and self._team_centers_fitted:
            self._update_other_bucket_assignments(sampled_outfield_ids)
            self._update_goalkeeper_assignments(sampled_goalkeeper_ids)
        elif self._team_centers_fitted:
            self._update_team_assignments(sampled_outfield_ids)
        self.last_team_evidence = {}
        maximum_team_distance = (
            self.OTHER_BUCKET_MAX_LAB_DISTANCE / 255.0
            if enable_other_bucket
            else None
        )
        for tracker_id in observed_outfield_ids:
            evidence = self._team_evidence(
                self.last_color_distances.get(tracker_id, {}),
                self.MIN_TEAM_DISTANCE_MARGIN,
                maximum_distance=maximum_team_distance,
            )
            if evidence is not None:
                self.last_observed_team_evidence[tracker_id] = evidence
        for tracker_id in sampled_outfield_ids:
            evidence = self.last_observed_team_evidence.get(tracker_id)
            if evidence is not None:
                self.last_team_evidence[tracker_id] = evidence
        for tracker_id in sampled_goalkeeper_ids:
            evidence = self._team_evidence(
                self.last_goalkeeper_distances.get(tracker_id, {}),
                self.MIN_GOALKEEPER_COLOR_MARGIN,
                maximum_distance=self.MAX_GOALKEEPER_COLOR_DISTANCE,
            )
            if evidence is not None:
                self.last_team_evidence[tracker_id] = evidence
        if not enable_other_bucket:
            self._update_goalkeeper_assignments(sampled_goalkeeper_ids)
        self._enforce_team_roster_cap(
            observed_outfield_ids | observed_goalkeeper_ids
        )
        return {
            tracker_id: self._assignments.get(
                tracker_id,
                self.UNKNOWN_GOALKEEPER
                if (
                    detection.get("object_type") == "goalkeeper"
                    or tracker_id in self._goalkeeper_trackers
                )
                else self.UNKNOWN,
            )
            for detection in detections
            if (tracker_id := detection.get("tracker_id")) is not None
        }

    @property
    def team_centers_fitted(self):
        return self._team_centers_fitted

    def is_goalkeeper_tracker(self, tracker_id):
        return tracker_id in self._goalkeeper_trackers

    def prefit_observe(self, frame, bbox):
        feature = self._team_torso_lab_feature(frame, bbox)
        if feature is not None:
            self._prefit_features.append(feature)
        return feature is not None

    def prefit_finalize(self):
        """Fit both team centers from an offline sample when it is balanced."""
        self.prefit_sample_count = len(self._prefit_features)
        self.prefit_rejected_reason = None
        if len(self._prefit_features) < self.MIN_TRACKS_TO_FIT:
            self.prefit_rejected_reason = "too_few_samples"
            return False

        samples = np.stack(self._prefit_features)
        chroma_samples = samples[:, 1:]
        if len(np.unique(np.round(chroma_samples, decimals=4), axis=0)) < 2:
            self.prefit_rejected_reason = "insufficient_chroma_variation"
            return False

        clustering = KMeans(n_clusters=2, n_init=10, random_state=0).fit(
            chroma_samples
        )
        counts = tuple(
            int(np.count_nonzero(clustering.labels_ == cluster_id))
            for cluster_id in range(2)
        )
        self.prefit_cluster_counts = counts
        if min(counts) < 0.25 * len(samples):
            self.prefit_rejected_reason = "cluster_below_25_percent"
            return False

        cluster_teams = self._map_clusters_to_teams(
            clustering.cluster_centers_
        )
        self._team_centers = {
            cluster_teams[cluster_id]: np.median(
                samples[clustering.labels_ == cluster_id], axis=0
            )
            for cluster_id in range(2)
        }
        self._team_centers_fitted = True
        return True

    def inherit_assignment(self, tracker_id, team):
        """Carry a confirmed team color across a verified Global ID relink."""
        if team not in (self.TEAM_1, self.TEAM_2):
            return
        self._assignments[tracker_id] = team
        self._provisional.pop(tracker_id, None)
        self._pending_assignments.pop(tracker_id, None)
        self._pending_counts.pop(tracker_id, None)

    @staticmethod
    def _overlapping_player_ids(detections, minimum_iou):
        players = [
            detection
            for detection in detections
            if detection.get("tracker_id") is not None
            and detection.get("object_type", "player")
            in ("player", "goalkeeper")
        ]
        contaminated = set()
        for index, first in enumerate(players):
            for second in players[index + 1 :]:
                if _box_iou(first["bbox"], second["bbox"]) > minimum_iou:
                    contaminated.add(first["tracker_id"])
                    contaminated.add(second["tracker_id"])
        return contaminated

    def is_provisional(self, tracker_id):
        return tracker_id in self._provisional

    @staticmethod
    def _update_history(histories, tracker_id, feature):
        history = histories.setdefault(
            tracker_id, deque(maxlen=TeamColorAssigner.FEATURE_HISTORY_SIZE)
        )
        history.append(feature)

    @staticmethod
    def _average_features(features):
        return np.median(np.stack(tuple(features)), axis=0).astype(np.float64)

    def _team_distances(self, feature):
        return {
            team: float(np.linalg.norm(feature - center))
            for team, center in self._team_centers.items()
        }

    def _team_chroma_distances(self, feature):
        return {
            team: float(np.linalg.norm(feature[1:] - center[1:]))
            for team, center in self._team_centers.items()
        }

    @staticmethod
    def _team_evidence(distances, minimum_margin, maximum_distance=None):
        team_distances = {
            team: distances[team]
            for team in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2)
            if team in distances
        }
        if len(team_distances) != 2:
            return None
        nearest, second = sorted(team_distances, key=team_distances.get)
        nearest_distance = team_distances[nearest]
        margin = team_distances[second] - nearest_distance
        if margin < minimum_margin or (
            maximum_distance is not None
            and nearest_distance > maximum_distance
        ):
            return None
        relative_margin = margin / (
            team_distances[nearest] + team_distances[second] + 1e-8
        )
        return nearest, min(1.0, max(0.0, relative_margin))

    def _mark_referee(self, tracker_id):
        self._assignments[tracker_id] = self.REFEREE
        self._goalkeeper_trackers.discard(tracker_id)
        self.last_goalkeeper_resets.pop(tracker_id, None)
        self._features.pop(tracker_id, None)
        self._feature_history.pop(tracker_id, None)
        self._goalkeeper_features.pop(tracker_id, None)
        self._goalkeeper_history.pop(tracker_id, None)
        self._team_votes.pop(tracker_id, None)
        self._last_feature_update.pop(tracker_id, None)
        self._force_feature_refresh.discard(tracker_id)
        self._provisional.pop(tracker_id, None)
        self._pending_assignments.pop(tracker_id, None)
        self._pending_counts.pop(tracker_id, None)
        self._goalkeeper_pending.pop(tracker_id, None)
        self._goalkeeper_pending_counts.pop(tracker_id, None)
        self._other_label_history.pop(tracker_id, None)

    def clear_goalkeeper_assignment(self, tracker_id):
        """Clear outfield team evidence after a track is recognized as a goalkeeper."""
        self._assignments.pop(tracker_id, None)
        self._features.pop(tracker_id, None)
        self._feature_history.pop(tracker_id, None)
        self._team_votes.pop(tracker_id, None)
        self._other_label_history.pop(tracker_id, None)
        self._provisional.pop(tracker_id, None)
        self._pending_assignments.pop(tracker_id, None)
        self._pending_counts.pop(tracker_id, None)
        self._last_feature_update.pop(tracker_id, None)
        self._force_feature_refresh.discard(tracker_id)

    def mark_referee(self, tracker_id):
        self._mark_referee(tracker_id)

    def observe_referee_detections(self, frame, detections):
        for detection in detections:
            if detection.get("confidence", 0.0) < self.REFEREE_COLOR_CONFIDENCE:
                continue
            feature = self._torso_lab_feature(frame, detection["bbox"])
            if feature is not None:
                self._referee_color_history.append(feature)
        if self._referee_color_history:
            self._referee_color_center = np.median(
                np.stack(tuple(self._referee_color_history)),
                axis=0,
            ).astype(np.float64)

    def _set_team_assignment(self, tracker_id, team):
        current = self._assignments.get(tracker_id)
        if (
            current in (self.TEAM_1, self.TEAM_2)
            and team in (self.TEAM_1, self.TEAM_2)
            and current != team
        ):
            self.team_label_change_count += 1
            self.team_label_changed_ids.add(tracker_id)
        self._assignments[tracker_id] = team

    def _fit_team_centers_once(self):
        if self._team_centers_fitted:
            return
        eligible_ids = [
            tracker_id
            for tracker_id, history in self._feature_history.items()
            if len(history) >= self.MIN_TRACK_OBSERVATIONS_TO_FIT
        ]
        if len(eligible_ids) < self.MIN_TRACKS_TO_FIT:
            return

        samples = np.stack([self._features[tracker_id] for tracker_id in eligible_ids])
        chroma_samples = samples[:, 1:]
        if len(np.unique(np.round(chroma_samples, decimals=4), axis=0)) < 2:
            return
        clustering = KMeans(n_clusters=2, n_init=10, random_state=0).fit(
            chroma_samples
        )
        if any(
            int(np.count_nonzero(clustering.labels_ == cluster_id))
            < self.MIN_TEAM_CLUSTER_TRACKS
            for cluster_id in range(2)
        ):
            return

        cluster_teams = self._map_clusters_to_teams(
            clustering.cluster_centers_
        )
        self._team_centers = {
            cluster_teams[cluster_id]: np.median(
                samples[clustering.labels_ == cluster_id], axis=0
            )
            for cluster_id in range(2)
        }
        self._team_centers_fitted = True

    def _map_clusters_to_teams(self, centers):
        cluster_ids = sorted(
            range(len(centers)),
            key=lambda cluster_id: centers[cluster_id][-1],
        )
        return {
            cluster_ids[0]: self.TEAM_1,
            cluster_ids[1]: self.TEAM_2,
        }

    def _update_team_assignments(self, observed_ids):
        for tracker_id in observed_ids:
            distances = self._team_distances(self._features[tracker_id])
            if len(distances) != 2:
                continue
            ranked = sorted(distances, key=distances.get)
            proposed = ranked[0]
            if distances[ranked[1]] - distances[proposed] < self.MIN_TEAM_DISTANCE_MARGIN:
                continue
            current = self._assignments.get(tracker_id)
            self._consider_assignment(
                tracker_id,
                proposed,
                confirmation_frames=(
                    self.RELABEL_CONFIRMATION_FRAMES
                    if current in (self.TEAM_1, self.TEAM_2)
                    and current != proposed
                    else self.PROVISIONAL_CONFIRMATION_FRAMES
                ),
            )

    def _update_other_bucket_assignments(self, observed_ids):
        for tracker_id in observed_ids:
            feature = self._features.get(
                tracker_id, self._goalkeeper_features.get(tracker_id)
            )
            if feature is None:
                continue
            lab_distances = self._team_distances(feature)
            team_distances = {
                team: lab_distances[team]
                for team in (self.TEAM_1, self.TEAM_2)
                if team in lab_distances
            }
            if len(team_distances) != 2:
                continue
            ranked_teams = sorted(team_distances, key=team_distances.get)
            nearest_team, second_team = ranked_teams
            team_margin = (
                team_distances[second_team] - team_distances[nearest_team]
            )
            if self._referee_color_center is not None:
                lab_distances[self.REFEREE] = float(
                    np.linalg.norm(feature - self._referee_color_center)
                )
            current = self._assignments.get(tracker_id)
            if current == self.REFEREE:
                continue
            nearest_distance = team_distances[nearest_team]
            referee_distance = lab_distances.get(self.REFEREE)
            votes = self._other_label_history.setdefault(
                tracker_id, deque(maxlen=self.OTHER_LABEL_HISTORY_SIZE)
            )
            if (
                referee_distance is not None
                and referee_distance < nearest_distance
                and referee_distance * 255.0
                <= self.OTHER_BUCKET_MAX_LAB_DISTANCE
            ):
                votes.append(self.REFEREE)
                self._consider_assignment(
                    tracker_id,
                    self.REFEREE,
                    confirmation_frames=self.MIN_OTHER_LABEL_OBSERVATIONS,
                )
                continue
            if nearest_distance * 255.0 > self.OTHER_BUCKET_MAX_LAB_DISTANCE:
                votes.append(self.OTHER)
                self._consider_assignment(tracker_id, self.OTHER)
                continue
            if team_margin < self.MIN_TEAM_DISTANCE_MARGIN:
                self._provisional.pop(tracker_id, None)
                self._pending_assignments.pop(tracker_id, None)
                self._pending_counts.pop(tracker_id, None)
                continue
            votes.append(nearest_team)
            self._consider_assignment(
                tracker_id,
                nearest_team,
                confirmation_frames=(
                    self.RELABEL_CONFIRMATION_FRAMES
                    if current in (self.TEAM_1, self.TEAM_2)
                    and current != nearest_team
                    else self.PROVISIONAL_CONFIRMATION_FRAMES
                ),
            )

    def _consider_assignment(
        self, tracker_id, candidate, confirmation_frames=None
    ):
        current = self._assignments.get(tracker_id)
        if current == candidate:
            self._provisional.pop(tracker_id, None)
            self._pending_assignments.pop(tracker_id, None)
            self._pending_counts.pop(tracker_id, None)
            return
        if candidate == self.OTHER and current in (self.TEAM_1, self.TEAM_2):
            return
        if current == self.OTHER and candidate in (self.TEAM_1, self.TEAM_2):
            confirmation_frames = max(
                self.OTHER_LABEL_HISTORY_SIZE,
                confirmation_frames or 0,
            )
        if confirmation_frames is None:
            confirmation_frames = (
                self.MIN_OTHER_LABEL_OBSERVATIONS
                if candidate == self.OTHER
                else self.PROVISIONAL_CONFIRMATION_FRAMES
            )
        if self._pending_assignments.get(tracker_id) == candidate:
            self._pending_counts[tracker_id] += 1
        else:
            self._pending_assignments[tracker_id] = candidate
            self._pending_counts[tracker_id] = 1
        self._provisional[tracker_id] = candidate
        if self._pending_counts[tracker_id] >= confirmation_frames:
            self._set_team_assignment(tracker_id, candidate)
            self._provisional.pop(tracker_id, None)
            self._pending_assignments.pop(tracker_id, None)
            self._pending_counts.pop(tracker_id, None)

    def _update_goalkeeper_assignments(self, observed_ids):
        # Goalkeeper kits do not reliably identify their outfield team.
        for tracker_id in observed_ids:
            self._assignments.pop(tracker_id, None)
            self._goalkeeper_pending.pop(tracker_id, None)
            self._goalkeeper_pending_counts.pop(tracker_id, None)
            self._provisional.pop(tracker_id, None)
            self._pending_assignments.pop(tracker_id, None)
            self._pending_counts.pop(tracker_id, None)

    def _enforce_team_roster_cap(self, observed_ids):
        for team in (self.TEAM_1, self.TEAM_2):
            assigned = [
                tracker_id
                for tracker_id in observed_ids
                if self._assignments.get(tracker_id) == team
            ]
            excess = len(assigned) - self.MAX_VISIBLE_PLAYERS_PER_TEAM
            if excess <= 0:
                continue

            def margin(tracker_id):
                feature = self._features.get(
                    tracker_id, self._goalkeeper_features.get(tracker_id)
                )
                if feature is None:
                    return float("inf")
                distances = self._team_distances(feature)
                if len(distances) != 2:
                    return float("inf")
                first, second = sorted(distances.values())
                return second - first

            for tracker_id in sorted(assigned, key=margin)[:excess]:
                self._assignments.pop(tracker_id, None)
                self._pending_assignments.pop(tracker_id, None)
                self._pending_counts.pop(tracker_id, None)
                self._provisional[tracker_id] = team

    @classmethod
    def _jersey_feature(cls, frame, bbox, central_torso=False):
        if central_torso:
            return cls._team_torso_lab_feature(frame, bbox)
        torso = cls._upper_body_bgr(frame, bbox)
        if torso is None:
            return None

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        height, width = hsv.shape[:2]
        hue, saturation, value = cv2.split(hsv)
        x_grid = np.broadcast_to(np.arange(width), (height, width))
        y_grid = np.broadcast_to(np.arange(height)[:, None], (height, width))
        green_hue = (hue >= 25) & (hue <= 95) & (saturation >= 45)
        crop_edge = (
            (x_grid < max(1, int(width * 0.10)))
            | (x_grid >= int(width * 0.90))
            | (y_grid < max(1, int(height * 0.08)))
            | (y_grid >= int(height * 0.82))
        )
        grass_pixels = green_hue & crop_edge
        background_grass = cls._background_grass_lab(frame, bbox)
        if background_grass is not None:
            lab_distance = np.linalg.norm(
                lab.astype(np.float32) / 255.0 - background_grass,
                axis=2,
            )
            grass_pixels |= green_hue & (lab_distance <= 0.12)
        valid = (value >= 20) & ~grass_pixels
        if int(np.count_nonzero(valid)) < max(10, int(width * height * 0.15)):
            return None

        feature = np.median(lab[valid], axis=0).astype(np.float64) / 255.0
        return feature

    @classmethod
    def _torso_lab_feature(cls, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        if width < 2 or height < 2:
            return None
        x_center = (x1 + x2) // 2
        half_width = max(1, int(round(width * 0.25)))
        left = max(0, x_center - half_width)
        right = min(frame.shape[1], x_center + half_width)
        top = max(0, y1 + int(round(height * 0.20)))
        bottom = min(frame.shape[0], y1 + int(round(height * 0.55)))
        if left >= right or top >= bottom:
            return None

        crop = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hue, saturation = cv2.split(hsv)[:2]
        grass = (hue >= 30) & (hue <= 90) & (saturation > 40)
        valid = ~grass
        if not np.any(valid):
            return None
        return np.median(lab[valid], axis=0).astype(np.float64) / 255.0

    @staticmethod
    def _team_torso_lab_feature(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        if width < 2 or height < 2:
            return None
        x_center = (x1 + x2) // 2
        half_width = max(1, int(round(width * 0.25)))
        left = max(0, x_center - half_width)
        right = min(frame.shape[1], x_center + half_width)
        top = max(0, y1 + int(round(height * 0.20)))
        bottom = min(frame.shape[0], y1 + int(round(height * 0.55)))
        if left >= right or top >= bottom:
            return None

        torso = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)
        hue, saturation, value = cv2.split(hsv)
        torso_height, torso_width = hue.shape
        x_grid = np.broadcast_to(
            np.arange(torso_width), (torso_height, torso_width)
        )
        y_grid = np.broadcast_to(
            np.arange(torso_height)[:, None], (torso_height, torso_width)
        )
        green_hue = (hue >= 25) & (hue <= 95) & (saturation >= 45)
        crop_edge = (
            (x_grid < max(1, int(torso_width * 0.10)))
            | (x_grid >= int(torso_width * 0.90))
            | (y_grid < max(1, int(torso_height * 0.08)))
            | (y_grid >= int(torso_height * 0.82))
        )
        grass_pixels = green_hue & crop_edge
        background_grass = TeamColorAssigner._background_grass_lab(
            frame, bbox
        )
        if background_grass is not None:
            lab_distance = np.linalg.norm(
                lab.astype(np.float32) / 255.0 - background_grass,
                axis=2,
            )
            grass_pixels |= green_hue & (lab_distance <= 0.12)
        valid = (value >= 20) & ~grass_pixels
        if int(np.count_nonzero(valid)) < max(
            10, int(torso_width * torso_height * 0.15)
        ):
            return None
        return np.median(lab[valid], axis=0).astype(np.float64) / 255.0

    @staticmethod
    def _background_grass_lab(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        if width < 2 or height < 2:
            return None
        pad = max(2, int(round(width * 0.12)))
        strips = []
        if y2 < frame.shape[0]:
            strips.append(
                frame[y2 : min(frame.shape[0], y2 + pad), max(0, x1) : min(frame.shape[1], x2)]
            )
        strips = [strip for strip in strips if strip.size]
        if not strips:
            return None

        context = np.concatenate(
            [strip.reshape(-1, 1, 3) for strip in strips],
            axis=0,
        )
        hsv = cv2.cvtColor(context, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        green = (
            (hsv[:, 0] >= 25)
            & (hsv[:, 0] <= 95)
            & (hsv[:, 1] >= 45)
            & (hsv[:, 2] >= 20)
        )
        if np.count_nonzero(green) < 10:
            return None
        lab = cv2.cvtColor(context, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(lab[green], axis=0).astype(np.float32) / 255.0

    @staticmethod
    def _upper_body_bgr(frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1
        if width < 2 or height < 2:
            return None

        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        center_x = (x1 + x2) // 2
        half_width = max(1, int(round((x2 - x1) * 0.30)))
        x1 = max(x1, center_x - half_width)
        x2 = min(x2, center_x + half_width)
        upper_y1 = max(0, y1 + int(height * 0.18))
        upper_y2 = min(frame.shape[0], y1 + int(height * 0.58))
        if x1 >= x2 or upper_y1 >= upper_y2:
            return None
        return frame[upper_y1:upper_y2, x1:x2]


class CanonicalTeamSmoother:
    """Smooth canonical-player team labels using relative color margins."""

    HISTORY_SIZE = 15
    MIN_OBSERVATIONS = 3
    MIN_WEIGHTED_SHARE = 0.75

    def __init__(self):
        self._history = {}
        self._stable_teams = {}

    def update(self, global_id, frame_number, evidence):
        if global_id is None or evidence is None:
            return self._stable_teams.get(global_id)
        stable_team = self._stable_teams.get(int(global_id))
        team, confidence = evidence
        if team not in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2):
            return self._stable_teams.get(global_id)
        confidence = float(confidence)
        if not np.isfinite(confidence) or confidence <= 0.0:
            return self._stable_teams.get(global_id)
        confidence = min(1.0, confidence)
        frame_number = int(frame_number)
        history = self._history.setdefault(
            int(global_id), deque(maxlen=self.HISTORY_SIZE)
        )
        if history and frame_number < history[-1][0]:
            raise ValueError("frame_number must be non-decreasing per identity")
        item = (frame_number, team, confidence)
        if history and frame_number == history[-1][0]:
            history[-1] = item
        else:
            history.append(item)

        if len(history) < self.MIN_OBSERVATIONS:
            return self._stable_teams.get(global_id)
        weights = {
            candidate: sum(
                weight
                for _, observed, weight in history
                if observed == candidate
            )
            for candidate in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2)
        }
        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            return self._stable_teams.get(global_id)
        winner = max(weights, key=weights.get)
        if weights[winner] / total_weight >= self.MIN_WEIGHTED_SHARE:
            self._stable_teams[int(global_id)] = winner
        return self._stable_teams.get(int(global_id))

    def clear(self, global_id):
        """Discard team evidence when an identity is recognized as a goalkeeper."""
        self._history.pop(int(global_id), None)
        self._stable_teams.pop(int(global_id), None)


class CanonicalTeamConflictTracker:
    """Flag sustained team-color conflicts without changing canonical IDs."""

    CONFIRMATION_FRAMES = 30
    RECOVERY_FRAMES = 30
    MIN_EVIDENCE_CONFIDENCE = 0.35
    MAX_EVIDENCE_GAP_FRAMES = 5

    def __init__(self):
        self._canonical_teams = {}
        self._states = {}

    def update(self, global_id, canonical_team, frame_number, evidence):
        if global_id is None or canonical_team not in (
            TeamColorAssigner.TEAM_1,
            TeamColorAssigner.TEAM_2,
        ):
            return False

        global_id = int(global_id)
        frame_number = int(frame_number)
        canonical_team = self._canonical_teams.setdefault(
            global_id, canonical_team
        )
        state = self._states.setdefault(
            global_id,
            {
                "conflicted": False,
                "opposing_team": None,
                "opposing_count": 0,
                "agreement_count": 0,
                "last_frame": None,
            },
        )
        last_frame = state["last_frame"]
        if last_frame is not None and frame_number < last_frame:
            raise ValueError(
                "frame_number must be non-decreasing per identity"
            )
        if (
            last_frame is not None
            and frame_number - last_frame > self.MAX_EVIDENCE_GAP_FRAMES
        ):
            state["opposing_team"] = None
            state["opposing_count"] = 0
            state["agreement_count"] = 0
        state["last_frame"] = frame_number

        if evidence is None:
            state["opposing_team"] = None
            state["opposing_count"] = 0
            state["agreement_count"] = 0
            return state["conflicted"]
        if not isinstance(evidence, (tuple, list)) or len(evidence) != 2:
            raise ValueError("team evidence must contain a label and confidence")

        observed_team, confidence = evidence
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as error:
            raise ValueError("team evidence confidence must be numeric") from error
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "team evidence confidence must be finite and between 0 and 1"
            )
        if observed_team not in (
            TeamColorAssigner.TEAM_1,
            TeamColorAssigner.TEAM_2,
        ) or confidence < self.MIN_EVIDENCE_CONFIDENCE:
            state["opposing_team"] = None
            state["opposing_count"] = 0
            state["agreement_count"] = 0
            return state["conflicted"]

        if observed_team == canonical_team:
            state["opposing_team"] = None
            state["opposing_count"] = 0
            if state["conflicted"]:
                state["agreement_count"] += 1
                if state["agreement_count"] >= self.RECOVERY_FRAMES:
                    state["conflicted"] = False
                    state["agreement_count"] = 0
            return state["conflicted"]

        state["agreement_count"] = 0
        if state["opposing_team"] == observed_team:
            state["opposing_count"] += 1
        else:
            state["opposing_team"] = observed_team
            state["opposing_count"] = 1
        if state["opposing_count"] >= self.CONFIRMATION_FRAMES:
            state["conflicted"] = True
        return state["conflicted"]

    def clear(self, global_id):
        if global_id is None:
            return
        global_id = int(global_id)
        self._canonical_teams.pop(global_id, None)
        self._states.pop(global_id, None)
