"""Distance-based estimated possession from tracked ball/player positions."""

import math

from utils.team_color_utils import TeamColorAssigner


POSSESSION_DISTANCE_THRESHOLD = 70.0
POSSESSION_CONFIRMATION_FRAMES = 4
POSSESSION_BALL_GAP_GRACE_FRAMES = 3


class PossessionEstimator:
    """Estimate possession from the closest stably assigned player's foot point."""

    def __init__(
        self,
        distance_threshold=POSSESSION_DISTANCE_THRESHOLD,
        confirmation_frames=POSSESSION_CONFIRMATION_FRAMES,
    ):
        self.distance_threshold = float(distance_threshold)
        self.confirmation_frames = int(confirmation_frames)
        self.state = "Unknown"
        self.player_id = None
        self.closest_player_id = None
        self.closest_player_team = None
        self.closest_distance = None
        self.valid_candidate_count = 0
        self._pending_state = None
        self._pending_player_id = None
        self._pending_count = 0
        self.team_1_possession_frames = 0
        self.team_2_possession_frames = 0
        self.valid_outfield_possession_frames = 0
        self.loose_frames = 0
        self.unknown_frames = 0
        self.valid_ball_frames = 0
        self.processed_frames = 0
        self._ball_gap_frames = 0

    def update(self, ball_position, candidates):
        """Update the estimated state for one frame."""
        self.processed_frames += 1
        if ball_position is None:
            self.closest_player_id = None
            self.closest_player_team = None
            self.closest_distance = None
            self.valid_candidate_count = 0
            self._reset_pending()
            self._ball_gap_frames += 1
            if (
                self.state
                in (
                    TeamColorAssigner.TEAM_1,
                    TeamColorAssigner.TEAM_2,
                )
                and self._ball_gap_frames
                <= POSSESSION_BALL_GAP_GRACE_FRAMES
            ):
                self._count_state_frame(has_valid_outfield_carrier=False)
                return self._result()
            self.state = "Unknown"
            self.player_id = None
            self.unknown_frames += 1
            return self._result()

        self._ball_gap_frames = 0
        self.valid_ball_frames += 1
        ball_x, ball_y = map(float, ball_position)
        if not math.isfinite(ball_x) or not math.isfinite(ball_y):
            raise ValueError("ball_position coordinates must be finite")
        distances = []
        for candidate in candidates:
            if self._candidate_value(candidate, "object_type", "player") != "player":
                continue
            team = self._candidate_value(candidate, "player_type")
            if team not in (
                TeamColorAssigner.TEAM_1,
                TeamColorAssigner.TEAM_2,
            ):
                continue
            if self._identity_value(candidate) is None:
                continue
            foot_point = self._candidate_value(candidate, "foot_xy")
            if foot_point is None:
                foot_point = (
                    self._candidate_value(candidate, "foot_x"),
                    self._candidate_value(candidate, "foot_y"),
                )
            if foot_point[0] is None or foot_point[1] is None:
                continue
            foot_x, foot_y = map(float, foot_point)
            if not math.isfinite(foot_x) or not math.isfinite(foot_y):
                continue
            distance = math.hypot(foot_x - ball_x, foot_y - ball_y)
            distances.append((distance, candidate))
        self.valid_candidate_count = len(distances)

        closest = min(distances, key=lambda item: item[0]) if distances else None
        if closest is None:
            self.closest_player_id = None
            self.closest_player_team = None
            self.closest_distance = None
            desired_state = "Loose"
            desired_player_id = None
        else:
            self.closest_distance, closest_candidate = closest
            self.closest_player_id = self._candidate_value(
                closest_candidate, "global_id"
            )
            if self.closest_player_id is None:
                self.closest_player_id = self._candidate_value(
                    closest_candidate, "tracker_id"
                )
            self.closest_player_team = self._candidate_value(
                closest_candidate, "player_type"
            )
            if self.closest_distance <= self.distance_threshold:
                desired_state = self.closest_player_team
                desired_player_id = self.closest_player_id
            else:
                desired_state = "Loose"
                desired_player_id = None

        if self._current_player_remains_near(candidates):
            current_candidate = next(
                candidate
                for candidate in candidates
                if self._identity_value(candidate)
                == self.player_id
                and self._candidate_value(candidate, "player_type") == self.state
            )
            current_point = self._candidate_value(current_candidate, "foot_xy")
            if current_point is None:
                current_point = (
                    self._candidate_value(current_candidate, "foot_x"),
                    self._candidate_value(current_candidate, "foot_y"),
                )
            current_x, current_y = current_point
            if math.hypot(current_x - ball_x, current_y - ball_y) <= (
                self.distance_threshold * 1.25
            ):
                desired_state = self.state
                desired_player_id = self.player_id

        if desired_state == self.state and (
            desired_state not in (
                TeamColorAssigner.TEAM_1,
                TeamColorAssigner.TEAM_2,
            )
            or desired_player_id == self.player_id
        ):
            self._reset_pending()
        elif (
            desired_state == self._pending_state
            and desired_player_id == self._pending_player_id
        ):
            self._pending_count += 1
        else:
            self._pending_state = desired_state
            self._pending_player_id = desired_player_id
            self._pending_count = 1

        if self._pending_count >= self.confirmation_frames:
            self.state = self._pending_state
            self.player_id = self._pending_player_id
            self._reset_pending()

        has_valid_outfield_carrier = (
            self.closest_distance is not None
            and self.closest_distance <= self.distance_threshold
            and self.closest_player_team == self.state
            and self.state
            in (
                TeamColorAssigner.TEAM_1,
                TeamColorAssigner.TEAM_2,
            )
        )
        self._count_state_frame(
            has_valid_outfield_carrier=has_valid_outfield_carrier
        )
        return self._result()

    def _count_state_frame(self, has_valid_outfield_carrier=True):
        if self.state == TeamColorAssigner.TEAM_1:
            if has_valid_outfield_carrier:
                self.team_1_possession_frames += 1
                self.valid_outfield_possession_frames += 1
            else:
                self.unknown_frames += 1
        elif self.state == TeamColorAssigner.TEAM_2:
            if has_valid_outfield_carrier:
                self.team_2_possession_frames += 1
                self.valid_outfield_possession_frames += 1
            else:
                self.unknown_frames += 1
        elif self.state == "Loose":
            self.loose_frames += 1
        else:
            self.unknown_frames += 1

    def _current_player_remains_near(self, candidates):
        return (
            self.state in (
                TeamColorAssigner.TEAM_1,
                TeamColorAssigner.TEAM_2,
            )
            and self.player_id is not None
            and any(
                self._identity_value(candidate) == self.player_id
                and self._candidate_value(candidate, "player_type") == self.state
                for candidate in candidates
            )
        )

    @classmethod
    def _identity_value(cls, candidate):
        global_id = cls._candidate_value(candidate, "global_id")
        return (
            global_id
            if global_id is not None
            else cls._candidate_value(candidate, "tracker_id")
        )

    @staticmethod
    def _candidate_value(candidate, key, default=None):
        if isinstance(candidate, dict):
            return candidate.get(key, default)
        return getattr(candidate, key, default)

    def _reset_pending(self):
        self._pending_state = None
        self._pending_player_id = None
        self._pending_count = 0

    def _result(self):
        return {
            "state": self.state,
            "player_id": self.player_id,
            "closest_player_id": self.closest_player_id,
            "closest_player_team": self.closest_player_team,
            "closest_distance": self.closest_distance,
            "threshold": self.distance_threshold,
            "candidate_count": self.valid_candidate_count,
        }

    def percentages(self):
        processed = self.processed_frames
        return {
            "team_1": (
                100.0 * self.team_1_possession_frames / processed
                if processed
                else 0.0
            ),
            "team_2": (
                100.0 * self.team_2_possession_frames / processed
                if processed
                else 0.0
            ),
            "loose": 100.0 * self.loose_frames / processed if processed else 0.0,
            "unknown": 100.0 * self.unknown_frames / processed if processed else 0.0,
        }

    def team_control_percentages(self):
        """Match track.py's denominator: frames assigned to either team only."""
        decided_frames = (
            self.team_1_possession_frames + self.team_2_possession_frames
        )
        if not decided_frames:
            return {"team_1": 0.0, "team_2": 0.0}
        return {
            "team_1": 100.0 * self.team_1_possession_frames / decided_frames,
            "team_2": 100.0 * self.team_2_possession_frames / decided_frames,
        }
