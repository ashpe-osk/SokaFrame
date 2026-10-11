"""Evidence-based pass-attempt detection from ball and confirmed possession data."""

import math


class EvidenceBasedPassDetector:
    """Count supported releases and resolve them from subsequent possession evidence."""

    TEAM_STATES = {"team_1", "team_2"}
    RELEASE_SEPARATION_MULTIPLIER = 1.0
    MIN_RELEASE_TRAVEL_RATIO = 0.04
    MAX_OUTCOME_WAIT_SECONDS = 2.0
    MAX_POSSESSION_GAP_SECONDS = 0.75
    SAME_PLAYER_CONTROL_FRAMES = 3

    def __init__(self, fps, image_width=1280):
        self.fps = float(fps)
        self.image_width = int(image_width)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if self.image_width <= 0:
            raise ValueError("image_width must be positive")
        self.player_proximity_threshold = max(1.0, self.image_width * 0.05)
        self.minimum_release_travel = max(
            1.0, self.image_width * self.MIN_RELEASE_TRAVEL_RATIO
        )
        self.release_separation_threshold = (
            self.player_proximity_threshold * self.RELEASE_SEPARATION_MULTIPLIER
        )
        self.passes = []
        self.team_counts = {
            "team_1": {"attempted": 0, "completed": 0, "incomplete": 0, "unknown": 0},
            "team_2": {"attempted": 0, "completed": 0, "incomplete": 0, "unknown": 0},
        }
        self._last_control = None
        self._pending_attempt = None
        self._last_confirmed_possession_time = None
        self._last_frame = None

    @property
    def counts(self):
        return self.team_counts

    @staticmethod
    def _coordinate(value):
        if value is None:
            return None
        try:
            x, y = value
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None

    def _valid_ball(self, ball_position, ball_source, ball_is_valid):
        coordinates = self._coordinate(ball_position)
        valid = (
            coordinates is not None
            and ball_source in {"real", "interpolated"}
            and bool(ball_is_valid)
        )
        return coordinates if valid else None

    @staticmethod
    def _player_id(player):
        if isinstance(player, dict):
            player_id = player.get("global_id")
            return player_id if player_id not in (None, "") else player.get("track_id")
        player_id = getattr(player, "global_id", None)
        return (
            player_id
            if player_id not in (None, "")
            else getattr(player, "track_id", None)
        )

    @staticmethod
    def _player_team(player):
        team = (
            player.get("player_type")
            if isinstance(player, dict)
            else getattr(player, "player_type", None)
        )
        if team is None:
            team = (
                player.get("team")
                if isinstance(player, dict)
                else getattr(player, "team", None)
            )
        return str(team) if team is not None else None

    @staticmethod
    def _player_position(player):
        x = (
            player.get("foot_x")
            if isinstance(player, dict)
            else getattr(player, "foot_x", None)
        )
        y = (
            player.get("foot_y")
            if isinstance(player, dict)
            else getattr(player, "foot_y", None)
        )
        if x is None:
            x = (
                player.get("image_x")
                if isinstance(player, dict)
                else getattr(player, "image_x", None)
            )
        if y is None:
            y = (
                player.get("image_y")
                if isinstance(player, dict)
                else getattr(player, "image_y", None)
            )
        if x is None or y is None:
            return None
        try:
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return None
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None

    def _possession_input(self, players, possession_state, possession_player_id):
        if possession_state is None:
            states = {
                self._value(player, "possession_state")
                for player in players
                if self._value(player, "possession_state") is not None
            }
            possession_state = next(iter(states)) if len(states) == 1 else None

        if possession_player_id is None:
            player_ids = {
                self._value(player, "possession_player_id")
                for player in players
                if self._value(player, "possession_player_id") is not None
            }
            possession_player_id = next(iter(player_ids)) if len(player_ids) == 1 else None

        return possession_state, possession_player_id

    def _confirmed_owner(self, players, possession_state, possession_player_id):
        if possession_state not in self.TEAM_STATES or possession_player_id is None:
            return None
        for player in players:
            player_id = self._player_id(player)
            if str(player_id) != str(possession_player_id):
                continue
            if self._player_team(player) != possession_state:
                return None
            object_type = self._value(player, "object_type", "player")
            visible = self._value(player, "is_visible", True)
            if object_type != "player" or str(visible).lower() in (
                "0",
                "false",
                "no",
            ):
                return None
            position = self._player_position(player)
            if position is None:
                return None
            return {
                "player_id": player_id,
                "team": possession_state,
                "position": position,
            }
        return None

    def _set_outcome(self, attempt, outcome, reason, receiver=None):
        team_counts = self.team_counts[attempt["team"]]
        team_counts["unknown"] -= 1
        team_counts[outcome] += 1
        attempt["outcome"] = outcome
        attempt["outcome_reason"] = reason

        if receiver is not None:
            attempt["receiver_id"] = receiver["player_id"]

    def _update_attempt_endpoint(
        self, frame_number, timestamp_seconds, ball_x, ball_y, valid_ball
    ):
        pending = self._pending_attempt
        if pending is None:
            return
        attempt = pending["record"]
        attempt["end_frame"] = frame_number
        attempt["end_timestamp_seconds"] = timestamp_seconds
        if valid_ball:
            attempt["end_x_image"] = ball_x
            attempt["end_y_image"] = ball_y

    def _finish_pending(
        self, frame_number, timestamp_seconds, ball_x=None, ball_y=None, valid_ball=False
    ):
        pending = self._pending_attempt
        if pending is None:
            return
        attempt = pending["record"]
        self._update_attempt_endpoint(
            frame_number, timestamp_seconds, ball_x, ball_y, valid_ball
        )
        self._set_outcome(attempt, "unknown", "outcome_unresolved")
        self._pending_attempt = None
        self._last_control = None

    def _start_attempt(self, owner, frame_number, timestamp_seconds, ball_x, ball_y, travel):
        attempt = {
            "passer_id": owner["player_id"],
            "team": owner["team"],
            "receiver_id": None,
            "pass_id": len(self.passes) + 1,
            "start_frame": frame_number,
            "end_frame": frame_number,
            "start_timestamp_seconds": timestamp_seconds,
            "end_timestamp_seconds": timestamp_seconds,
            "start_x_image": ball_x,
            "start_y_image": ball_y,
            "end_x_image": ball_x,
            "end_y_image": ball_y,
            "start_x_pitch": "",
            "start_y_pitch": "",
            "end_x_pitch": "",
            "end_y_pitch": "",
            "coordinate_space": "image_pixels",
            "pitch_coordinates_status": "uncalibrated",
            "outcome": "unknown",
            "outcome_reason": "release_detected_pending_outcome",
            "confidence": min(0.9, 0.5 + travel / max(1.0, self.image_width)),
        }
        self.passes.append(attempt)
        counts = self.team_counts[owner["team"]]
        counts["attempted"] += 1
        counts["unknown"] += 1
        self._pending_attempt = {
            "record": attempt,
            "same_player_control_frames": 0,
        }
        self._last_control = None

    def update(
        self,
        frame_number,
        timestamp_seconds,
        ball_position,
        ball_source,
        ball_is_valid,
        players,
        possession_state=None,
        possession_player_id=None,
    ):
        """Consume one frame using the existing confirmed-possession result."""
        frame_number = int(frame_number)
        timestamp_seconds = float(timestamp_seconds)
        if not math.isfinite(timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        self._last_frame = frame_number
        possession_state, possession_player_id = self._possession_input(
            players, possession_state, possession_player_id
        )
        owner = self._confirmed_owner(players, possession_state, possession_player_id)
        current_ball = self._valid_ball(ball_position, ball_source, ball_is_valid)
        valid_ball = current_ball is not None
        ball_x, ball_y = current_ball if valid_ball else (None, None)

        if owner is not None:
            self._last_confirmed_possession_time = timestamp_seconds

        if self._pending_attempt is not None:
            attempt = self._pending_attempt["record"]
            if (
                timestamp_seconds - attempt["start_timestamp_seconds"]
                >= self.MAX_OUTCOME_WAIT_SECONDS
            ):
                self._finish_pending(
                    frame_number, timestamp_seconds, ball_x, ball_y, valid_ball
                )
                return self.counts

            self._update_attempt_endpoint(
                frame_number, timestamp_seconds, ball_x, ball_y, valid_ball
            )
            pending = self._pending_attempt
            attempt = pending["record"]

            if owner is not None and str(owner["player_id"]) != str(
                attempt["passer_id"]
            ):
                outcome = (
                    "completed"
                    if owner["team"] == attempt["team"]
                    else "incomplete"
                )
                self._set_outcome(
                    attempt, outcome, "confirmed_possession_change", owner
                )
                self._pending_attempt = None
                self._last_control = (
                    {
                        **owner,
                        "ball": (ball_x, ball_y),
                        "frame": frame_number,
                        "timestamp": timestamp_seconds,
                    }
                    if valid_ball
                    else None
                )
            elif owner is not None and valid_ball:
                distance_to_passer = math.hypot(
                    ball_x - owner["position"][0], ball_y - owner["position"][1]
                )
                if distance_to_passer <= self.player_proximity_threshold:
                    pending["same_player_control_frames"] += 1
                    if (
                        pending["same_player_control_frames"]
                        >= self.SAME_PLAYER_CONTROL_FRAMES
                    ):
                        self._set_outcome(
                            attempt,
                            "unknown",
                            "original_possessor_regained_control",
                        )
                        self._pending_attempt = None
                        self._last_control = {
                            **owner,
                            "ball": (ball_x, ball_y),
                            "frame": frame_number,
                            "timestamp": timestamp_seconds,
                        }
                else:
                    pending["same_player_control_frames"] = 0

            return self.counts

        if owner is None:
            if (
                valid_ball
                and self._last_control is not None
                and self._last_confirmed_possession_time is not None
                and timestamp_seconds - self._last_confirmed_possession_time
                <= self.MAX_POSSESSION_GAP_SECONDS
            ):
                controlled_ball = self._last_control["ball"]
                travel = math.hypot(
                    ball_x - controlled_ball[0], ball_y - controlled_ball[1]
                )
                separation = math.hypot(
                    ball_x - self._last_control["position"][0],
                    ball_y - self._last_control["position"][1],
                )
                if (
                    separation >= self.release_separation_threshold
                    and travel >= self.minimum_release_travel
                ):
                    self._start_attempt(
                        self._last_control,
                        frame_number,
                        timestamp_seconds,
                        ball_x,
                        ball_y,
                        travel,
                    )
                    return self.counts
            if (
                self._last_confirmed_possession_time is not None
                and timestamp_seconds - self._last_confirmed_possession_time
                > self.MAX_POSSESSION_GAP_SECONDS
            ):
                self._last_control = None
            return self.counts

        if not valid_ball:
            return self.counts

        ball = (ball_x, ball_y)
        if self._last_control is None or (
            str(self._last_control["player_id"]) != str(owner["player_id"])
        ):
            self._last_control = {
                **owner,
                "ball": ball,
                "frame": frame_number,
                "timestamp": timestamp_seconds,
            }
            return self.counts

        controlled_ball = self._last_control["ball"]
        travel = math.hypot(
            ball_x - controlled_ball[0], ball_y - controlled_ball[1]
        )
        separation = math.hypot(
            ball_x - owner["position"][0], ball_y - owner["position"][1]
        )
        if (
            separation >= self.release_separation_threshold
            and travel >= self.minimum_release_travel
        ):
            self._start_attempt(
                owner, frame_number, timestamp_seconds, ball_x, ball_y, travel
            )
        elif separation <= self.player_proximity_threshold:
            self._last_control = {
                **owner,
                "ball": ball,
                "frame": frame_number,
                "timestamp": timestamp_seconds,
            }

        return self.counts

    def finish(self):
        """Finalize an unresolved in-flight attempt as unknown."""
        if self._pending_attempt is not None:
            attempt = self._pending_attempt["record"]
            self._finish_pending(
                self._last_frame
                if self._last_frame is not None
                else attempt["end_frame"],
                attempt["end_timestamp_seconds"],
                attempt["end_x_image"],
                attempt["end_y_image"],
                attempt["end_x_image"] is not None and attempt["end_y_image"] is not None,
            )
        return list(self.passes)

    def get_passes(self):
        return self.passes

    def get_team_counts(self):
        return self.team_counts

    def counts_by_team(self):
        return self.team_counts

    def total_count(self):
        return len(self.passes)

    @staticmethod
    def _value(value, key, default=None):
        return (
            value.get(key, default)
            if isinstance(value, dict)
            else getattr(value, key, default)
        )


__all__ = ["EvidenceBasedPassDetector"]
