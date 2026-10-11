"""Normalized per-frame player and ball observations for analytics consumers."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PlayerObservation:
    frame_number: int
    timestamp_seconds: float
    tracker_id: int
    object_type: str
    player_type: str
    image_x: float
    image_y: float
    foot_x: float
    foot_y: float
    pitch_x: Optional[float]
    pitch_y: Optional[float]
    confidence: float
    is_visible: bool
    bbox: Tuple[float, float, float, float]
    team_color: Tuple[int, int, int]
    possession_state: Optional[str] = None
    possession_player_id: Optional[int] = None
    global_id: Optional[int] = None

    @property
    def foot_xy(self):
        return (self.foot_x, self.foot_y)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BallObservation:
    frame_number: int
    timestamp_seconds: float
    image_x: Optional[float]
    image_y: Optional[float]
    pitch_x: Optional[float]
    pitch_y: Optional[float]
    confidence: Optional[float]
    source: str
    is_valid: bool

    @property
    def position(self):
        if not self.is_valid or self.image_x is None or self.image_y is None:
            return None
        return (self.image_x, self.image_y)

    def to_dict(self):
        return asdict(self)
