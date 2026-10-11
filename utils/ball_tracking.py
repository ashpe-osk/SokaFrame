"""Ball selection with bounded, two-sided short-gap interpolation."""


class BallPositionTracker:
    MAX_GAP_FRAMES = 5

    def __init__(self, max_gap_frames=MAX_GAP_FRAMES):
        self.max_gap_frames = int(max_gap_frames)
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative")
        self.last_position = None
        self.last_real_frame = None
        self.last_frame = 0
        self.pending_frames = []
        self.missing_frames = 0
        self.real_frames = 0
        self.interpolated_frames = 0
        self.filled_frames = 0
        self.unknown_frames = 0

    def update(self, ball_detections, frame_number):
        frame_number = int(frame_number)
        if frame_number <= self.last_frame:
            raise ValueError("frame_number must be strictly increasing")
        self.last_frame = frame_number
        detection = (
            max(ball_detections, key=lambda item: item["confidence"])
            if ball_detections
            else None
        )
        if detection is None:
            self.pending_frames.append(frame_number)
            self.missing_frames = len(self.pending_frames)
            if len(self.pending_frames) > self.max_gap_frames:
                missing_frame = self.pending_frames.pop(0)
                self.unknown_frames += 1
                return [self._record(missing_frame, None, "missing")]
            return []

        x1, y1, x2, y2 = detection["bbox"]
        position = (
            (float(x1) + float(x2)) / 2.0,
            (float(y1) + float(y2)) / 2.0,
        )
        confidence = float(detection["confidence"])
        resolved = []
        gap = (
            frame_number - self.last_real_frame - 1
            if self.last_real_frame is not None
            else None
        )
        if self.pending_frames:
            can_interpolate = (
                self.last_position is not None
                and gap is not None
                and 0 < gap <= self.max_gap_frames
                and len(self.pending_frames) == gap
            )
            if can_interpolate:
                frame_span = frame_number - self.last_real_frame
                start_x, start_y = self.last_position
                end_x, end_y = position
                for pending_frame in self.pending_frames:
                    fraction = (
                        pending_frame - self.last_real_frame
                    ) / frame_span
                    interpolated = (
                        start_x + fraction * (end_x - start_x),
                        start_y + fraction * (end_y - start_y),
                    )
                    resolved.append(
                        self._record(
                            pending_frame,
                            interpolated,
                            "interpolated",
                        )
                    )
                    self.interpolated_frames += 1
                    self.filled_frames += 1
            else:
                resolved.extend(
                    self._record_missing(pending_frame)
                    for pending_frame in self.pending_frames
                )
            self.pending_frames.clear()
        self.last_position = position
        self.last_real_frame = frame_number
        self.missing_frames = 0
        self.real_frames += 1
        resolved.append(
            self._record(
                frame_number,
                position,
                "real",
                confidence=confidence,
            )
        )
        return resolved

    def finish(self):
        resolved = [
            self._record_missing(frame_number)
            for frame_number in self.pending_frames
        ]
        self.pending_frames.clear()
        self.missing_frames = 0
        return resolved

    def _record_missing(self, frame_number):
        self.unknown_frames += 1
        return self._record(frame_number, None, "missing")

    @staticmethod
    def _record(frame_number, position, source, confidence=None):
        return {
            "frame_number": frame_number,
            "position": position,
            "source": source,
            "confidence": confidence,
        }
