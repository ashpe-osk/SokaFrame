"""Phase 1 tracking cache and diagnostic artifact writers."""

import csv
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np

from utils.team_color_utils import TeamColorAssigner
from utils.visualization_utils import draw_annotations


STUB_VERSION = 1


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _iou(first, second):
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TrackingStubWriter:
    """Write one metadata header followed by one pickled record per frame."""

    def __init__(self, path, *, video_path, class_names, fps, frame_size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        pickle.dump(
            {
                "stub_version": STUB_VERSION,
                "video_path": str(video_path),
                "class_names": class_names,
                "fps": float(fps),
                "frame_size": tuple(frame_size),
            },
            self._file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        self.frames_written = 0

    def write_frame(
        self,
        frame_number,
        detections,
        camera_motion,
        filter_counts,
        valid_detection_counts,
        total_detection_count,
        lost_track_ids,
    ):
        record = {
            "frame_number": int(frame_number),
            "detections": [
                {
                    **detection,
                    "bbox": tuple(map(float, detection["bbox"])),
                    "tracker_id": (
                        int(detection["tracker_id"])
                        if detection.get("tracker_id") is not None
                        else None
                    ),
                    "reid_feature": (
                        np.asarray(
                            detection["reid_feature"], dtype=np.float32
                        ).reshape(-1)
                        if detection.get("reid_feature") is not None
                        else None
                    ),
                }
                for detection in detections
            ],
            "camera_motion": camera_motion.to_dict(),
            "pretracker_filter_counts": dict(filter_counts),
            "pretracker_valid_detection_counts": dict(valid_detection_counts),
            "pretracker_total_detection_count": int(total_detection_count),
            "lost_track_ids": sorted(int(track_id) for track_id in lost_track_ids),
        }
        pickle.dump(record, self._file, protocol=pickle.HIGHEST_PROTOCOL)
        self.frames_written += 1

    def close(self):
        if not self._file.closed:
            pickle.dump(
                {"end_of_stub": True, "frames_written": self.frames_written},
                self._file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            self._file.flush()
            self._file.close()


class TrackingStubReader:
    """Read and validate the streaming tracking-stub format."""

    def __init__(self, path, *, video_path, frame_size):
        self.path = Path(path)
        self._file = self.path.open("rb")
        self.metadata = pickle.load(self._file)
        if self.metadata.get("stub_version") != STUB_VERSION:
            self.close()
            raise ValueError(f"Unsupported tracking stub version in {self.path}")
        if tuple(self.metadata.get("frame_size", ())) != tuple(frame_size):
            self.close()
            raise ValueError(
                "Tracking stub video dimensions do not match the input "
                f"({self.metadata.get('frame_size')} vs {frame_size})."
            )
        cached_video = Path(self.metadata.get("video_path", "")).resolve()
        if cached_video != Path(video_path).resolve():
            self.close()
            raise ValueError(
                "Tracking stub was generated from a different input video: "
                f"{cached_video}"
            )
        self.finished = False

    def read_frame(self, expected_frame_number):
        if self.finished:
            return None
        try:
            record = pickle.load(self._file)
        except EOFError:
            raise ValueError(f"Tracking stub ended before frame {expected_frame_number}.")
        if record.get("end_of_stub"):
            self.finished = True
            self.frames_read = expected_frame_number - 1
            if record.get("frames_written") != self.frames_read:
                raise ValueError(
                    "Tracking stub frame count mismatch: read "
                    f"{self.frames_read}, sentinel says "
                    f"{record.get('frames_written')}."
                )
            return None
        if record.get("frame_number") != expected_frame_number:
            raise ValueError(
                "Tracking stub frame order mismatch: expected "
                f"{expected_frame_number}, got {record.get('frame_number')}."
            )
        return record

    def close(self):
        self._file.close()


def camera_motion_from_dict(record):
    from utils.camera_motion import CameraMotion

    translation = tuple(record.get("translation_px", (0.0, 0.0)))
    transform = record.get("transform_previous_to_current")
    return CameraMotion(
        np.asarray(transform, dtype=np.float64) if transform is not None else None,
        (float(translation[0]), float(translation[1])),
        float(record.get("rotation_degrees", 0.0)),
        float(record.get("scale", 1.0)),
        float(record.get("confidence", 0.0)),
        int(record.get("inliers", 0)),
        int(record.get("tracked_features", 0)),
        bool(record.get("scene_cut", False)),
        bool(record.get("available", False)),
    )


class Phase1Analyzer:
    """Collect per-detection diagnostics, event windows, and case evidence."""

    CSV_FIELDS = (
        "frame",
        "tracker_id",
        "global_id",
        "object_type",
        "class_id",
        "team_label",
        "provisional",
        "bbox",
        "foot_x",
        "foot_y",
        "confidence",
        "team_color_distances",
        "nearest_neighbor_iou",
        "nearest_neighbor_tracker_id",
        "camera_epoch",
    )

    def __init__(self, output_dir, fps):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.csv_file = (self.output_dir / "debug_detections.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self.csv_writer = csv.DictWriter(
            self.csv_file, fieldnames=self.CSV_FIELDS
        )
        self.csv_writer.writeheader()
        self.events_file = (self.output_dir / "events.jsonl").open(
            "w", encoding="utf-8"
        )
        self.events = []
        self._events_by_frame = set()
        self.previous = {}
        self.active_tracker_ids = set()
        self.recent_tiles = {}
        self.contact_tiles = {}
        self.pending_event_frames = set()
        self.global_intervals = {}
        self.global_observations = {}
        self.global_latest_tracker = {}
        self.frames_analyzed = 0

    def _log_event(self, event):
        event = _json_value(event)
        self.events.append(event)
        self._events_by_frame.add(int(event["frame"]))
        self.events_file.write(json.dumps(event, separators=(",", ":")) + "\n")
        frame_number = int(event["frame"])
        for selected_frame in range(max(1, frame_number - 3), frame_number + 4):
            if selected_frame in self.recent_tiles:
                self.contact_tiles[selected_frame] = self.recent_tiles[
                    selected_frame
                ]
            elif selected_frame > frame_number:
                self.pending_event_frames.add(selected_frame)

    def observe(
        self,
        frame,
        frame_number,
        detections,
        player_types,
        global_id_by_tracker,
        team_assigner,
        camera_epoch,
    ):
        self.frames_analyzed = frame_number
        tracked = [
            detection
            for detection in detections
            if detection.get("tracker_id") is not None
        ]
        current_ids = {int(detection["tracker_id"]) for detection in tracked}
        for tracker_id in sorted(current_ids - self.active_tracker_ids):
            self._log_event(
                {"frame": frame_number, "event": "tracker_appears", "tracker_id": tracker_id}
            )
        for tracker_id in sorted(self.active_tracker_ids - current_ids):
            self._log_event(
                {"frame": frame_number, "event": "tracker_disappears", "tracker_id": tracker_id}
            )

        observations = []
        for detection in tracked:
            tracker_id = int(detection["tracker_id"])
            object_type = str(detection.get("object_type", "other"))
            team_label = player_types.get(tracker_id, "")
            global_id = global_id_by_tracker.get(tracker_id)
            distances = (
                team_assigner.last_goalkeeper_distances.get(tracker_id)
                if object_type == "goalkeeper"
                else team_assigner.last_color_distances.get(tracker_id)
            )
            provisional = (
                team_label
                in (
                    "",
                    TeamColorAssigner.UNKNOWN,
                    TeamColorAssigner.UNKNOWN_GOALKEEPER,
                )
                or tracker_id in team_assigner._pending_assignments
                or tracker_id in team_assigner._other_label_history
                or tracker_id in team_assigner._goalkeeper_pending
            )
            bbox = tuple(map(float, detection["bbox"]))
            foot_x = (bbox[0] + bbox[2]) / 2.0
            foot_y = bbox[3]
            candidates = [
                (
                    _iou(bbox, other["bbox"]),
                    int(other["tracker_id"]),
                )
                for other in tracked
                if int(other["tracker_id"]) != tracker_id
                and other.get("object_type") in ("player", "goalkeeper")
                and object_type in ("player", "goalkeeper")
            ]
            nearest_iou, nearest_tracker_id = (
                max(candidates) if candidates else (0.0, None)
            )
            row = {
                "frame": frame_number,
                "tracker_id": tracker_id,
                "global_id": global_id if global_id is not None else "",
                "object_type": object_type,
                "class_id": int(detection.get("class_id", -1)),
                "team_label": team_label,
                "provisional": bool(provisional),
                "bbox": json.dumps(bbox),
                "foot_x": round(foot_x, 3),
                "foot_y": round(foot_y, 3),
                "confidence": float(detection.get("confidence", 0.0)),
                "team_color_distances": json.dumps(_json_value(distances or {})),
                "nearest_neighbor_iou": round(float(nearest_iou), 4),
                "nearest_neighbor_tracker_id": (
                    nearest_tracker_id if nearest_tracker_id is not None else ""
                ),
                "camera_epoch": int(camera_epoch),
            }
            self.csv_writer.writerow(row)
            observations.append(
                {
                    **detection,
                    "tracker_id": tracker_id,
                    "global_id": global_id,
                    "player_type": team_label,
                    "team_label": team_label,
                    "provisional": provisional,
                    "team_color": team_assigner.team_colors.get(
                        team_label, TeamColorAssigner.NEUTRAL_COLOR
                    ),
                    "camera_epoch": camera_epoch,
                    "nearest_neighbor_iou": float(nearest_iou),
                    "nearest_neighbor_tracker_id": nearest_tracker_id,
                }
            )

            old = self.previous.get(tracker_id)
            current = {
                "object_type": object_type,
                "team_label": team_label,
                "global_id": global_id,
            }
            if old is not None:
                for field, event_name in (
                    ("team_label", "team_label_changed"),
                    ("object_type", "object_type_changed"),
                    ("global_id", "global_id_changed"),
                ):
                    if old[field] != current[field]:
                        self._log_event(
                            {
                                "frame": frame_number,
                                "event": event_name,
                                "tracker_id": tracker_id,
                                "from": old[field],
                                "to": current[field],
                            }
                        )
            self.previous[tracker_id] = current
            if global_id is not None and object_type in ("player", "goalkeeper"):
                previous_tracker_id = self.global_latest_tracker.get(global_id)
                if (
                    previous_tracker_id is not None
                    and previous_tracker_id != tracker_id
                ):
                    self._log_event(
                        {
                            "frame": frame_number,
                            "event": "global_id_relinked",
                            "global_id": global_id,
                            "from_tracker_id": previous_tracker_id,
                            "to_tracker_id": tracker_id,
                        }
                    )
                self.global_latest_tracker[global_id] = tracker_id
                self.global_observations.setdefault(global_id, []).append(
                    {
                        "frame": frame_number,
                        "tracker_id": tracker_id,
                        "team_label": team_label,
                        "object_type": object_type,
                        "camera_epoch": int(camera_epoch),
                    }
                )
                intervals = self.global_intervals.setdefault(global_id, [])
                starts_new_interval = (
                    not intervals
                    or intervals[-1]["end"] + 1 != frame_number
                    or intervals[-1]["tracker_id"] != tracker_id
                    or intervals[-1]["team_label"] != team_label
                    or intervals[-1]["object_type"] != object_type
                    or intervals[-1]["camera_epoch"] != int(camera_epoch)
                )
                if starts_new_interval:
                    intervals.append(
                        {
                            "start": frame_number,
                            "end": frame_number,
                            "tracker_id": tracker_id,
                            "team_label": team_label,
                            "object_type": object_type,
                            "camera_epoch": int(camera_epoch),
                        }
                    )
                else:
                    intervals[-1]["end"] = frame_number

        player_boxes = [
            item
            for item in observations
            if item["object_type"] in ("player", "goalkeeper")
        ]
        for index, first in enumerate(player_boxes):
            for second in player_boxes[index + 1 :]:
                overlap = _iou(first["bbox"], second["bbox"])
                if overlap > 0.1:
                    self._log_event(
                        {
                            "frame": frame_number,
                            "event": "player_boxes_overlap",
                            "tracker_ids": sorted(
                                (first["tracker_id"], second["tracker_id"])
                            ),
                            "global_ids": [
                                first.get("global_id"),
                                second.get("global_id"),
                            ],
                            "team_labels": [
                                first.get("team_label"),
                                second.get("team_label"),
                            ],
                            "iou": round(float(overlap), 4),
                        }
                    )

        self.active_tracker_ids = current_ids
        tile = self._make_contact_tile(frame, frame_number, observations)
        self.recent_tiles[frame_number] = tile
        for old_frame in sorted(self.recent_tiles):
            if old_frame < frame_number - 3:
                del self.recent_tiles[old_frame]
        if (
            frame_number % 5 == 0
            or frame_number in self.pending_event_frames
            or frame_number in self._events_by_frame
        ):
            self.contact_tiles[frame_number] = tile
            self.pending_event_frames.discard(frame_number)

    @staticmethod
    def _make_contact_tile(frame, frame_number, observations):
        annotated = draw_annotations(frame, observations)
        boxes = [
            detection["bbox"]
            for detection in observations
            if detection["object_type"] in ("player", "goalkeeper")
        ]
        if boxes:
            pad = 36
            x1 = max(0, int(min(box[0] for box in boxes)) - pad)
            y1 = max(0, int(min(box[1] for box in boxes)) - pad)
            x2 = min(frame.shape[1], int(max(box[2] for box in boxes)) + pad)
            y2 = min(frame.shape[0], int(max(box[3] for box in boxes)) + pad)
            crop = annotated[y1:y2, x1:x2]
        else:
            crop = annotated
        if crop.size == 0:
            crop = annotated
        tile = cv2.resize(crop, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (118, 25), (0, 0, 0), cv2.FILLED)
        cv2.putText(
            tile,
            f"Frame {frame_number}",
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        success, encoded = cv2.imencode(
            ".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not success:
            raise OSError(f"Could not encode contact-sheet frame {frame_number}")
        return encoded.tobytes()

    def finish(self, last_frame):
        for tracker_id in sorted(self.active_tracker_ids):
            self._log_event(
                {
                    "frame": last_frame + 1,
                    "event": "tracker_disappears_at_end",
                    "tracker_id": tracker_id,
                }
            )
        for frame_number in sorted(self.pending_event_frames):
            tile = self.recent_tiles.get(frame_number)
            if tile is not None:
                self.contact_tiles[frame_number] = tile

        self.csv_file.flush()
        self.events_file.flush()
        self.csv_file.close()
        self.events_file.close()
        self._write_contact_sheet()
        self._write_case_summary()

    def _write_contact_sheet(self):
        if not self.contact_tiles:
            return
        columns = 5
        tile_width, tile_height = 320, 180
        selected = sorted(self.contact_tiles.items())
        rows = math.ceil(len(selected) / columns)
        sheet = np.zeros(
            (rows * tile_height, columns * tile_width, 3), dtype=np.uint8
        )
        for index, (_, encoded) in enumerate(selected):
            tile = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if tile is None:
                raise OSError("Could not decode a contact-sheet tile.")
            row, column = divmod(index, columns)
            y = row * tile_height
            x = column * tile_width
            sheet[y : y + tile_height, x : x + tile_width] = tile
        path = self.output_dir / "contact_sheet.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"Could not write contact sheet: {path}")

    def _write_case_summary(self):
        cases = {}
        for global_id in (5, 8, 12, 32, 34, 37, 39):
            observations = self.global_observations.get(global_id, [])
            cases[str(global_id)] = {
                "observed": bool(observations),
                "intervals": self.global_intervals.get(global_id, []),
                "distinct_tracker_ids": sorted(
                    {item["tracker_id"] for item in observations}
                ),
                "distinct_team_labels": sorted(
                    {item["team_label"] for item in observations}
                ),
                "distinct_object_types": sorted(
                    {item["object_type"] for item in observations}
                ),
            }
        pairs = ((5, 8), (32, 34))
        overlap_evidence = [
            event
            for event in self.events
            if event["event"] == "player_boxes_overlap"
            and event.get("global_ids")
            and any(
                set(event["global_ids"]) == set(pair) for pair in pairs
            )
        ]
        identity_12 = self.global_observations.get(12, [])
        identity_39 = self.global_observations.get(39, [])
        camera_return = {
            "global_id_12_last_frame": (
                max((item["frame"] for item in identity_12), default=None)
            ),
            "global_id_39_first_frame": (
                min((item["frame"] for item in identity_39), default=None)
            ),
            "global_id_12_camera_epochs": sorted(
                {item["camera_epoch"] for item in identity_12}
            ),
            "global_id_39_camera_epochs": sorted(
                {item["camera_epoch"] for item in identity_39}
            ),
        }
        camera_return["frame_gap"] = (
            camera_return["global_id_39_first_frame"]
            - camera_return["global_id_12_last_frame"]
            if camera_return["global_id_12_last_frame"] is not None
            and camera_return["global_id_39_first_frame"] is not None
            else None
        )
        path = self.output_dir / "case_evidence.json"
        path.write_text(
            json.dumps(
                {
                    "warning": (
                        "Framewise labels and tracker IDs are evidence, not "
                        "physical-identity ground truth."
                    ),
                    "global_id_cases": cases,
                    "overlap_evidence_for_requested_pairs": overlap_evidence,
                    "camera_return_12_to_39": camera_return,
                    "events": self.events,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
