"""Reconcile final team labels and rebuild outputs from finalized identities."""

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

import cv2

from utils.camera_radar import CameraRadarRenderer
from utils.dashboard import LAYOUT, compose_dashboard, output_size
from utils.pass_detection import EvidenceBasedPassDetector
from utils.possession_utils import PossessionEstimator
from utils.team_color_utils import TeamColorAssigner
from utils.visualization_utils import draw_annotations, is_displayable_player


TEAM_LABELS = {TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2}
MIN_TEAM_SHARE = 0.55


def reconcile_final_team_labels(
    observations_path,
    source_video_path,
    output_video_path,
    *,
    fps,
    ball_gate,
    layout=LAYOUT,
    min_team_share=MIN_TEAM_SHARE,
):
    """Apply each identity's strongest team evidence throughout the final outputs."""
    observations_path = Path(observations_path)
    source_video_path = Path(source_video_path)
    output_video_path = Path(output_video_path)
    if observations_path.resolve() == source_video_path.resolve():
        raise ValueError("The observations CSV cannot also be the source video.")
    if source_video_path.resolve() == output_video_path.resolve():
        raise ValueError("The source and output video paths must differ.")
    if not 0.5 <= min_team_share <= 1.0:
        raise ValueError("min_team_share must be between 0.5 and 1.0.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    team_weights = defaultdict(Counter)
    with observations_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("object_type") not in ("player", "goalkeeper"):
                continue
            global_id = row.get("global_id") or ""
            team = row.get("player_type") or ""
            if not global_id or team not in TEAM_LABELS:
                continue
            raw_confidence = row.get("confidence")
            if not raw_confidence:
                raise ValueError(
                    "A known player team label is missing detection "
                    f"confidence for global ID {global_id}."
                )
            confidence = float(raw_confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    "Detection confidence must be finite and between 0 and 1 "
                    f"for global ID {global_id}."
                )
            if confidence > 0:
                team_weights[global_id][team] += confidence

    final_teams = {}
    team_evidence = {}
    for global_id, weights in team_weights.items():
        total_weight = sum(weights.values())
        winning_team, winning_weight = weights.most_common(1)[0]
        share = winning_weight / total_weight
        final_team = (
            winning_team
            if share >= min_team_share
            else TeamColorAssigner.UNKNOWN
        )
        final_teams[global_id] = final_team
        team_evidence[global_id] = {
            "team_weights": dict(weights),
            "winning_share": round(share, 4),
            "assigned_team": final_team,
        }

    observations_temp = _temporary_path(observations_path, ".csv")
    video_temp = _temporary_path(output_video_path, ".mp4")
    capture = None
    writer = None
    processed_frames = 0
    possession = PossessionEstimator()
    team_colors = dict(TeamColorAssigner.TEAM_COLORS)
    radar_renderer = CameraRadarRenderer()
    try:
        capture = cv2.VideoCapture(str(source_video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open source video: {source_video_path}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"Invalid source video dimensions: {source_video_path}")
        pass_detector = EvidenceBasedPassDetector(
            fps, image_width=frame_width
        )

        writer = cv2.VideoWriter(
            str(video_temp),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            output_size(frame_width, frame_height, layout),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create reconciled video: {video_temp}")

        with (
            observations_path.open(
                newline="", encoding="utf-8-sig"
            ) as input_stream,
            observations_temp.open(
                "w", newline="", encoding="utf-8"
            ) as output_stream,
        ):
            reader = csv.DictReader(input_stream)
            if reader.fieldnames is None:
                raise ValueError("The observations CSV has no header.")
            csv_writer = csv.DictWriter(
                output_stream, fieldnames=reader.fieldnames
            )
            csv_writer.writeheader()

            expected_frame = 1
            for frame_key, grouped_rows in groupby(
                reader, key=lambda row: int(row["frame_number"])
            ):
                frame_number = int(frame_key)
                if frame_number != expected_frame:
                    raise ValueError(
                        "Observations must contain sequential frames; "
                        f"expected {expected_frame}, found {frame_number}."
                    )
                rows = list(grouped_rows)
                players, markers, ball_position = _finalize_frame_rows(
                    rows,
                    final_teams,
                    possession,
                    ball_gate,
                )
                ball_row = next(
                    row for row in rows if row.get("object_type") == "ball"
                )
                timestamp_seconds = _optional_float(
                    ball_row.get("timestamp_seconds")
                )
                pass_detector.update(
                    frame_number,
                    timestamp_seconds
                    if timestamp_seconds is not None
                    else (frame_number - 1) / fps,
                    ball_position,
                    ball_row.get("source") or "missing",
                    ball_position is not None,
                    rows,
                )

                success, source_frame = capture.read()
                if not success:
                    raise OSError(
                        "Source video ended before observations at frame "
                        f"{frame_number}."
                    )
                annotated = draw_annotations(
                    source_frame,
                    players,
                    copy_frame=False,
                    identity_resolved=True,
                )
                radar = radar_renderer.render(
                    markers,
                    source_frame.shape,
                    ball_position=ball_position,
                    team_colors=team_colors,
                )
                writer.write(
                    compose_dashboard(
                        annotated,
                        radar,
                        possession,
                        layout,
                        team_colors=team_colors,
                        pass_counts=pass_detector.counts_by_team(),
                    )
                )
                csv_writer.writerows(rows)
                processed_frames += 1
                expected_frame += 1

        writer.release()
        writer = None
        capture.release()
        capture = None
        if processed_frames == 0:
            raise ValueError("The observations CSV contains no frames.")

        os.replace(observations_temp, observations_path)
        os.replace(video_temp, output_video_path)
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        for path in (observations_temp, video_temp):
            if path.exists():
                path.unlink()

    return {
        "processed_frames": processed_frames,
        "minimum_team_share": min_team_share,
        "final_teams": final_teams,
        "team_evidence": team_evidence,
        "unresolved_global_ids": sorted(
            global_id
            for global_id, team in final_teams.items()
            if team == TeamColorAssigner.UNKNOWN
        ),
    }


def _finalize_frame_rows(rows, final_teams, possession, ball_gate):
    ball_rows = [row for row in rows if row.get("object_type") == "ball"]
    if len(ball_rows) != 1:
        raise ValueError(
            "Each frame must contain exactly one ball observation; "
            f"found {len(ball_rows)}."
        )

    players = []
    markers = {}
    candidates = []
    for row in rows:
        object_type = row.get("object_type") or ""
        if object_type not in ("player", "goalkeeper"):
            continue

        global_id = row.get("global_id") or None
        if object_type == "goalkeeper":
            player_type = (
                final_teams.get(
                    global_id, TeamColorAssigner.UNKNOWN_GOALKEEPER
                )
                if global_id is not None
                else TeamColorAssigner.UNKNOWN_GOALKEEPER
            )
        elif row.get("player_type") == TeamColorAssigner.TEAM_CONFLICT:
            player_type = TeamColorAssigner.TEAM_CONFLICT
        elif global_id is None:
            player_type = TeamColorAssigner.UNKNOWN
        else:
            player_type = final_teams.get(
                global_id,
                (
                    TeamColorAssigner.OTHER
                    if row.get("player_type") == TeamColorAssigner.OTHER
                    else TeamColorAssigner.UNKNOWN
                ),
            )
        color = (
            TeamColorAssigner.TEAM_COLORS[player_type]
            if player_type in TEAM_LABELS
            else (
                TeamColorAssigner.TEAM_COLORS[
                    TeamColorAssigner.GOALKEEPER
                ]
                if object_type == "goalkeeper"
                else TeamColorAssigner.NEUTRAL_COLOR
            )
        )
        row["player_type"] = player_type
        row["team_color"] = json.dumps(list(color))

        bbox = json.loads(row["bbox"])
        detection = {
            "object_type": object_type,
            "bbox": bbox,
            "global_id": global_id,
            "player_type": player_type,
            "team_color": color,
        }

        tracker_id = row.get("tracker_id")
        foot_x = _optional_float(row.get("foot_x"))
        foot_y = _optional_float(row.get("foot_y"))
        if is_displayable_player(detection):
            players.append(detection)
        if (
            is_displayable_player(detection)
            and tracker_id
            and foot_x is not None
            and foot_y is not None
        ):
            markers[int(tracker_id)] = (
                foot_x,
                foot_y,
                color,
                global_id,
            )
        if (
            object_type == "player"
            and global_id is not None
            and player_type in TEAM_LABELS
            and foot_x is not None
            and foot_y is not None
        ):
            candidates.append(
                {
                    "object_type": object_type,
                    "global_id": global_id,
                    "tracker_id": tracker_id,
                    "player_type": player_type,
                    "foot_x": foot_x,
                    "foot_y": foot_y,
                }
            )

    ball_row = ball_rows[0]
    ball_position = None
    if _csv_bool(ball_row.get("is_valid")):
        image_x = _optional_float(ball_row.get("image_x"))
        image_y = _optional_float(ball_row.get("image_y"))
        if image_x is not None and image_y is not None:
            ball_position = (image_x, image_y)
    possession_position = ball_position
    if ball_gate and ball_row.get("source") != "real":
        possession_position = None
    result = possession.update(possession_position, candidates)
    for row in rows:
        row["possession_state"] = result["state"]
        row["possession_player_id"] = (
            ""
            if result["player_id"] is None
            else str(result["player_id"])
        )

    return players, markers, ball_position


def _optional_float(value):
    return None if value in (None, "") else float(value)


def _csv_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Invalid boolean value in observations CSV: {value!r}")


def _temporary_path(target, suffix):
    descriptor, name = tempfile.mkstemp(
        prefix=f".{Path(target).stem}.reconciled_",
        suffix=suffix,
        dir=Path(target).parent,
    )
    os.close(descriptor)
    return Path(name)
