"""Export summary and statistics derived from finalized frame observations."""

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from utils.pass_detection import EvidenceBasedPassDetector

PLAYER_OBJECT_TYPES = {"player", "goalkeeper"}
TEAM_LABELS = {"team_1", "team_2"}


def write_match_analytics(
    observations_path,
    output_dir,
    fps,
    tracking_events_path=None,
    image_width=1280,
):
    observations_path = Path(observations_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    team_frames = Counter()
    team_players = defaultdict(set)
    players = {}
    frame_states = {}
    frame_possessors = {}
    frame_timestamps = {}
    frame_rows = defaultdict(list)
    object_counts = Counter()

    with observations_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            frame_number = int(row["frame_number"])
            object_type = row.get("object_type") or ""
            object_counts[object_type] += 1
            frame_rows[frame_number].append(row)
            timestamp = row.get("timestamp_seconds")
            if timestamp:
                frame_timestamps[frame_number] = float(timestamp)

            possession_state = row.get("possession_state") or "Unknown"
            previous_state = frame_states.setdefault(
                frame_number, possession_state
            )
            if previous_state != possession_state:
                raise ValueError(
                    "Conflicting possession states in observations for "
                    f"frame {frame_number}: {previous_state!r} and "
                    f"{possession_state!r}."
                )
            possessor = row.get("possession_player_id") or ""
            previous_possessor = frame_possessors.setdefault(
                frame_number, possessor
            )
            if previous_possessor != possessor:
                raise ValueError(
                    "Conflicting possession player IDs in observations for "
                    f"frame {frame_number}."
                )

            if object_type not in PLAYER_OBJECT_TYPES:
                continue
            team = row.get("player_type") or "unknown"
            team_frames[team] += 1
            global_id = row.get("global_id") or ""
            if not global_id:
                continue

            team_players[team].add(global_id)
            player = players.setdefault(
                global_id,
                {
                    "global_id": global_id,
                    "teams": set(),
                    "object_types": set(),
                    "tracker_ids": set(),
                    "visible_frames": 0,
                    "first_seen_frame": frame_number,
                    "last_seen_frame": frame_number,
                    "confidence_total": 0.0,
                },
            )
            player["teams"].add(team)
            player["object_types"].add(object_type)
            if row.get("tracker_id"):
                player["tracker_ids"].add(row["tracker_id"])
            player["visible_frames"] += 1
            player["first_seen_frame"] = min(
                player["first_seen_frame"], frame_number
            )
            player["last_seen_frame"] = max(
                player["last_seen_frame"], frame_number
            )
            player["confidence_total"] += float(row.get("confidence") or 0.0)

    pass_detector = EvidenceBasedPassDetector(fps, image_width=image_width)
    valid_team_possession_frames = Counter()
    for frame_number in sorted(frame_rows):
        rows = frame_rows[frame_number]
        state = frame_states[frame_number]
        possessor_id = frame_possessors.get(frame_number, "")
        ball_rows = [row for row in rows if row.get("object_type") == "ball"]
        if len(ball_rows) > 1:
            raise ValueError(
                f"Expected at most one ball observation in frame {frame_number}."
            )
        ball_row = ball_rows[0] if ball_rows else {}
        is_valid_ball = str(ball_row.get("is_valid", "")).strip().lower() in (
            "1",
            "true",
            "yes",
        )
        valid_carrier = any(
            row.get("object_type") == "player"
            and row.get("global_id") == possessor_id
            and row.get("player_type") == state
            and row.get("is_visible", "true").strip().lower()
            not in ("0", "false", "no")
            for row in rows
        )
        if (
            state in TEAM_LABELS
            and possessor_id
            and valid_carrier
            and is_valid_ball
        ):
            valid_team_possession_frames[state] += 1

        x = _optional_float(ball_row.get("image_x"))
        y = _optional_float(ball_row.get("image_y"))
        ball_position = (x, y) if x is not None and y is not None else None
        pass_detector.update(
            frame_number,
            frame_timestamps.get(frame_number, (frame_number - 1) / fps),
            ball_position,
            ball_row.get("source") or "missing",
            is_valid_ball,
            rows,
        )
    passes = pass_detector.finish()

    pass_counts_by_player = defaultdict(Counter)
    pass_counts_by_team = defaultdict(Counter)
    passing_edges = Counter()
    for attempted_pass in passes:
        team = attempted_pass["team"]
        outcome = attempted_pass["outcome"]
        passer_id = str(attempted_pass["passer_id"])
        pass_counts_by_player[passer_id]["attempted"] += 1
        pass_counts_by_player[passer_id][outcome] += 1
        pass_counts_by_team[team]["attempted"] += 1
        pass_counts_by_team[team][outcome] += 1
        receiver_id = attempted_pass["receiver_id"]
        if outcome == "completed" and receiver_id:
            receiver_id = str(receiver_id)
            pass_counts_by_player[receiver_id]["received"] += 1
            passing_edges[(team, passer_id, receiver_id)] += 1

    state_counts = Counter(frame_states.values())
    possession_sequences = []
    current = None
    previous_frame = None
    for frame_number in sorted(frame_states):
        state = frame_states[frame_number]
        possessor = frame_possessors.get(frame_number, "")
        key = (state, possessor)
        if (
            current is None
            or key != current["key"]
            or previous_frame is None
            or frame_number != previous_frame + 1
        ):
            current = {
                "key": key,
                "frame_start": frame_number,
                "frame_end": frame_number,
            }
            possession_sequences.append(current)
        else:
            current["frame_end"] = frame_number
        previous_frame = frame_number

    sequence_fields = (
        "frame_start",
        "frame_end",
        "start_seconds",
        "end_seconds",
        "duration_seconds",
        "frame_count",
        "possession_state",
        "possession_player_id",
    )
    with (output_dir / "possession_sequences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=sequence_fields)
        writer.writeheader()
        for sequence in possession_sequences:
            first = sequence["frame_start"]
            last = sequence["frame_end"]
            start_seconds = frame_timestamps.get(
                first, (first - 1) / fps
            )
            end_seconds = frame_timestamps.get(last, (last - 1) / fps)
            writer.writerow(
                {
                    "frame_start": first,
                    "frame_end": last,
                    "start_seconds": round(start_seconds, 3),
                    "end_seconds": round(end_seconds, 3),
                    "duration_seconds": round(
                        (last - first + 1) / fps, 3
                    ),
                    "frame_count": last - first + 1,
                    "possession_state": sequence["key"][0],
                    "possession_player_id": sequence["key"][1],
                }
            )

    valid_team_possession_total = sum(valid_team_possession_frames.values())
    team_fields = (
        "team",
        "visible_player_frames",
        "unique_global_ids",
        "valid_outfield_possession_frames",
        "possession_percentage",
        "passes_attempted",
        "passes_completed",
        "passes_incomplete",
        "passes_unknown",
        "pass_completion_percentage",
    )
    with (output_dir / "team_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=team_fields)
        writer.writeheader()
        for team in sorted(TEAM_LABELS):
            decided_passes = (
                pass_counts_by_team[team]["completed"]
                + pass_counts_by_team[team]["incomplete"]
            )
            writer.writerow(
                {
                    "team": team,
                    "visible_player_frames": team_frames[team],
                    "unique_global_ids": len(team_players[team]),
                    "valid_outfield_possession_frames": (
                        valid_team_possession_frames[team]
                    ),
                    "possession_percentage": round(
                        100.0
                        * valid_team_possession_frames[team]
                        / valid_team_possession_total,
                        2,
                    )
                    if valid_team_possession_total
                    else 0.0,
                    "passes_attempted": pass_counts_by_team[team]["attempted"],
                    "passes_completed": pass_counts_by_team[team]["completed"],
                    "passes_incomplete": pass_counts_by_team[team]["incomplete"],
                    "passes_unknown": pass_counts_by_team[team]["unknown"],
                    "pass_completion_percentage": round(
                        100.0
                        * pass_counts_by_team[team]["completed"]
                        / decided_passes,
                        2,
                    )
                    if decided_passes
                    else 0.0,
                }
            )

    player_fields = (
        "global_id",
        "team",
        "visible_frames",
        "first_seen_frame",
        "last_seen_frame",
        "tracker_ids",
        "object_types",
        "mean_confidence",
        "passes_attempted",
        "passes_completed",
        "passes_incomplete",
        "passes_unknown",
        "passes_received",
        "pass_completion_percentage",
    )
    with (output_dir / "player_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=player_fields)
        writer.writeheader()
        for global_id, player in sorted(
            players.items(), key=lambda item: int(item[0])
        ):
            known_teams = player["teams"] & TEAM_LABELS
            team = (
                next(iter(known_teams))
                if len(known_teams) == 1
                else "inconsistent"
                if len(known_teams) > 1
                else "unknown"
            )
            pass_counts = pass_counts_by_player[global_id]
            decided_passes = (
                pass_counts["completed"] + pass_counts["incomplete"]
            )
            writer.writerow(
                {
                    "global_id": global_id,
                    "team": team,
                    "visible_frames": player["visible_frames"],
                    "first_seen_frame": player["first_seen_frame"],
                    "last_seen_frame": player["last_seen_frame"],
                    "tracker_ids": json.dumps(
                        sorted(player["tracker_ids"], key=int)
                    ),
                    "object_types": json.dumps(
                        sorted(player["object_types"])
                    ),
                    "mean_confidence": round(
                        player["confidence_total"]
                        / player["visible_frames"],
                        4,
                    ),
                    "passes_attempted": pass_counts["attempted"],
                    "passes_completed": pass_counts["completed"],
                    "passes_incomplete": pass_counts["incomplete"],
                    "passes_unknown": pass_counts["unknown"],
                    "passes_received": pass_counts["received"],
                    "pass_completion_percentage": round(
                        100.0 * pass_counts["completed"] / decided_passes,
                        2,
                    )
                    if decided_passes
                    else 0.0,
                }
            )

    pass_fields = (
        "pass_id",
        "start_frame",
        "end_frame",
        "start_timestamp_seconds",
        "end_timestamp_seconds",
        "passer_id",
        "receiver_id",
        "team",
        "outcome",
        "start_x_image",
        "start_y_image",
        "end_x_image",
        "end_y_image",
        "start_x_pitch",
        "start_y_pitch",
        "end_x_pitch",
        "end_y_pitch",
        "coordinate_space",
        "pitch_coordinates_status",
        "confidence",
        "outcome_reason",
    )
    with (output_dir / "passes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pass_fields)
        writer.writeheader()
        writer.writerows(passes)

    network_fields = (
        "team",
        "passer_id",
        "receiver_id",
        "completed_passes",
    )
    with (output_dir / "passing_network.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=network_fields)
        writer.writeheader()
        for (team, passer_id, receiver_id), count in sorted(passing_edges.items()):
            writer.writerow(
                {
                    "team": team,
                    "passer_id": passer_id,
                    "receiver_id": receiver_id,
                    "completed_passes": count,
                }
            )

    events_file = None
    event_count = None
    if tracking_events_path is not None and Path(tracking_events_path).is_file():
        events_file = output_dir / "tracking_events.jsonl"
        shutil.copyfile(tracking_events_path, events_file)
        with events_file.open(encoding="utf-8") as stream:
            event_count = sum(1 for line in stream if line.strip())

    frame_count = len(frame_states)
    ordered_frames = sorted(frame_states)
    duration_seconds = (
        (ordered_frames[-1] - ordered_frames[0] + 1) / fps
        if ordered_frames and fps > 0
        else 0.0
    )
    summary = {
        "schema_version": "1.0",
        "processed_frames": frame_count,
        "first_frame": ordered_frames[0] if ordered_frames else None,
        "last_frame": ordered_frames[-1] if ordered_frames else None,
        "fps": fps,
        "image_width": image_width,
        "duration_seconds": round(duration_seconds, 3),
        "object_observation_rows": dict(sorted(object_counts.items())),
        "unique_global_players": len(players),
        "global_ids_with_conflicting_known_teams": sorted(
            (
                global_id
                for global_id, player in players.items()
                if len(player["teams"] & TEAM_LABELS) > 1
            ),
            key=int,
        ),
        "possession_frames": dict(sorted(state_counts.items())),
        "valid_outfield_possession_frames": {
            team: valid_team_possession_frames[team]
            for team in sorted(TEAM_LABELS)
        },
        "team_possession_percentages": {
            team: round(
                100.0 * valid_team_possession_frames[team]
                / valid_team_possession_total,
                2,
            )
            if valid_team_possession_total
            else 0.0
            for team in sorted(TEAM_LABELS)
        },
        "pass_statistics": _pass_statistics(passes, pass_counts_by_team),
        "passing_network": [
            {
                "team": team,
                "passer_id": passer_id,
                "receiver_id": receiver_id,
                "completed_passes": count,
            }
            for (team, passer_id, receiver_id), count in sorted(
                passing_edges.items()
            )
        ],
        "pass_visualizations": [
            {
                "pass_id": item["pass_id"],
                "start_timestamp_seconds": item[
                    "start_timestamp_seconds"
                ],
                "end_timestamp_seconds": item["end_timestamp_seconds"],
                "start_x_image": item["start_x_image"],
                "start_y_image": item["start_y_image"],
                "end_x_image": item["end_x_image"],
                "end_y_image": item["end_y_image"],
                "team": item["team"],
                "outcome": item["outcome"],
            }
            for item in passes
        ],
        "team_summaries": _team_summaries(
            valid_team_possession_frames,
            valid_team_possession_total,
            pass_counts_by_team,
        ),
        "player_summaries": _player_summaries(
            players, pass_counts_by_player
        ),
        "outputs": {
            "team_statistics": "team_statistics.csv",
            "player_statistics": "player_statistics.csv",
            "possession_sequences": "possession_sequences.csv",
            "passes": "passes.csv",
            "passing_network": "passing_network.csv",
            "match_summary": "match_summary.json",
            "tracking_events": (
                events_file.name if events_file is not None else None
            ),
        },
        "passes": {
            "available": True,
            "attempted": len(passes),
            "completed": sum(item["outcome"] == "completed" for item in passes),
            "incomplete": sum(item["outcome"] == "incomplete" for item in passes),
            "unknown": sum(item["outcome"] == "unknown" for item in passes),
        },
        "football_events": {
            "available": False,
            "reason": (
                "The current diagnostic events are tracker lifecycle and "
                "identity events, not validated football events."
            ),
        },
        "tracking_event_count": event_count,
    }
    (output_dir / "match_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _optional_float(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if number == number else None


def _pass_statistics(passes, counts_by_team):
    completed = sum(item["outcome"] == "completed" for item in passes)
    incomplete = sum(item["outcome"] == "incomplete" for item in passes)
    unknown = sum(item["outcome"] == "unknown" for item in passes)
    decided = completed + incomplete
    return {
        "attempted": len(passes),
        "completed": completed,
        "incomplete": incomplete,
        "unknown": unknown,
        "completion_percentage": round(100.0 * completed / decided, 2)
        if decided
        else 0.0,
        "by_team": {
            team: {
                "attempted": counts_by_team[team]["attempted"],
                "completed": counts_by_team[team]["completed"],
                "incomplete": counts_by_team[team]["incomplete"],
                "unknown": counts_by_team[team]["unknown"],
            }
            for team in sorted(TEAM_LABELS)
        },
    }


def _team_summaries(possession_frames, total_frames, pass_counts):
    summaries = []
    for team in sorted(TEAM_LABELS):
        decided_passes = (
            pass_counts[team]["completed"] + pass_counts[team]["incomplete"]
        )
        summaries.append(
            {
                "team": team,
                "valid_outfield_possession_frames": possession_frames[team],
                "possession_percentage": round(
                    100.0 * possession_frames[team] / total_frames, 2
                )
                if total_frames
                else 0.0,
                "passes_attempted": pass_counts[team]["attempted"],
                "passes_completed": pass_counts[team]["completed"],
                "passes_incomplete": pass_counts[team]["incomplete"],
                "passes_unknown": pass_counts[team]["unknown"],
                "pass_completion_percentage": round(
                    100.0
                    * pass_counts[team]["completed"]
                    / decided_passes,
                    2,
                )
                if decided_passes
                else 0.0,
            }
        )
    return summaries


def _player_summaries(players, pass_counts):
    summaries = []
    for global_id, player in sorted(
        players.items(), key=lambda item: int(item[0])
    ):
        known_teams = player["teams"] & TEAM_LABELS
        team = (
            next(iter(known_teams))
            if len(known_teams) == 1
            else "inconsistent"
            if len(known_teams) > 1
            else "unknown"
        )
        counts = pass_counts[global_id]
        decided_passes = counts["completed"] + counts["incomplete"]
        summaries.append(
            {
                "global_id": global_id,
                "team": team,
                "passes_attempted": counts["attempted"],
                "passes_completed": counts["completed"],
                "passes_incomplete": counts["incomplete"],
                "passes_unknown": counts["unknown"],
                "passes_received": counts["received"],
                "pass_completion_percentage": round(
                    100.0 * counts["completed"] / decided_passes, 2
                )
                if decided_passes
                else 0.0,
            }
        )
    return summaries
