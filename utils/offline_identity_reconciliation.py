"""Conservative offline tracklet reconciliation for persisted observations."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2


EVIDENCE_FIELDS = (
    "tracklet_a",
    "tracklet_b",
    "raw_tracker_id_a",
    "raw_tracker_id_b",
    "start_frame_a",
    "end_frame_a",
    "start_frame_b",
    "end_frame_b",
    "motion_score",
    "appearance_reid_score",
    "team_agreement",
    "team_conflict",
    "scale_consistency_score",
    "time_gap_score",
    "pitch_space_score",
    "image_plane_distance_pixels",
    "combined_score",
    "best_candidate_score",
    "runner_up_score",
    "score_margin",
    "decision",
    "reason",
    "constraint",
    "algorithm_version",
)

MERGE_FIELDS = (
    "source_fragment_id",
    "prior_fragment_id",
    "proposed_canonical_global_id",
    "prior_start_frame",
    "prior_end_frame",
    "fragment_start_frame",
    "gap_frames",
    "gap_seconds",
    "spatial_distance_pixels",
    "team_consistency",
    "appearance_reid_score",
    "overall_confidence",
    "ambiguity_flag",
    "recommended_action",
    "reason",
    "review_status",
)


def _read_tracklets(observations_path):
    grouped = defaultdict(list)
    with Path(observations_path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):
            if row.get("object_type") not in ("player", "goalkeeper"):
                continue
            if not row.get("tracker_id"):
                continue
            row["_frame"] = int(row["frame_number"])
            row["_tracker"] = int(row["tracker_id"])
            row["_bbox"] = json.loads(row["bbox"]) if row.get("bbox") else None
            row["_foot"] = (
                (float(row["foot_x"]), float(row["foot_y"]))
                if row.get("foot_x") and row.get("foot_y")
                else None
            )
            grouped[row["_tracker"]].append(row)

    tracklets = []
    for tracker_id, rows in grouped.items():
        rows.sort(key=lambda row: row["_frame"])
        segment = []
        for row in rows:
            if segment and row["_frame"] != segment[-1]["_frame"] + 1:
                tracklets.append(_summarize_tracklet(tracker_id, segment))
                segment = []
            segment.append(row)
        if segment:
            tracklets.append(_summarize_tracklet(tracker_id, segment))
    return sorted(
        tracklets,
        key=lambda item: (
            item["start_frame"],
            item["raw_tracker_id"],
            item["end_frame"],
        ),
    )


def _summarize_tracklet(tracker_id, rows):
    labels = Counter(
        row.get("player_type", "")
        for row in rows
        if row.get("player_type") in ("team_1", "team_2")
    )
    team = labels.most_common(1)[0][0] if labels else ""
    boxes = [row["_bbox"] for row in rows if row["_bbox"] and len(row["_bbox"]) == 4]
    sizes = [
        (max(1.0, float(box[2]) - float(box[0])),
         max(1.0, float(box[3]) - float(box[1])))
        for box in boxes
    ]
    mean_size = (
        (
            sum(size[0] for size in sizes) / len(sizes),
            sum(size[1] for size in sizes) / len(sizes),
        )
        if sizes
        else None
    )
    feet = [row["_foot"] for row in rows if row["_foot"] is not None]
    return {
        "raw_tracker_id": tracker_id,
        "start_frame": rows[0]["_frame"],
        "end_frame": rows[-1]["_frame"],
        "tracklet_id": (
            f"{tracker_id}:{rows[0]['_frame']}:{rows[-1]['_frame']}"
        ),
        "observations": len(rows),
        "team": team,
        "team_share": (
            labels[team] / sum(labels.values()) if team else None
        ),
        "mean_size": mean_size,
        "mean_foot": (
            (
                sum(point[0] for point in feet) / len(feet),
                sum(point[1] for point in feet) / len(feet),
            )
            if feet
            else None
        ),
    }


def _candidate_pairs(tracklets, max_gap_frames):
    for index, first in enumerate(tracklets):
        later = []
        for second in tracklets[index + 1 :]:
            gap = second["start_frame"] - first["end_frame"] - 1
            if gap < 0:
                continue
            if gap > max_gap_frames:
                break
            if not first["team"] or first["team"] != second["team"]:
                continue
            later.append((gap, second))
        for gap, second in later[:3]:
            distance = None
            if first["mean_foot"] and second["mean_foot"]:
                dx = first["mean_foot"][0] - second["mean_foot"][0]
                dy = first["mean_foot"][1] - second["mean_foot"][1]
                distance = (dx * dx + dy * dy) ** 0.5
            size_score = None
            if first["mean_size"] and second["mean_size"]:
                ratios = [
                    min(a, b) / max(a, b)
                    for a, b in zip(first["mean_size"], second["mean_size"])
                ]
                size_score = (ratios[0] * ratios[1]) ** 0.5
            yield first, second, gap, distance, size_score


def reconcile_observations(
    observations_path,
    output_dir,
    *,
    fps=30.0,
    max_candidate_gap_frames=900,
    existing_map_path=None,
    video_path=None,
):
    """Build deterministic tracklet IDs and unresolved review proposals.

    The current observation schema contains no appearance embeddings,
    per-observation team confidence, or calibrated pitch coordinates. For that
    reason this version proposes candidates for human review but never merges
    separate tracklets automatically.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_candidate_gap_frames < 0:
        raise ValueError("max_candidate_gap_frames must be non-negative")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracklets = _read_tracklets(observations_path)

    preserved = {}
    preserved_ranges = defaultdict(list)
    if existing_map_path and Path(existing_map_path).is_file():
        with Path(existing_map_path).open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            for row in csv.DictReader(stream):
                tracklet_id = row["tracklet_id"]
                canonical_id = int(row["canonical_id"])
                preserved[tracklet_id] = canonical_id
                preserved_ranges[canonical_id].append(
                    (int(row["start_frame"]), int(row["end_frame"]))
                )
    for ranges in preserved_ranges.values():
        ranges.sort()
        if any(
            current_start <= previous_end
            for (_, previous_end), (current_start, _) in zip(
                ranges, ranges[1:]
            )
        ):
            raise ValueError(
                "Existing canonical ID map violates simultaneous-visibility constraints."
            )
    next_id = max(preserved.values(), default=0) + 1
    mapping = []
    for tracklet in tracklets:
        canonical_id = preserved.get(tracklet["tracklet_id"])
        if canonical_id is None:
            canonical_id = next_id
            next_id += 1
        mapping.append({
            "tracklet_id": tracklet["tracklet_id"],
            "raw_tracker_id": tracklet["raw_tracker_id"],
            "start_frame": tracklet["start_frame"],
            "end_frame": tracklet["end_frame"],
            "canonical_id": canonical_id,
            "team_label": tracklet["team"],
            "team_label_share": tracklet["team_share"],
            "decision": "unmerged_tracklet",
        })
    _write_csv(
        output_dir / "canonical_id_map.csv",
        mapping,
        (
            "tracklet_id",
            "raw_tracker_id",
            "start_frame",
            "end_frame",
            "canonical_id",
            "team_label",
            "team_label_share",
            "decision",
        ),
    )

    evidence = []
    merges = []
    cannot_links = []
    for index, first in enumerate(tracklets):
        for second in tracklets[index + 1 :]:
            if second["start_frame"] > first["end_frame"]:
                break
            if first["raw_tracker_id"] == second["raw_tracker_id"]:
                continue
            cannot_links.append({
                "tracklet_a": first["tracklet_id"],
                "tracklet_b": second["tracklet_id"],
                "raw_tracker_id_a": first["raw_tracker_id"],
                "raw_tracker_id_b": second["raw_tracker_id"],
                "start_frame_a": first["start_frame"],
                "end_frame_a": first["end_frame"],
                "start_frame_b": second["start_frame"],
                "end_frame_b": second["end_frame"],
                "motion_score": "",
                "appearance_reid_score": "",
                "team_agreement": (
                    first["team"] == second["team"]
                    if first["team"] and second["team"]
                    else ""
                ),
                "team_conflict": (
                    first["team"] != second["team"]
                    if first["team"] and second["team"]
                    else ""
                ),
                "scale_consistency_score": "",
                "time_gap_score": "",
                "pitch_space_score": "",
                "image_plane_distance_pixels": "",
                "combined_score": "",
                "best_candidate_score": "",
                "runner_up_score": "",
                "score_margin": "",
                "decision": "rejected",
                "reason": "tracklets overlap in at least one frame",
                "constraint": "simultaneous_visibility_cannot_link",
                "algorithm_version": "offline-tracklet-review-v1",
            })
    for first, second, gap, distance, size_score in _candidate_pairs(
        tracklets, max_candidate_gap_frames
    ):
        evidence.append({
            "tracklet_a": first["tracklet_id"],
            "tracklet_b": second["tracklet_id"],
            "raw_tracker_id_a": first["raw_tracker_id"],
            "raw_tracker_id_b": second["raw_tracker_id"],
            "start_frame_a": first["start_frame"],
            "end_frame_a": first["end_frame"],
            "start_frame_b": second["start_frame"],
            "end_frame_b": second["end_frame"],
            "motion_score": "",
            "appearance_reid_score": "",
            "team_agreement": True,
            "team_conflict": False,
            "scale_consistency_score": size_score,
            "time_gap_score": "",
            "pitch_space_score": "",
            "image_plane_distance_pixels": distance,
            "combined_score": "",
            "best_candidate_score": "",
            "runner_up_score": "",
            "score_margin": "",
            "decision": "unresolved",
            "reason": (
                "appearance/ReID, team confidence, and calibrated pitch "
                "or camera-motion evidence are unavailable"
            ),
            "constraint": "insufficient_evidence_for_automatic_merge",
            "algorithm_version": "offline-tracklet-review-v1",
        })
        merges.append({
            "source_fragment_id": second["tracklet_id"],
            "prior_fragment_id": first["tracklet_id"],
            "proposed_canonical_global_id": "",
            "prior_start_frame": first["start_frame"],
            "prior_end_frame": first["end_frame"],
            "fragment_start_frame": second["start_frame"],
            "gap_frames": gap,
            "gap_seconds": round(gap / fps, 3),
            "spatial_distance_pixels": distance,
            "team_consistency": "same saved label; confidence unavailable",
            "appearance_reid_score": "",
            "overall_confidence": "",
            "ambiguity_flag": "unresolved",
            "recommended_action": "human_review",
            "reason": evidence[-1]["reason"],
            "review_status": "unreviewed",
        })
    _write_csv(
        output_dir / "identity_association_evidence.csv",
        evidence + cannot_links,
        EVIDENCE_FIELDS,
    )
    _write_csv(
        output_dir / "ambiguous_identity_matches.csv",
        evidence,
        EVIDENCE_FIELDS,
    )
    _write_csv(
        output_dir / "candidate_identity_merges.csv",
        merges,
        MERGE_FIELDS,
    )
    contact_sheet_path = None
    if video_path is not None:
        contact_sheet_path = _write_review_contact_sheet(
            video_path,
            evidence,
            output_dir / "review_images_or_clips",
        )
    report = {
        "tracklets": len(tracklets),
        "canonical_identities": len(
            {row["canonical_id"] for row in mapping}
        ),
        "automatic_merges": 0,
        "unresolved_candidate_pairs": len(evidence),
        "simultaneous_visibility_cannot_link_pairs": len(cannot_links),
        "simultaneous_identity_conflicts": 0,
        "cannot_link_violations": 0,
        "automatic_merge_accuracy": "NOT EVALUATED: no labeled examples",
        "possession_accuracy": "NOT EVALUATED: no hand-labeled slice",
        "review_contact_sheet": (
            str(contact_sheet_path)
            if contact_sheet_path
            else (
                "NOT GENERATED: no candidate pairs available"
                if video_path is not None
                else "NOT GENERATED: input video was not provided"
            )
        ),
        "notes": [
            "Each contiguous raw tracklet remains a separate canonical identity.",
            "Image-plane distances are pixels, not metres.",
            "No associations are automatically accepted without the missing evidence.",
        ],
    }
    (output_dir / "reconciliation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reconciliation_report.md").write_text(
        "# Offline identity reconciliation\n\n"
        f"- Tracklets: {report['tracklets']}\n"
        f"- Canonical identities: {report['canonical_identities']}\n"
        f"- Automatic merges: {report['automatic_merges']}\n"
        f"- Unresolved candidate pairs: {report['unresolved_candidate_pairs']}\n"
        f"- Simultaneous-visibility cannot-link pairs: "
        f"{report['simultaneous_visibility_cannot_link_pairs']}\n"
        f"- Simultaneous-visibility identity conflicts: "
        f"{report['simultaneous_identity_conflicts']}\n"
        f"- Cannot-link violations: {report['cannot_link_violations']}\n"
        f"- Automatic-merge accuracy: {report['automatic_merge_accuracy']}\n"
        f"- Possession accuracy: {report['possession_accuracy']}\n\n"
        f"- Review contact sheet: {report['review_contact_sheet']}\n\n"
        "No tracklets were automatically merged because appearance/ReID, "
        "team confidence, and calibrated pitch/camera-motion evidence are "
        "not present in the persisted observations. Image-plane distances "
        "are pixels and are not interpreted as metres.\n",
        encoding="utf-8",
    )
    return report


def _write_review_contact_sheet(video_path, evidence, output_dir):
    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open review video: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    try:
        for row in evidence[:12]:
            pair = []
            for frame_key, tracklet_key in (
                ("start_frame_a", "tracklet_a"),
                ("start_frame_b", "tracklet_b"),
            ):
                frame_number = int(row[frame_key])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                success, frame = capture.read()
                if not success:
                    raise OSError(
                        f"Could not read review frame {frame_number} "
                        f"from {video_path}."
                    )
                thumbnail = cv2.resize(frame, (240, 135))
                cv2.putText(
                    thumbnail,
                    f"{row[tracklet_key]} frame {frame_number}",
                    (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                pair.append(thumbnail)
            tiles.append(cv2.hconcat(pair))
    finally:
        capture.release()

    sheet = cv2.vconcat(tiles) if tiles else None
    if sheet is None:
        return None
    path = output_dir / "identity_review_contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Could not write identity review contact sheet: {path}")
    return path


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
