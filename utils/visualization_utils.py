import cv2

from utils.team_color_utils import TeamColorAssigner


def is_displayable_player(detection):
    """Show resolved outfield players and identified goalkeepers."""
    if (
        detection.get("object_type") == "goalkeeper"
        and detection.get("global_id") is not None
    ):
        return True
    return (
        detection.get("object_type") == "player"
        and detection.get("global_id") is not None
        and detection.get("player_type")
        in (
            TeamColorAssigner.TEAM_1,
            TeamColorAssigner.TEAM_2,
            TeamColorAssigner.OTHER,
            TeamColorAssigner.TEAM_CONFLICT,
        )
    )


def draw_annotations(
    frame, detections, *, copy_frame=True, identity_resolved=True
):
    """Draw team-colored player foot markers and readable IDs on the frame."""
    output = frame.copy() if copy_frame else frame
    for detection in detections:
        if not is_displayable_player(detection):
            continue

        bbox = detection["bbox"]
        x1, _, x2, y2 = map(float, bbox)
        x_center = int((x1 + x2) / 2)
        y_bottom = int(y2)
        width = int(x2 - x1)
        player_type = detection["player_type"]
        color = detection.get("team_color", TeamColorAssigner.NEUTRAL_COLOR)
        if (
            detection.get("object_type") == "goalkeeper"
            and detection.get("player_type")
            not in (TeamColorAssigner.TEAM_1, TeamColorAssigner.TEAM_2)
        ):
            color = TeamColorAssigner.TEAM_COLORS[
                TeamColorAssigner.GOALKEEPER
            ]
        ellipse_x_radius = width
        ellipse_y_radius = int(0.35 * width)
        cv2.ellipse(
            output,
            center=(x_center, y_bottom),
            axes=(ellipse_x_radius, ellipse_y_radius),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        track_id = str(detection["global_id"])
        rect_width, rect_height = 40, 20
        rect_top = y_bottom + ellipse_y_radius - rect_height // 2
        rect_left = x_center - rect_width // 2
        cv2.rectangle(
            output,
            (rect_left, rect_top),
            (rect_left + rect_width, rect_top + rect_height),
            color,
            cv2.FILLED,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 2
        text_width, text_height = cv2.getTextSize(
            track_id, font, font_scale, text_thickness
        )[0]
        cv2.putText(
            output,
            track_id,
            (
                x_center - text_width // 2,
                rect_top + (rect_height + text_height) // 2,
            ),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
        )
    return output
