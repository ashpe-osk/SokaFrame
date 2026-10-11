"""Bottom-panel composition for team possession, live pass stats, and radar."""

import cv2
import numpy as np

from utils.team_color_utils import TeamColorAssigner


LAYOUT = "panel"
PANEL_HEIGHT = 270
RADAR_SIZE = (420, 245)


def output_size(frame_width, frame_height, layout=LAYOUT):
    if layout == "panel":
        return frame_width, frame_height + PANEL_HEIGHT
    if layout == "overlay":
        return frame_width, frame_height
    raise ValueError(f"Unsupported dashboard layout: {layout}")


def compose_dashboard(
    frame,
    radar,
    estimator,
    layout=LAYOUT,
    team_colors=None,
    pass_counts=None,
):
    if layout == "overlay":
        return _compose_overlay(
            frame,
            radar,
            estimator,
            team_colors,
            pass_counts,
        )
    if layout != "panel":
        raise ValueError(f"Unsupported dashboard layout: {layout}")

    frame_height, frame_width = frame.shape[:2]
    canvas = np.empty(
        (frame_height + PANEL_HEIGHT, frame_width, 3), dtype=np.uint8
    )
    canvas[:frame_height] = frame
    canvas[frame_height:] = (22, 28, 34)
    colors = {**TeamColorAssigner.TEAM_COLORS, **(team_colors or {})}
    if pass_counts is not None:
        _draw_pass_overlay(canvas[:frame_height], pass_counts, colors)
    radar_width = min(RADAR_SIZE[0], frame_width - 20)
    radar_height = min(RADAR_SIZE[1], PANEL_HEIGHT - 16)
    resized_radar = (
        radar
        if radar.shape[:2] == (radar_height, radar_width)
        else cv2.resize(
            radar,
            (radar_width, radar_height),
            interpolation=cv2.INTER_AREA,
        )
    )
    radar_x = frame_width - radar_width - 10
    radar_y = frame_height + (PANEL_HEIGHT - radar_height) // 2
    canvas[
        radar_y : radar_y + radar_height,
        radar_x : radar_x + radar_width,
    ] = resized_radar
    _draw_possession_panel(
        canvas[frame_height:],
        estimator,
        radar_x,
        colors,
    )
    return canvas


def _draw_possession_panel(panel, estimator, available_width, team_colors=None):
    colors = {**TeamColorAssigner.TEAM_COLORS, **(team_colors or {})}
    control = estimator.team_control_percentages()
    x_label = 20
    x_bar = 160
    available_bar_width = available_width - x_bar - 110
    bar_width = max(120, min(440, available_bar_width))

    cv2.putText(
        panel,
        "POSSESSION - TEAM CONTROL",
        (x_label, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    for team, label, y in (
        (TeamColorAssigner.TEAM_1, "Team 1", 78),
        (TeamColorAssigner.TEAM_2, "Team 2", 120),
    ):
        color = colors[team]
        percentage = control[team]
        cv2.putText(
            panel,
            label,
            (x_label, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            panel,
            (x_bar, y - 12),
            (x_bar + bar_width, y + 10),
            (55, 62, 66),
            cv2.FILLED,
        )
        fill_width = round(bar_width * percentage / 100.0)
        if fill_width:
            cv2.rectangle(
                panel,
                (x_bar, y - 12),
                (x_bar + fill_width, y + 10),
                color,
                cv2.FILLED,
            )
        cv2.putText(
            panel,
            f"{percentage:.1f}%",
            (x_bar + bar_width + 12, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    legend = (
        (TeamColorAssigner.TEAM_1, "Team 1"),
        (TeamColorAssigner.TEAM_2, "Team 2"),
        (TeamColorAssigner.GOALKEEPER, "Goalkeeper"),
    )
    legend_x = x_label
    for key, label in legend:
        color = colors[key]
        cv2.rectangle(
            panel,
            (legend_x, 198),
            (legend_x + 18, 214),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            panel,
            label,
            (legend_x + 25, 212),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (225, 230, 232),
            1,
            cv2.LINE_AA,
        )
        legend_x += 150 if key != TeamColorAssigner.GOALKEEPER else 175

def _draw_pass_summary(panel, x, pass_counts, colors, y_offset=0):
    cv2.putText(
        panel,
        "PASSES",
        (x, y_offset + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    columns = (
        ("Attempted", x + 68, x + 88),
        ("Completed", x + 151, x + 175),
        ("Incomplete", x + 238, x + 258),
    )
    for label, column_x, _ in columns:
        cv2.putText(
            panel,
            label,
            (column_x, y_offset + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (165, 175, 180),
            1,
            cv2.LINE_AA,
        )
    for team, label, y in (
        (TeamColorAssigner.TEAM_1, "Team 1", 91),
        (TeamColorAssigner.TEAM_2, "Team 2", 133),
    ):
        stats = pass_counts.get(team, {})
        cv2.putText(
            panel,
            label,
            (x, y_offset + y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            colors[team],
            1,
            cv2.LINE_AA,
        )
        for key, (_, _, value_x) in zip(
            ("attempted", "completed", "incomplete"),
            columns,
        ):
            cv2.putText(
                panel,
                str(max(0, int(stats.get(key, 0)))),
                (value_x, y_offset + y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )


def _draw_pass_overlay(frame, pass_counts, colors):
    frame_width = frame.shape[1]
    panel_width = min(320, frame_width - 24)
    panel_height = 160
    x = frame_width - panel_width - 12
    y = 12
    roi = frame[y : y + panel_height, x : x + panel_width]
    background = np.full_like(roi, (22, 28, 34))
    frame[y : y + panel_height, x : x + panel_width] = cv2.addWeighted(
        background,
        0.62,
        roi,
        0.38,
        0,
    )
    cv2.rectangle(
        frame,
        (x, y),
        (x + panel_width - 1, y + panel_height - 1),
        (95, 112, 119),
        1,
        cv2.LINE_AA,
    )
    _draw_pass_summary(frame, x + 12, pass_counts, colors, y_offset=y)


def _compose_overlay(
    frame, radar, estimator, team_colors=None, pass_counts=None
):
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    radar_width, radar_height = RADAR_SIZE
    scale = min(0.20, (frame_width - 24) / radar_width, (frame_height - 24) / radar_height)
    radar_width = max(1, round(radar_width * scale))
    radar_height = max(1, round(radar_height * scale))
    small_radar = cv2.resize(radar, (radar_width, radar_height))
    x = frame_width - radar_width - 12
    y = frame_height - radar_height - 12
    roi = output[y : y + radar_height, x : x + radar_width]
    output[y : y + radar_height, x : x + radar_width] = cv2.addWeighted(
        small_radar, 0.90, roi, 0.10, 0
    )

    colors = {**TeamColorAssigner.TEAM_COLORS, **(team_colors or {})}
    control = estimator.team_control_percentages()
    labels = (
        (f"T1 {control['team_1']:.0f}%", colors[TeamColorAssigner.TEAM_1]),
        (f"T2 {control['team_2']:.0f}%", colors[TeamColorAssigner.TEAM_2]),
    )
    x = 20
    for label, color in labels:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(
            output,
            (x - 6, 20),
            (x + text_size[0] + 6, 44),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            label,
            (x, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (8, 10, 12),
            1,
            cv2.LINE_AA,
        )
        x += text_size[0] + 12
    if pass_counts is not None:
        _draw_pass_overlay(output, pass_counts, colors)
    return output
