"""Plotly visualizations for wrist-relative hand landmarks."""

import numpy as np
import plotly.graph_objects as go


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

_HAND_FEATURES = 63
_N_FRAMES = 30
_N_FEATURES = 126
_HAND_COLORS = ("#2864dc", "#ed8b23")


def _check_shape(x):
    """Convert landmarks to an array and enforce the expected sequence shape."""
    values = np.asarray(x)
    if values.shape != (_N_FRAMES, _N_FEATURES):
        raise ValueError(f"Expected shape (30, 126), got {values.shape}")
    return values


def _hand_traces(frame, offset, joint_values, hand_color, colorbar=False):
    """Build fixed line and joint traces for a detected or missing hand."""
    if np.all(frame == 0):
        return [
            go.Scatter(x=[], y=[], mode="lines", showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip"),
        ]

    points = frame.reshape(21, 3)
    px = points[:, 0] + offset
    py = points[:, 1]
    line_x, line_y = [], []
    for first, second in HAND_CONNECTIONS:
        line_x.extend((px[first], px[second], None))
        line_y.extend((py[first], py[second], None))
    line = go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line={"color": hand_color, "width": 2},
        showlegend=False,
        hoverinfo="skip",
    )
    marker = {
        "size": [13] + [8] * 20,
        "line": {"width": 1, "color": "white"},
    }
    if joint_values is None:
        marker["color"] = hand_color
    else:
        marker.update({
            "color": joint_values,
            "colorscale": "Viridis",
            "cmin": float(np.min(joint_values)),
            "cmax": float(np.max(joint_values)),
            "showscale": colorbar,
        })
    joints = go.Scatter(
        x=px,
        y=py,
        mode="markers",
        marker=marker,
        showlegend=False,
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
    )
    return [line, joints]


def _frame_data(x, offsets, joint_values=None):
    """Create same-order traces for one or more sequences at one frame."""
    data = []
    for sequence_index, (sample, hand_offsets) in enumerate(zip(x, offsets)):
        for hand_index, offset in enumerate(hand_offsets):
            values = None if joint_values is None else joint_values[hand_index * 21:(hand_index + 1) * 21]
            data.extend(_hand_traces(
                sample[hand_index * _HAND_FEATURES:(hand_index + 1) * _HAND_FEATURES],
                offset,
                values,
                _HAND_COLORS[hand_index],
                colorbar=(sequence_index == 0 and hand_index == 0),
            ))
    return data


def _all_coordinates(sequences, offsets):
    """Collect visible trace coordinates across every frame for fixed bounds."""
    xs, ys = [], []
    for frame_id in range(_N_FRAMES):
        for trace in _frame_data([sequence[frame_id] for sequence in sequences], offsets):
            xs.extend(value for value in trace.x if value is not None and np.isfinite(value))
            ys.extend(value for value in trace.y if value is not None and np.isfinite(value))
    if not xs:
        return (-1, 1), (1, -1), 0.25
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    margin = max(span_x, span_y, 1.0) * 0.05
    return (min(xs) - margin, max(xs) + margin), (max(ys) + margin, min(ys) - margin), margin


def _animated_figure(sequences, offsets, title, joint_values=None, half_titles=None):
    """Assemble synchronized frames, controls, labels, and fixed axes."""
    data = _frame_data([sequence[0] for sequence in sequences], offsets, joint_values)
    trace_indices = list(range(len(data)))
    frames = [
        go.Frame(
            name=str(frame_id),
            data=_frame_data([sequence[frame_id] for sequence in sequences], offsets, joint_values),
            traces=trace_indices,
        )
        for frame_id in range(_N_FRAMES)
    ]
    x_range, y_range, margin = _all_coordinates(sequences, offsets)
    annotations = []
    for sequence_index, hand_offsets in enumerate(offsets):
        for hand_index, offset in enumerate(hand_offsets):
            annotations.append({
                "text": ("Left", "Right")[hand_index],
                "x": offset,
                "y": y_range[1] + margin * 0.28,
                "xref": "x",
                "yref": "y",
                "showarrow": False,
                "font": {"size": 10, "color": _HAND_COLORS[hand_index]},
            })
    if half_titles:
        annotations.extend({
            "text": label,
            "x": position,
            "y": 0.94,
            "xref": "x",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 15},
        } for label, position in half_titles)
    steps = [
        {
            "method": "animate",
            "label": str(frame_id),
            "args": [[str(frame_id)], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
            }],
        }
        for frame_id in range(_N_FRAMES)
    ]
    shapes = []
    if half_titles:
        shapes.append({
            "type": "line",
            "x0": 0,
            "x1": 0,
            "y0": y_range[0],
            "y1": y_range[1],
            "xref": "x",
            "yref": "y",
            "line": {"color": "#9aa3af", "width": 1},
        })
    return go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title={"text": title, "x": 0.5, "y": 1.0},
            showlegend=False,
            xaxis={
                "range": x_range,
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "title": None,
            },
            yaxis={
                "range": y_range,
                "showticklabels": False,
                "showgrid": False,
                "zeroline": False,
                "showline": False,
                "title": None,
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            annotations=annotations + [{
                "text": "Landmark replay (hand landmarks only). Each hand is drawn relative to its own wrist.",
                "x": 0.5,
                "y": -0.34,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 10, "color": "#596273"},
            }],
            shapes=shapes,
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": -0.16,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {
                        "frame": {"duration": 80, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                    }]},
                    {"label": "Pause", "method": "animate", "args": [[None], {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    }]},
                ],
            }],
            sliders=[{
                "active": 0,
                "y": -0.04,
                "steps": steps,
                "currentvalue": {"prefix": "Frame "},
            }],
            margin={"t": 135, "b": 155, "l": 30, "r": 30},
        ),
    )


def skeleton_figure(x, title, joint_values=None):
    """Create an animated two-hand landmark skeleton for one sequence."""
    sample = _check_shape(x)
    values = None if joint_values is None else np.asarray(joint_values, dtype=float)
    if values is not None and values.shape != (42,):
        raise ValueError(f"Expected joint_values shape (42,), got {values.shape}")
    return _animated_figure([sample], [(-1.5, 1.5)], title, values)


def side_by_side_figure(x_original, x_stressed, titles):
    """Create one synchronized animated comparison of original and stressed hands."""
    original = _check_shape(x_original)
    stressed = _check_shape(x_stressed)
    if len(titles) != 2:
        raise ValueError("titles must contain original and stressed labels")
    return _animated_figure(
        [original, stressed],
        [(-7.5, -4.5), (4.5, 7.5)],
        "Landmark comparison",
        half_titles=[(titles[0], -6.0), (titles[1], 6.0)],
    )
