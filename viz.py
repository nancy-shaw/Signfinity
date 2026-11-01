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


def _check_shape(x):
    """Convert landmarks to an array and enforce the expected sequence shape."""
    values = np.asarray(x)
    if values.shape != (_N_FRAMES, _N_FEATURES):
        raise ValueError(f"Expected shape (30, 126), got {values.shape}")
    return values


def _hand_traces(frame, offset, joint_values, colorbar=False):
    """Build a line trace and a joint trace for one detected hand."""
    traces = []
    detected = not np.all(frame == 0)
    if detected:
        points = frame.reshape(21, 3)
        px = points[:, 0] + offset
        py = points[:, 1]
        line_x, line_y = [], []
        for first, second in HAND_CONNECTIONS:
            line_x.extend((px[first], px[second], None))
            line_y.extend((py[first], py[second], None))
        traces.append(go.Scatter(x=line_x, y=line_y, mode="lines", line={"color": "#7b8794", "width": 2}, hoverinfo="skip"))
        marker = {"size": 8, "line": {"width": 1, "color": "white"}}
        if joint_values is None:
            marker["color"] = "#2864dc"
        else:
            marker.update({"color": joint_values, "colorscale": "Viridis", "cmin": float(np.min(joint_values)), "cmax": float(np.max(joint_values))})
            if colorbar:
                marker["showscale"] = True
        traces.append(go.Scatter(x=px, y=py, mode="markers", marker=marker, hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"))
    else:
        traces.extend((
            go.Scatter(x=[], y=[], mode="lines", line={"color": "#7b8794", "width": 2}, hoverinfo="skip"),
            go.Scatter(x=[], y=[], mode="markers", marker={"size": 8, "color": "#2864dc"}, hoverinfo="skip"),
        ))
    return traces


def _frame_data(x, offsets, joint_values=None):
    """Create the fixed-order traces for one or more sequences at one frame."""
    data = []
    for sequence_index, (sample, hand_offsets) in enumerate(zip(x, offsets)):
        for hand_index, offset in enumerate(hand_offsets):
            colors = None if joint_values is None else joint_values[hand_index * 21:(hand_index + 1) * 21]
            data.extend(_hand_traces(
                sample[hand_index * _HAND_FEATURES:(hand_index + 1) * _HAND_FEATURES],
                offset,
                colors,
                colorbar=(sequence_index == 0 and hand_index == 0),
            ))
    return data


def _all_coordinates(sequences, offsets):
    """Collect visible coordinates to derive stable plot bounds."""
    xs, ys = [], []
    for sequence, hand_offsets in zip(sequences, offsets):
        for frame in sequence:
            for hand_index, offset in enumerate(hand_offsets):
                hand = frame[hand_index * _HAND_FEATURES:(hand_index + 1) * _HAND_FEATURES]
                if np.all(hand == 0):
                    continue
                points = hand.reshape(21, 3)
                xs.extend((points[:, 0] + offset).tolist())
                ys.extend(points[:, 1].tolist())
    if not xs:
        return (-1, 1), (1, -1)
    pad_x = max((max(xs) - min(xs)) * 0.05, 0.25)
    pad_y = max((max(ys) - min(ys)) * 0.05, 0.25)
    return (min(xs) - pad_x, max(xs) + pad_x), (max(ys) + pad_y, min(ys) - pad_y)


def _animated_figure(sequences, offsets, title, joint_values=None, labels=None):
    """Assemble synchronized frames, controls, labels, and fixed axes."""
    data = _frame_data([sequence[0] for sequence in sequences], offsets, joint_values)
    frames = [
        go.Frame(name=str(frame_id), data=_frame_data([sequence[frame_id] for sequence in sequences], offsets, joint_values))
        for frame_id in range(_N_FRAMES)
    ]
    x_range, y_range = _all_coordinates(sequences, offsets)
    annotations = []
    if labels:
        annotations = [
            {"text": label, "x": position, "y": 1.08, "xref": "x", "yref": "paper", "showarrow": False, "font": {"size": 14}}
            for label, position in labels
        ]
    steps = [
        {"method": "animate", "label": str(frame_id), "args": [[str(frame_id)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}]}
        for frame_id in range(_N_FRAMES)
    ]
    return go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis={"range": x_range, "title": "x", "zeroline": False},
            yaxis={"range": y_range, "title": "y", "scaleanchor": "x", "scaleratio": 1, "zeroline": False},
            annotations=annotations,
            updatemenus=[{"type": "buttons", "showactive": False, "x": 0, "y": 1.16, "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
            ]}],
            sliders=[{"active": 0, "steps": steps, "currentvalue": {"prefix": "Frame "}}],
            margin={"t": 105 if labels else 80},
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
        labels=[(titles[0], -6.0), (titles[1], 6.0)],
    )
