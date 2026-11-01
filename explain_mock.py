"""Deterministic mock landmarks, predictions, and stress transforms."""

from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go


_TEMPLATE = np.array([
    (0.00001, -0.00001, 0.00001),
    (-0.08, -0.06, 0.012), (-0.15, -0.12, 0.015), (-0.22, -0.19, 0.018), (-0.27, -0.27, 0.021),
    (-0.16, -0.18, 0.024), (-0.18, -0.29, 0.027), (-0.19, -0.39, 0.030), (-0.20, -0.48, 0.033),
    (-0.06, -0.22, 0.036), (-0.07, -0.35, 0.039), (-0.07, -0.47, 0.042), (-0.07, -0.55, 0.045),
    (0.06, -0.21, 0.048), (0.08, -0.33, 0.051), (0.09, -0.44, 0.054), (0.10, -0.51, 0.057),
    (0.17, -0.17, 0.060), (0.20, -0.27, 0.063), (0.22, -0.36, 0.066), (0.24, -0.43, 0.069),
], dtype=np.float32)
_FINGERTIPS = (4, 8, 12, 16, 20)
_INDEX_PULSE = {8: 0.25, 9: 0.6, 10: 1.0, 11: 0.6, 12: 0.25}
_FRAME_WEIGHTS = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
_INDEX_BASE = float(_TEMPLATE[5:9, 2].mean())


def make_mock_data(n=60, n_classes=10, seed=0):
    """Create deterministic smooth two-hand landmark sequences and labels."""
    if n < 0 or n_classes < 1:
        raise ValueError("n must be nonnegative and n_classes must be positive")
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=int) % n_classes
    X = np.empty((n, 30, 126), dtype=np.float32)
    for sample_id, class_id in enumerate(labels):
        # Misclassified examples encode the class the mock model will predict.
        model_class_id = (class_id + 1) % n_classes if sample_id % 13 == 0 else class_id
        phase = rng.uniform(0, 2 * np.pi)
        sequence = np.empty((30, 126), dtype=np.float32)
        for frame_id in range(30):
            wave = np.sin(2 * np.pi * frame_id / 30 + phase)
            for hand_id, mirror in enumerate((-1.0, 1.0)):
                hand = _TEMPLATE.copy()
                hand[:, 0] *= mirror
                for tip_order, joint_id in enumerate(_FINGERTIPS):
                    hand[joint_id, 0] += 0.012 * wave * (tip_order + 1) / 5
                    hand[joint_id, 1] += 0.008 * np.cos(2 * np.pi * frame_id / 30 + phase + tip_order) 
                if hand_id == 1 and frame_id in _INDEX_PULSE:
                    target_z = 0.06 + 0.01 * model_class_id
                    base_z = _TEMPLATE[5:9, 2]
                    hand[5:9, 2] = base_z + _INDEX_PULSE[frame_id] * (target_z - base_z)
                start = hand_id * 63
                sequence[frame_id, start:start + 63] = hand.reshape(-1)
        # Keep every stored coordinate nonzero, including nominal wrist coordinates.
        sequence[sequence == 0] = np.float32(1e-6)
        X[sample_id] = sequence
    return X, labels


def make_mock_words(n_classes):
    """Return simple display labels for mock class ids."""
    return [f"sign_{class_id}" for class_id in range(n_classes)]


def build_error_matrix(counts, words, max_classes=15):
    """Return ranked words and a square NaN-diagonal matrix of errors."""
    counts = np.asarray(counts)
    errors = counts.astype(float, copy=True)
    np.fill_diagonal(errors, 0)
    pairs = [
        (int(true_id), int(pred_id), int(errors[true_id, pred_id]))
        for true_id, pred_id in zip(*np.nonzero(errors))
    ]
    pairs.sort(key=lambda pair: (-pair[2], pair[0], pair[1]))
    involved = {class_id for true_id, pred_id, _ in pairs[:15] for class_id in (true_id, pred_id)}
    if not involved or max_classes <= 0:
        return [], np.empty((0, 0), dtype=float)
    error_mass = errors.sum(axis=0) + errors.sum(axis=1)
    class_ids = sorted(sorted(involved, key=lambda class_id: (-int(error_mass[class_id]), class_id))[:max_classes])
    labels = [words[class_id] for class_id in class_ids]
    matrix = counts[np.ix_(class_ids, class_ids)].astype(float)
    np.fill_diagonal(matrix, np.nan)
    return labels, matrix


def build_error_heatmap(matrix, labels):
    """Build a serializable square error heatmap with a muted diagonal."""
    matrix = np.asarray(matrix, dtype=float)
    text = np.full(matrix.shape, "", dtype=object)
    for row, col in zip(*np.nonzero(np.isfinite(matrix) & (matrix > 0))):
        text[row, col] = str(int(matrix[row, col]))
    zmax = max(1, int(np.nanmax(matrix)))
    errors = go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        zmin=0,
        zmax=zmax,
        colorscale=[[0, "white"], [1, "red"]],
        colorbar={"title": "Errors"},
        hovertemplate="True: %{y} / Predicted: %{x} / Count: %{z}",
    )
    diagonal = np.full(matrix.shape, np.nan)
    np.fill_diagonal(diagonal, 1.0)
    diagonal_trace = go.Heatmap(
        z=diagonal,
        x=labels,
        y=labels,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#e5e7eb"], [1, "#e5e7eb"]],
        showscale=False,
        hoverinfo="skip",
        hoverongaps=False,
    )
    figure = go.Figure([errors, diagonal_trace])
    figure.update_layout(
        title="Most confused signs (errors only)",
        height=520,
        margin={"t": 65, "b": 185, "l": 160, "r": 35},
        xaxis={"title": "Predicted sign", "tickangle": 45, "automargin": True},
        yaxis={
            "title": "True sign (actual)",
            "autorange": "reversed",
            "scaleanchor": "x",
            "scaleratio": 1,
            "automargin": True,
        },
    )
    return figure


def build_importance_bar(values, labels, title, x_title):
    """Build a signed probability-drop bar chart, or None for negligible values."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).any() or np.nanmax(np.abs(values)) < 1e-6:
        return None
    finite_values = np.nan_to_num(values, nan=0.0)
    low = min(0.0, float(finite_values.min()))
    high = max(0.0, float(finite_values.max()))
    padding = max(high - low, 1e-6) * 0.12
    colors = ["#2864dc" if value >= 0 else "#ed8b23" for value in finite_values]
    figure = go.Figure(go.Bar(x=labels, y=finite_values, marker_color=colors))
    figure.update_layout(
        title=title,
        height=340,
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title="Drop in probability",
        yaxis={
            "range": [low - padding, high + padding],
            "zeroline": True,
            "zerolinecolor": "#333333",
            "zerolinewidth": 1,
        },
    )
    return figure


def fake_predict_proba(X, n_classes=10):
    """Return deterministic softmax probabilities from encoded landmark poses."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (30, 126):
        raise ValueError(f"X must have shape (N, 30, 126), got {values.shape}")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    index_z_features = np.array([80, 83, 86, 89])
    frame_signal = values[:, 8:13, index_z_features].mean(axis=2)
    class_signal = frame_signal @ _FRAME_WEIGHTS
    centers = _INDEX_BASE + 0.69 * ((0.06 + 0.01 * np.arange(n_classes)) - _INDEX_BASE)
    temperature = 2.0
    logits = -temperature * ((class_signal[:, None] - centers[None, :]) / 0.01) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _noise(X):
    """Return a copy with reproducible small Gaussian perturbations."""
    rng = np.random.default_rng(0)
    return (X + rng.normal(0, 0.02, size=X.shape)).astype(X.dtype, copy=False)


def _slower(X):
    """Resample frames to a slower temporal sequence without mutating input."""
    values = np.asarray(X)
    source_t = np.arange(values.shape[1], dtype=float)
    target_t = np.clip(np.arange(values.shape[1], dtype=float) * 0.7, 0, values.shape[1] - 1)
    lower = np.floor(target_t).astype(int)
    upper = np.minimum(lower + 1, values.shape[1] - 1)
    alpha = (target_t - lower)[None, :, None]
    output = values[:, lower, :] * (1 - alpha) + values[:, upper, :] * alpha
    del source_t
    return output.astype(values.dtype, copy=False)


def _mirror(X):
    """Negate joint x values and swap left and right hand feature blocks."""
    values = np.asarray(X)
    shaped = values.reshape(values.shape[0], values.shape[1], 42, 3).copy()
    shaped[..., 0] *= -1
    mirrored = np.concatenate((shaped[..., 21:, :], shaped[..., :21, :]), axis=2)
    return mirrored.reshape(values.shape).astype(values.dtype, copy=False)


def mock_tests():
    """Return the ordered deterministic mock stress transformations."""
    return OrderedDict((
        ("Noise", _noise),
        ("Slower", _slower),
        ("Mirror", _mirror),
    ))
