"""Deterministic mock landmarks, predictions, and stress transforms."""

from collections import OrderedDict

import numpy as np


_TEMPLATE = np.array([
    (0.00001, -0.00001, 0.00001),
    (-0.08, -0.06, 0.012), (-0.15, -0.12, 0.015), (-0.22, -0.19, 0.018), (-0.27, -0.27, 0.021),
    (-0.16, -0.18, 0.024), (-0.18, -0.29, 0.027), (-0.19, -0.39, 0.030), (-0.20, -0.48, 0.033),
    (-0.06, -0.22, 0.036), (-0.07, -0.35, 0.039), (-0.07, -0.47, 0.042), (-0.07, -0.55, 0.045),
    (0.06, -0.21, 0.048), (0.08, -0.33, 0.051), (0.09, -0.44, 0.054), (0.10, -0.51, 0.057),
    (0.17, -0.17, 0.060), (0.20, -0.27, 0.063), (0.22, -0.36, 0.066), (0.24, -0.43, 0.069),
], dtype=np.float32)
_FINGERTIPS = (4, 8, 12, 16, 20)


def make_mock_data(n=60, n_classes=10, seed=0):
    """Create deterministic smooth two-hand landmark sequences and labels."""
    if n < 0 or n_classes < 1:
        raise ValueError("n must be nonnegative and n_classes must be positive")
    rng = np.random.default_rng(seed)
    labels = np.arange(n, dtype=int) % n_classes
    X = np.empty((n, 30, 126), dtype=np.float32)
    for sample_id, class_id in enumerate(labels):
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
                if hand_id == 0:
                    hand[8, 2] = 0.07 + 0.01 * class_id
                    hand[4, 2] = 0.08 if sample_id % 13 == 0 else 0.03
                start = hand_id * 63
                sequence[frame_id, start:start + 63] = hand.reshape(-1)
        # Keep every stored coordinate nonzero, including nominal wrist coordinates.
        sequence[sequence == 0] = np.float32(1e-6)
        X[sample_id] = sequence
    return X, labels


def make_mock_words(n_classes):
    """Return simple display labels for mock class ids."""
    return [f"sign_{class_id}" for class_id in range(n_classes)]


def fake_predict_proba(X, n_classes=10):
    """Return deterministic softmax probabilities from encoded landmark poses."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (30, 126):
        raise ValueError(f"X must have shape (N, 30, 126), got {values.shape}")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    class_signal = values[:, :, 26].mean(axis=1)
    centers = 0.07 + 0.01 * np.arange(n_classes)
    logits = -2.0 * ((class_signal[:, None] - centers[None, :]) / 0.01) ** 2
    predicted = logits.argmax(axis=1)
    error_marker = values[:, :, 14].mean(axis=1) > 0.055
    for sample_id in np.flatnonzero(error_marker):
        wrong_id = int((predicted[sample_id] + 1) % n_classes)
        logits[sample_id] = -8.0
        logits[sample_id, wrong_id] = 8.0
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
