"""Apply deterministic stress transformations to sign landmarks."""

from collections import OrderedDict
from functools import partial

import numpy as np


def mirror(X):
    """Negate joint x coordinates and swap the left and right hands."""
    shaped = np.asarray(X).reshape(X.shape[0], X.shape[1], 42, 3)
    mirrored = shaped.copy()
    mirrored[..., 0] *= -1
    mirrored = np.concatenate((mirrored[..., 21:42, :], mirrored[..., :21, :]), axis=2)
    return mirrored.reshape(X.shape).astype(X.dtype, copy=False)


def change_speed(X, factor):
    """Resample each sequence at the requested temporal speed."""
    if factor <= 0:
        raise ValueError("factor must be positive")
    if factor == 1.0:
        return X.copy()
    sample_count, frame_count, feature_count = X.shape
    times = np.minimum(np.arange(frame_count, dtype=np.float64) * factor, frame_count - 1)
    lower = np.floor(times).astype(np.intp)
    upper = np.minimum(lower + 1, frame_count - 1)
    weight = (times - lower).astype(np.float64)[None, :, None]
    values = X[:, lower, :].astype(np.float64) * (1.0 - weight)
    values += X[:, upper, :].astype(np.float64) * weight
    return values.astype(X.dtype, copy=False).reshape(sample_count, frame_count, feature_count)


def add_noise(X, sigma=0.01, seed=0):
    """Add seeded Gaussian noise only to nonzero landmark values."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=X.shape)
    return np.where(X != 0, X + noise, 0).astype(X.dtype, copy=False)


def scale_jitter(X, scale=0.8):
    """Multiply all landmark values by a scale factor."""
    return (X * scale).astype(X.dtype, copy=False)


def drop_frames(X, p=0.15, seed=0):
    """Drop seeded frames and interpolate the remaining frames to full length."""
    if not 0 <= p < 1:
        raise ValueError("p must be in [0, 1)")
    sample_count, frame_count, feature_count = X.shape
    drop_count = round(p * frame_count)
    if drop_count == 0:
        return X.copy()
    rng = np.random.default_rng(seed)
    output = np.empty_like(X)
    target_times = np.arange(frame_count)
    for sample_idx in range(sample_count):
        dropped = rng.choice(frame_count, size=drop_count, replace=False)
        keep = np.ones(frame_count, dtype=bool)
        keep[dropped] = False
        kept_times = target_times[keep]
        values = X[sample_idx, keep, :]
        for feature_idx in range(feature_count):
            output[sample_idx, :, feature_idx] = np.interp(
                target_times, kept_times, values[:, feature_idx]
            )
    return output


TESTS = OrderedDict(
    [
        ("Mirror (left hand)", mirror),
        ("Slower (0.7x)", partial(change_speed, factor=0.7)),
        ("Faster (1.3x)", partial(change_speed, factor=1.3)),
        ("Noise", partial(add_noise, sigma=0.01, seed=0)),
        ("Scale jitter", partial(scale_jitter, scale=0.8)),
        ("Missing frames", partial(drop_frames, p=0.15, seed=0)),
    ]
)
