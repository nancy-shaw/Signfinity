"""Unit tests for the landmark stress transformations."""

import numpy as np
import pytest

from stress import TESTS, add_noise, change_speed, drop_frames, mirror, scale_jitter


def sample_array(seed=11):
    """Create a small float32 landmark batch with no fully zero frames."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(4, 30, 126)).astype(np.float32)


def test_mirror_is_an_involution():
    """Applying the mirror transform twice restores the original values."""
    X = sample_array()
    np.testing.assert_array_equal(mirror(mirror(X)), X)


def test_mirror_swaps_and_negates_joint_zero_x():
    """The left wrist x value moves to the right wrist with its sign flipped."""
    X = sample_array()
    np.testing.assert_array_equal(mirror(X)[..., 63], -X[..., 0])


def test_change_speed_one_returns_original_values():
    """A speed factor of one leaves every value unchanged."""
    X = sample_array()
    np.testing.assert_array_equal(change_speed(X, 1.0), X)


@pytest.mark.parametrize("transform", TESTS.values())
def test_each_stress_test_preserves_shape_and_dtype(transform):
    """Every registered transform preserves the input shape and dtype."""
    X = sample_array()
    result = transform(X)
    assert result.shape == X.shape
    assert result.dtype == X.dtype


@pytest.mark.parametrize(
    "transform",
    [mirror, lambda X: change_speed(X, 0.7), add_noise, scale_jitter, drop_frames],
)
def test_transform_does_not_modify_input(transform):
    """Stress transforms leave their input array untouched."""
    X = sample_array()
    original = X.copy()
    transform(X)
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("transform", [add_noise, drop_frames])
def test_seeded_transforms_are_repeatable_and_seed_sensitive(transform):
    """Randomized transforms repeat for a seed and change for another seed."""
    X = sample_array()
    first = transform(X, seed=4)
    np.testing.assert_array_equal(first, transform(X, seed=4))
    assert not np.array_equal(first, transform(X, seed=5))


def test_add_noise_preserves_zero_values():
    """Noise leaves undetected landmarks and zero wrist values exactly zero."""
    X = sample_array()
    X[:, :, :3] = 0
    X[:, 0, 30:60] = 0
    result = add_noise(X, seed=7)
    assert np.all(result[:, :, :3] == 0)
    assert np.all(result[:, 0, 30:60] == 0)


def test_drop_frames_does_not_create_zero_frames():
    """Interpolation does not introduce fully zero frames into nonzero data."""
    X = sample_array()
    result = drop_frames(X, seed=9)
    assert not np.any(np.all(result == 0, axis=2))


def test_registered_test_names_are_ordered_and_exact():
    """The registry exposes the six required tests in the required order."""
    assert list(TESTS) == [
        "Mirror (left hand)",
        "Slower (0.7x)",
        "Faster (1.3x)",
        "Noise",
        "Scale jitter",
        "Missing frames",
    ]
