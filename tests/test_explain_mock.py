"""Tests for deterministic mock explanation inputs."""

import numpy as np

from explain_mock import fake_predict_proba, make_mock_data, make_mock_words, mock_tests


def test_mock_data_has_expected_shapes_dtype_and_nonzero_landmarks():
    """Mock samples are float32 hand sequences with integer labels."""
    X, y = make_mock_data(n=12, n_classes=4, seed=3)
    assert X.shape == (12, 30, 126)
    assert X.dtype == np.float32
    assert y.shape == (12,)
    assert y.dtype.kind in "iu"
    assert np.all(X != 0)
    assert np.all(np.isfinite(X))


def test_mock_words_and_fake_probabilities_have_expected_shapes():
    """Labels and softmax predictions use the requested class count."""
    X, _ = make_mock_data(n=8, n_classes=5)
    words = make_mock_words(5)
    probabilities = fake_predict_proba(X, n_classes=5)
    assert words == [f"sign_{index}" for index in range(5)]
    assert probabilities.shape == (8, 5)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(probabilities.argmax(axis=1) == np.arange(8) % 5) >= 0.75


def test_mock_generation_and_predictions_are_deterministic_and_input_sensitive():
    """A fixed seed repeats exactly and changed landmarks alter probabilities."""
    X, y = make_mock_data(n=15, n_classes=5, seed=4)
    other_X, other_y = make_mock_data(n=15, n_classes=5, seed=4)
    np.testing.assert_array_equal(X, other_X)
    np.testing.assert_array_equal(y, other_y)
    first = fake_predict_proba(X)
    np.testing.assert_array_equal(first, fake_predict_proba(X.copy()))
    changed = X.copy()
    changed[:, :, 26] += 0.002
    assert not np.allclose(first, fake_predict_proba(changed))


def test_mock_stress_transforms_preserve_shape_and_input():
    """Every mock transformation returns a same-shaped copy without mutation."""
    X, _ = make_mock_data(n=4, n_classes=3, seed=5)
    original = X.copy()
    transforms = mock_tests()
    assert list(transforms) == ["Noise", "Slower", "Mirror"]
    for transform in transforms.values():
        result = transform(X)
        assert result.shape == X.shape
        assert result.dtype == X.dtype
        np.testing.assert_array_equal(X, original)
        assert not np.shares_memory(result, X)
