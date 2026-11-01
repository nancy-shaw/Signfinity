"""Tests for deterministic mock explanation inputs."""

import numpy as np

from explain_mock import (
    build_error_matrix,
    fake_predict_proba,
    make_mock_data,
    make_mock_words,
    mock_tests,
)

try:
    from explain import frame_importance
except ImportError:
    def frame_importance(predict_proba, x, class_id):
        """Calculate frame drops for this isolated mock test if explain.py is absent."""
        variants = np.repeat(np.asarray(x)[None, :, :], 30, axis=0)
        variants[0, 0] = x[1]
        variants[-1, -1] = x[-2]
        for frame_id in range(1, 29):
            variants[frame_id, frame_id] = (x[frame_id - 1] + x[frame_id + 1]) / 2
        probabilities = predict_proba(np.concatenate((np.asarray(x)[None, :, :], variants)))
        return probabilities[0, class_id] - probabilities[1:, class_id]


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
    top_probabilities = probabilities.max(axis=1)
    second_probabilities = np.sort(probabilities, axis=1)[:, -2]
    assert np.all((top_probabilities >= 0.55) & (top_probabilities <= 0.85))
    assert np.all(second_probabilities > 0.05)


def test_mock_generation_and_predictions_are_deterministic_and_input_sensitive():
    """A fixed seed repeats exactly and changed landmarks alter probabilities."""
    X, y = make_mock_data(n=15, n_classes=5, seed=4)
    other_X, other_y = make_mock_data(n=15, n_classes=5, seed=4)
    np.testing.assert_array_equal(X, other_X)
    np.testing.assert_array_equal(y, other_y)
    first = fake_predict_proba(X)
    np.testing.assert_array_equal(first, fake_predict_proba(X.copy()))
    changed = X.copy()
    changed[:, 10, 89] += 0.002
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


def test_build_error_matrix_uses_both_ids_and_blanks_diagonal():
    """A ranked error matrix is square and preserves every involved sign pair."""
    counts = np.zeros((6, 6), dtype=int)
    counts[0, 1] = 10
    counts[1, 2] = 8
    counts[2, 0] = 7
    counts[4, 5] = 2
    words = make_mock_words(6)
    labels, matrix = build_error_matrix(counts, words, max_classes=15)
    assert labels == ["sign_0", "sign_1", "sign_2", "sign_4", "sign_5"]
    assert matrix.shape == (len(labels), len(labels))
    assert np.all(np.isnan(np.diag(matrix)))
    positions = {label: index for index, label in enumerate(labels)}
    for true_id, pred_id in ((0, 1), (1, 2), (2, 0), (4, 5)):
        assert matrix[positions[words[true_id]], positions[words[pred_id]]] > 0


def test_build_error_matrix_returns_empty_square_for_no_errors():
    """A diagonal-only confusion matrix has no error labels or cells."""
    counts = np.diag([3, 4, 2])
    labels, matrix = build_error_matrix(counts, make_mock_words(3))
    assert labels == []
    assert matrix.shape == (0, 0)


def test_mock_frame_importance_peaks_in_frames_eight_through_twelve():
    """The mock predictor's right-index class cue is concentrated at frame ten."""
    X, y = make_mock_data(n=2, n_classes=5, seed=7)
    sample = X[1]
    class_id = int(y[1])
    predict_proba = lambda batch: fake_predict_proba(batch, n_classes=5)
    probabilities = predict_proba(sample[None, :, :])[0]
    drops = frame_importance(predict_proba, sample, class_id)
    assert int(np.argmax(probabilities)) == class_id
    assert drops.shape == (30,)
    assert 8 <= int(np.argmax(drops)) <= 12
    assert float(np.max(drops)) > 0.05
    assert 0.55 <= float(np.max(probabilities)) <= 0.85
    assert np.sort(probabilities)[-2] > 0.05
