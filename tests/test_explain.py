"""Unit tests for confusion analysis helpers."""

import numpy as np

from explain import (
    confusion_counts,
    frame_importance,
    joint_group_importance,
    new_errors,
    top_confused_classes,
    top_confused_pairs,
    wrong_prediction_report,
)


def test_confusion_counts_uses_true_classes_as_rows_and_fixed_shape():
    """Counts predictions in true-class rows, including absent classes."""
    result = confusion_counts([0, 0, 1, 2], [0, 1, 1, 0], n_classes=4)
    expected = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype.kind in "iu"


def test_confusion_counts_empty_inputs_return_zero_matrix():
    """Empty label arrays produce the requested all-zero matrix."""
    np.testing.assert_array_equal(
        confusion_counts([], [], n_classes=3), np.zeros((3, 3), dtype=int)
    )


def test_top_confused_pairs_excludes_diagonal_and_sorts_ties_by_ids():
    """Off-diagonal pairs sort by count descending, then class ids."""
    pairs = top_confused_pairs(
        [0, 0, 1, 1, 2], [1, 1, 0, 2, 2], words=["A", "B", "C"], k=3
    )
    assert pairs == [
        {"true_id": 0, "pred_id": 1, "true_word": "A", "pred_word": "B", "count": 2},
        {"true_id": 1, "pred_id": 0, "true_word": "B", "pred_word": "A", "count": 1},
        {"true_id": 1, "pred_id": 2, "true_word": "B", "pred_word": "C", "count": 1},
    ]


def test_new_errors_counts_regressions_improvements_and_transitions():
    """Compare correct-to-wrong and wrong-to-correct prediction changes."""
    result = new_errors(
        [0, 1, 2, 2, 1],
        [0, 0, 2, 1, 1],
        [1, 1, 0, 2, 2],
    )
    assert result == {
        "n_newly_wrong": 3,
        "n_newly_right": 2,
        "top_transitions": [
            {"true_id": 0, "pred_id": 1, "count": 1},
            {"true_id": 1, "pred_id": 2, "count": 1},
            {"true_id": 2, "pred_id": 0, "count": 1},
        ],
    }


def test_top_confused_classes_ranks_error_counts_and_returns_none_when_clean():
    """Rank classes by their off-diagonal counts and skip classes without errors."""
    counts = np.array([[2, 3, 0], [1, 1, 2], [0, 0, 4]])
    assert top_confused_classes(counts, k=2) == [0, 1]
    assert top_confused_classes(np.diag([3, 2, 1])) == []


def test_all_correct_predictions_have_no_confusions_or_new_errors():
    """Clean predictions yield empty confusion rankings and zero regressions."""
    labels = [0, 1, 2]
    assert top_confused_pairs(labels, labels) == []
    assert new_errors(labels, labels, labels) == {
        "n_newly_wrong": 0,
        "n_newly_right": 0,
        "top_transitions": [],
    }


def test_new_errors_empty_input_returns_empty_result():
    """Empty prediction arrays yield zero changes and no transitions."""
    assert new_errors([], [], []) == {
        "n_newly_wrong": 0,
        "n_newly_right": 0,
        "top_transitions": [],
    }


def test_frame_importance_peaks_at_frame_ten_and_preserves_input():
    """Batch one original with 30 frame edits and identify the influential frame."""
    calls = []

    def predict_proba(batch):
        calls.append(batch.copy())
        probability = 1 / (1 + np.exp(-batch[:, 10].mean(axis=1)))
        result = np.zeros((len(batch), 129), dtype=float)
        result[:, 3] = probability
        result[:, 2] = 1 - probability
        return result

    x = np.zeros((30, 126), dtype=float)
    x[10] = 1
    original = x.copy()
    importance = frame_importance(predict_proba, x, 3)
    assert importance.shape == (30,)
    assert int(np.argmax(importance)) == 10
    assert len(calls) == 1
    assert calls[0].shape == (31, 30, 126)
    np.testing.assert_array_equal(x, original)


def test_joint_group_importance_peaks_at_right_index_and_preserves_input():
    """Batch one original with ten edits and isolate right index motion."""
    calls = []

    def predict_proba(batch):
        calls.append(batch.copy())
        group = batch[:, :, 78:90]
        motion = group.std(axis=1).mean(axis=1)
        probability = 1 / (1 + np.exp(-motion))
        result = np.zeros((len(batch), 129), dtype=float)
        result[:, 3] = probability
        result[:, 2] = 1 - probability
        return result

    x = np.zeros((30, 126), dtype=float)
    x[:, 78:90] = np.linspace(-1, 1, 30)[:, None]
    original = x.copy()
    importance = joint_group_importance(predict_proba, x, 3)
    assert len(importance) == 10
    assert max(importance, key=importance.get) == ("right", "index")
    assert importance[("right", "index")] > 0
    assert len(calls) == 1
    assert calls[0].shape == (11, 30, 126)
    np.testing.assert_array_equal(x, original)


def test_wrong_prediction_report_returns_probabilities_and_frame_shapes():
    """Report the argmax, class probabilities, and both frame-importance arrays."""
    def predict_proba(batch):
        probability = 1 / (1 + np.exp(-batch[:, 10].mean(axis=1)))
        result = np.zeros((len(batch), 129), dtype=float)
        result[:, 3] = probability
        result[:, 2] = 1 - probability
        return result

    x = np.zeros((30, 126), dtype=float)
    x[10] = 1
    original = x.copy()
    report = wrong_prediction_report(predict_proba, x, true_id=2)
    assert report["pred_id"] == 3
    assert report["p_pred"] > report["p_true"]
    assert report["frame_importance_for_pred"].shape == (30,)
    assert report["frame_importance_for_true"].shape == (30,)
    np.testing.assert_array_equal(x, original)
