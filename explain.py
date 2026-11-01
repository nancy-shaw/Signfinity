"""Confusion analysis helpers for sign classifier predictions."""

from collections.abc import Mapping

import numpy as np


def confusion_counts(y_true, y_pred, n_classes=129):
    """Return an integer confusion matrix with true classes in rows."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (y_true, y_pred), 1)
    return counts


def _word(words, class_id):
    """Look up a display word, falling back to the numeric class id."""
    if words is None:
        return class_id
    if isinstance(words, Mapping):
        return words.get(class_id, class_id)
    try:
        return words[class_id]
    except (IndexError, KeyError, TypeError):
        return class_id


def top_confused_pairs(y_true, y_pred, words=None, k=10):
    """Return the most common off-diagonal true-to-predicted class pairs."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 and y_pred.size == 0:
        return []
    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    counts = confusion_counts(y_true, y_pred, n_classes=n_classes)
    pairs = [
        {
            "true_id": true_id,
            "pred_id": pred_id,
            "true_word": _word(words, true_id),
            "pred_word": _word(words, pred_id),
            "count": int(counts[true_id, pred_id]),
        }
        for true_id, pred_id in zip(*np.nonzero(counts * (1 - np.eye(n_classes, dtype=int))))
    ]
    pairs.sort(key=lambda item: (-item["count"], item["true_id"], item["pred_id"]))
    return pairs[: max(0, k)]


def new_errors(y_true, pred_base, pred_stress, words=None):
    """Compare predictions and summarize newly wrong stress transitions."""
    y_true = np.asarray(y_true)
    pred_base = np.asarray(pred_base)
    pred_stress = np.asarray(pred_stress)
    newly_wrong = (pred_base == y_true) & (pred_stress != y_true)
    newly_right = (pred_base != y_true) & (pred_stress == y_true)
    transitions = confusion_counts(
        y_true[newly_wrong], pred_stress[newly_wrong],
        n_classes=int(max(np.max(y_true), np.max(pred_stress))) + 1,
    ) if y_true.size else np.zeros((0, 0), dtype=int)
    top_transitions = [
        {"true_id": int(true_id), "pred_id": int(pred_id), "count": int(transitions[true_id, pred_id])}
        for true_id, pred_id in zip(*np.nonzero(transitions))
    ]
    top_transitions.sort(key=lambda item: (-item["count"], item["true_id"], item["pred_id"]))
    return {
        "n_newly_wrong": int(np.count_nonzero(newly_wrong)),
        "n_newly_right": int(np.count_nonzero(newly_right)),
        "top_transitions": top_transitions,
    }


def top_confused_classes(counts, k=15):
    """Return class ids with the most off-diagonal errors in each row."""
    counts = np.asarray(counts)
    errors = counts.sum(axis=1) - np.diag(counts)
    ids = np.flatnonzero(errors > 0)
    return sorted(ids.tolist(), key=lambda class_id: (-int(errors[class_id]), class_id))[: max(0, k)]


def frame_importance(predict_proba, x, class_id):
    """Measure probability drops when each frame is replaced by its neighbors."""
    x = np.asarray(x)
    modified = np.repeat(x[np.newaxis, :, :], 30, axis=0)
    modified[0, 0] = x[1]
    modified[-1, -1] = x[-2]
    for frame_id in range(1, 29):
        modified[frame_id, frame_id] = (x[frame_id - 1] + x[frame_id + 1]) / 2
    batch = np.concatenate((x[np.newaxis, :, :], modified), axis=0)
    probabilities = np.asarray(predict_proba(batch))
    return (probabilities[0, class_id] - probabilities[1:, class_id]).astype(float)


def joint_group_importance(predict_proba, x, class_id):
    """Measure probability drops after removing each hand finger's motion."""
    x = np.asarray(x)
    finger_joints = {
        "thumb": range(1, 5),
        "index": range(5, 9),
        "middle": range(9, 13),
        "ring": range(13, 17),
        "pinky": range(17, 21),
    }
    groups = []
    modified = []
    for hand, offset in (("left", 0), ("right", 21)):
        for finger, joints in finger_joints.items():
            features = [feature for joint in joints for feature in range(3 * (joint + offset), 3 * (joint + offset) + 3)]
            groups.append((hand, finger, features))
            changed = x.copy()
            changed[:, features] = x[:, features].mean(axis=0)
            modified.append(changed)
    batch = np.concatenate((x[np.newaxis, :, :], np.stack(modified)), axis=0)
    probabilities = np.asarray(predict_proba(batch))
    return {
        (hand, finger): float(probabilities[0, class_id] - probabilities[index + 1, class_id])
        for index, (hand, finger, _) in enumerate(groups)
    }


def wrong_prediction_report(predict_proba, x, true_id):
    """Report the top prediction and frame importance for predicted and true classes."""
    x = np.asarray(x)
    original = np.asarray(predict_proba(x[np.newaxis, :, :]))[0]
    pred_id = int(np.argmax(original))
    return {
        "pred_id": pred_id,
        "p_true": float(original[true_id]),
        "p_pred": float(original[pred_id]),
        "frame_importance_for_pred": frame_importance(predict_proba, x, pred_id),
        "frame_importance_for_true": frame_importance(predict_proba, x, true_id),
    }
