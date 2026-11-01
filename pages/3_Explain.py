"""Streamlit page for exploring classifier errors and stress effects."""

import importlib
import os
from pathlib import Path

import numpy as np
import streamlit as st

from explain_mock import (
    build_error_heatmap,
    build_error_matrix,
    build_importance_bar,
    fake_predict_proba,
    make_mock_data,
    make_mock_words,
    mock_tests,
)
from viz import side_by_side_figure, skeleton_figure


def _fallback_confusion_counts(y_true, y_pred, n_classes):
    """Build a confusion matrix if the optional explain module is unavailable."""
    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return counts


def _fallback_top_pairs(y_true, y_pred, words=None, k=10):
    """List ranked off-diagonal errors when explain.py is unavailable."""
    counts = _fallback_confusion_counts(y_true, y_pred, len(words))
    pairs = [
        {"true_id": row, "pred_id": col, "true_word": words[row], "pred_word": words[col], "count": int(counts[row, col])}
        for row, col in zip(*np.nonzero(counts * (1 - np.eye(len(words), dtype=int))))
    ]
    return sorted(pairs, key=lambda pair: (-pair["count"], pair["true_id"], pair["pred_id"]))[:k]


def _fallback_new_errors(y_true, pred_base, pred_stress, words=None):
    """Compare baseline and stressed predictions without the optional module."""
    y_true = np.asarray(y_true)
    pred_base = np.asarray(pred_base)
    pred_stress = np.asarray(pred_stress)
    newly_wrong = (pred_base == y_true) & (pred_stress != y_true)
    newly_right = (pred_base != y_true) & (pred_stress == y_true)
    transitions = _fallback_confusion_counts(y_true[newly_wrong], pred_stress[newly_wrong], len(words))
    top = [
        {"true_id": row, "pred_id": col, "count": int(transitions[row, col])}
        for row, col in zip(*np.nonzero(transitions))
    ]
    top.sort(key=lambda item: (-item["count"], item["true_id"], item["pred_id"]))
    return {"n_newly_wrong": int(newly_wrong.sum()), "n_newly_right": int(newly_right.sum()), "top_transitions": top}


def _fallback_top_classes(counts, k=15):
    """Rank classes by their off-diagonal error counts."""
    errors = np.asarray(counts).sum(axis=1) - np.diag(counts)
    ids = np.flatnonzero(errors > 0)
    return sorted(ids.tolist(), key=lambda class_id: (-int(errors[class_id]), class_id))[:k]


try:
    import explain
except Exception:
    explain = None

try:
    from explain import confusion_counts, new_errors, top_confused_classes, top_confused_pairs
except Exception:
    confusion_counts = _fallback_confusion_counts
    new_errors = _fallback_new_errors
    top_confused_classes = _fallback_top_classes
    top_confused_pairs = _fallback_top_pairs

@st.cache_data(show_spinner=False)
def _cached_mock_data(n, n_classes, seed):
    """Cache generated mock samples for repeated page reruns."""
    return make_mock_data(n=n, n_classes=n_classes, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_mock_variant(X, name):
    """Cache a deterministic mock stress transform."""
    return mock_tests()[name](X)


@st.cache_data(show_spinner=False)
def _cached_mock_probabilities(X, n_classes):
    """Cache mock model probabilities for a landmark batch."""
    return fake_predict_proba(X, n_classes=n_classes)


def _real_context(model_path, x_path, y_path):
    """Load real model inputs and a probability function when adapters support them."""
    adapters = importlib.import_module("adapters")
    stress_module = importlib.import_module("stress")
    predict_fn = getattr(adapters, "predict_proba", None)
    if not callable(predict_fn):
        raise RuntimeError("adapters.predict_proba is unavailable")
    if not model_path or not x_path or not y_path:
        raise RuntimeError("Model and data paths are required")
    if not all(Path(path).is_file() for path in (model_path, x_path, y_path)):
        raise RuntimeError("One or more configured files do not exist")
    model = adapters.load_model(model_path)
    X, y = adapters.load_data(x_path, y_path)

    def probability_function(batch):
        """Run the configured adapter probability function."""
        try:
            return np.asarray(predict_fn(model, batch))
        except TypeError:
            return np.asarray(predict_fn(batch))

    return X, y, probability_function, stress_module.TESTS


def _probability_matrix(predict_fn, X, n_classes):
    """Validate model output probabilities for one landmark batch."""
    probabilities = np.asarray(predict_fn(X))
    if probabilities.shape != (len(X), n_classes):
        raise ValueError(f"Expected probability shape {(len(X), n_classes)}, got {probabilities.shape}")
    return probabilities


def _word(words, class_id):
    """Return a class display word with a numeric fallback."""
    return words[class_id] if 0 <= int(class_id) < len(words) else str(class_id)


st.set_page_config(page_title="Explain model errors", layout="wide")
st.title("Explain: why the model fails")
if "explain_seed" not in st.session_state:
    st.session_state.explain_seed = 0
use_mock = st.sidebar.checkbox("Use mock data", value=True, key="explain_use_mock")

real_paths = ("", "", "")
if not use_mock:
    real_paths = (
        st.sidebar.text_input("Model path", value=os.environ.get("SIGNFINITY_MODEL_PATH", os.environ.get("MODEL_PATH", ""))),
        st.sidebar.text_input("Feature data path", value=os.environ.get("SIGNFINITY_X_PATH", os.environ.get("X_PATH", ""))),
        st.sidebar.text_input("Label data path", value=os.environ.get("SIGNFINITY_Y_PATH", os.environ.get("Y_PATH", ""))),
    )

context = None
with st.spinner("Preparing explanation data..."):
    if use_mock:
        try:
            X, y = _cached_mock_data(60, 10, st.session_state.explain_seed)
            context = {
                "X": X,
                "y": y,
                "words": make_mock_words(10),
                "predict_fn": lambda batch: _cached_mock_probabilities(batch, 10),
                "tests": mock_tests(),
                "is_mock": True,
            }
        except Exception as error:
            st.error(f"Could not prepare mock data: {error}")
    else:
        try:
            X, y, predict_fn, transforms = _real_context(*real_paths)
            context = {
                "X": X,
                "y": y,
                "words": make_mock_words(int(np.max(y)) + 1) if len(y) else [],
                "predict_fn": predict_fn,
                "tests": transforms,
                "is_mock": False,
            }
        except Exception as error:
            st.info("Real mode is not ready yet, using mock data")
            st.caption(str(error))
            try:
                X, y = _cached_mock_data(60, 10, st.session_state.explain_seed)
                context = {
                    "X": X,
                    "y": y,
                    "words": make_mock_words(10),
                    "predict_fn": lambda batch: _cached_mock_probabilities(batch, 10),
                    "tests": mock_tests(),
                    "is_mock": True,
                }
            except Exception as mock_error:
                st.error(f"Could not prepare fallback mock data: {mock_error}")

if context is None:
    st.stop()

X = context["X"]
y = np.asarray(context["y"], dtype=int)
words = context["words"]
predict_fn = context["predict_fn"]
test_functions = context["tests"]
n_classes = len(words)

try:
    with st.spinner("Evaluating original and stressed samples..."):
        base_probabilities = _probability_matrix(predict_fn, X, n_classes)
        variants = {"Original": X}
        for test_name, transform in test_functions.items():
            variants[test_name] = (
                _cached_mock_variant(X, test_name) if context["is_mock"] else transform(X)
            )
        probability_results = {"Original": base_probabilities}
        for test_name, variant in variants.items():
            if test_name != "Original":
                probability_results[test_name] = _probability_matrix(predict_fn, variant, n_classes)
        predictions = {name: probabilities.argmax(axis=1) for name, probabilities in probability_results.items()}
        st.session_state["explain_context"] = {
            "X": X,
            "y": y,
            "variants": variants,
            "predictions": predictions,
            "probabilities": probability_results,
            "words": words,
            "predict_fn": predict_fn,
        }
        if not context["is_mock"]:
            accuracy = float(np.mean(predictions["Original"] == y)) if len(y) else 0.0
            if abs(accuracy - 0.8695) <= 0.005:
                st.success(f"Baseline accuracy: {accuracy:.2%} (expected about 86.95%)")
            else:
                st.warning(f"Baseline accuracy: {accuracy:.2%}; expected about 86.95%")
except Exception as error:
    st.error(f"Could not evaluate model data: {error}")
    st.stop()

available_tests = ["Original", *test_functions.keys()]
st.header("1. What breaks")
try:
    selected_test = st.selectbox("Stress test", available_tests, key="explain_selected_test")
except Exception as error:
    st.error(f"Could not select a stress test: {error}")
    st.stop()

try:
    with st.spinner("Analyzing confusion..."):
        selected_predictions = predictions[selected_test]
        counts = confusion_counts(y, selected_predictions, n_classes)
        pairs = top_confused_pairs(y, selected_predictions, words=words, k=10)
        if not pairs:
            st.success("No errors for this test.")
        else:
            st.dataframe(pairs, hide_index=True, width="stretch")
            labels, matrix = build_error_matrix(counts, words, max_classes=15)
            if labels:
                st.plotly_chart(build_error_heatmap(matrix, labels), width="stretch")
                st.caption("Shows only mistakes between the most confused signs. Each class has about 3 test samples, so counts are small.")
except Exception as error:
    st.error(f"What breaks analysis failed: {error}")

try:
    st.header("2. What changed under stress")
    with st.spinner("Comparing changes under stress..."):
        if selected_test == "Original":
            st.info("Pick a stress test above to see what changes.")
        else:
            delta = new_errors(y, predictions["Original"], predictions[selected_test], words=words)
            left, right = st.columns(2)
            left.metric("Newly wrong", delta["n_newly_wrong"])
            right.metric("Newly right", delta["n_newly_right"])
            transitions = [
                {
                    "true_word": _word(words, item["true_id"]),
                    "pred_word": _word(words, item["pred_id"]),
                    "count": item["count"],
                }
                for item in delta["top_transitions"]
            ]
            if transitions:
                st.dataframe(transitions, hide_index=True, width="stretch")
            else:
                st.info("No prediction transitions for this comparison.")
except Exception as error:
    st.error(f"Stress comparison failed: {error}")

try:
    st.header("3. Inspect one sample")
    with st.spinner("Preparing sample inspection..."):
        wrong_indices = np.flatnonzero(predictions[selected_test] != y)
        if not len(wrong_indices):
            st.info("There are no wrong samples under this test.")
            sample_id = None
        else:
            labels = {
                int(sample_id): f"{sample_id}: {_word(words, y[sample_id])} -> {_word(words, predictions[selected_test][sample_id])}"
                for sample_id in wrong_indices
            }
            sample_id = st.selectbox(
                "Wrong sample",
                wrong_indices.tolist(),
                format_func=lambda value: labels[int(value)],
                key="explain_sample_id",
            )
except Exception as error:
    st.error(f"Sample inspection failed: {error}")
else:
    if sample_id is not None:
        try:
            if selected_test == "Original":
                replay = skeleton_figure(X[sample_id], "Original sequence")
            else:
                replay = side_by_side_figure(X[sample_id], variants[selected_test][sample_id], ("Original", selected_test))
            replay.update_layout(height=480)
            if replay.layout.sliders:
                replay.layout.sliders[0].bgcolor = "#e5e7eb"
                replay.layout.sliders[0].bordercolor = "#cbd5e1"
                replay.layout.sliders[0].borderwidth = 0
                replay.layout.sliders[0].font = {"size": 9, "color": "#64748b"}
                replay.layout.sliders[0].tickcolor = "#94a3b8"
                replay.layout.sliders[0].ticklen = 2
                replay.layout.sliders[0].tickwidth = 1
            st.plotly_chart(replay, width="stretch")
        except Exception as error:
            st.error(f"Skeleton replay failed: {error}")

        if explain is not None and callable(getattr(explain, "wrong_prediction_report", None)):
            try:
                report = explain.wrong_prediction_report(predict_fn, variants[selected_test][sample_id], int(y[sample_id]))
                st.caption(f"Predicted {_word(words, report['pred_id'])}: {report['p_pred']:.3f}; true-class probability: {report['p_true']:.3f}")
            except Exception as error:
                st.error(f"Prediction report failed: {error}")

        has_importance = (
            explain is not None
            and callable(getattr(explain, "frame_importance", None))
            and callable(getattr(explain, "joint_group_importance", None))
        )
        if not has_importance:
            st.warning("Importance functions missing: merge branch f3/explain")
        else:
            pred_id = int(predictions[selected_test][sample_id])
            stressed_sample = variants[selected_test][sample_id]
            frame_values = None
            group_values = None
            try:
                frame_values = np.asarray(explain.frame_importance(predict_fn, stressed_sample, pred_id), dtype=float)
            except Exception as error:
                st.error(f"Frame importance failed: {error}")
            try:
                group_values = explain.joint_group_importance(predict_fn, stressed_sample, pred_id)
            except Exception as error:
                st.error(f"Finger-group importance failed: {error}")
            group_array = np.asarray(list(group_values.values()), dtype=float) if group_values is not None else None
            frame_zero = frame_values is not None and (not frame_values.size or np.nanmax(np.abs(frame_values)) < 1e-6)
            group_zero = group_array is not None and (not group_array.size or np.nanmax(np.abs(group_array)) < 1e-6)
            if frame_zero and group_zero:
                st.info("This sample shows no measurable importance.")
            else:
                if frame_values is not None and not frame_zero:
                    try:
                        frame_chart = build_importance_bar(frame_values, list(range(30)), "Frame importance for predicted class", "Frame")
                        st.plotly_chart(frame_chart, width="stretch")
                    except Exception as error:
                        st.error(f"Frame importance chart failed: {error}")
                if group_values is not None and group_array is not None and not group_zero:
                    try:
                        group_names = [f"{hand.title()} {finger}" for hand, finger in group_values]
                        group_chart = build_importance_bar(group_array, group_names, "Finger-group importance for predicted class", "Finger group")
                        st.plotly_chart(group_chart, width="stretch")
                    except Exception as error:
                        st.error(f"Finger-group importance chart failed: {error}")

st.info(
    "Importance means the model's probability changes when a frame or finger group is neutralized. "
    "It is a hint, not proof of cause. The data has hand landmarks only, so this is a skeleton replay, "
    "not video. Each hand is drawn relative to its own wrist."
)
