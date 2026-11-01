"""Tests for Plotly landmark visualizations."""

import numpy as np
import pytest

from explain_mock import make_mock_data
from viz import side_by_side_figure, skeleton_figure


def test_skeleton_figure_has_thirty_animation_frames():
    """A sequence produces one animation frame per input frame."""
    x = np.zeros((30, 126), dtype=float)
    x[:, :63] = 1
    original = x.copy()
    figure = skeleton_figure(x, "Hand replay")
    assert len(figure.frames) == 30
    assert figure.layout.sliders
    assert {button.label for menu in figure.layout.updatemenus for button in menu.buttons} == {"Play", "Pause"}
    np.testing.assert_array_equal(x, original)


def test_all_zero_hand_is_not_drawn_in_any_frame():
    """Missing-hand trace coordinates stay empty through the whole animation."""
    x = np.zeros((30, 126), dtype=float)
    x[:, :63] = 1
    figure = skeleton_figure(x, "Left hand only")
    assert list(figure.data[2].x) == []
    assert list(figure.data[3].x) == []
    for frame in figure.frames:
        assert list(frame.data[2].x) == []
        assert list(frame.data[3].x) == []


@pytest.mark.parametrize("shape", [(29, 126), (30, 125)])
def test_skeleton_figure_rejects_wrong_shape(shape):
    """Sequences with anything other than 30 by 126 values are rejected."""
    with pytest.raises(ValueError):
        skeleton_figure(np.zeros(shape), "Invalid")


def test_side_by_side_figure_has_one_thirty_frame_animation():
    """The comparison uses one figure and one synchronized frame timeline."""
    original = np.zeros((30, 126), dtype=float)
    stressed = np.zeros((30, 126), dtype=float)
    original[:, :63] = 1
    stressed[:, 63:] = 2
    figure = side_by_side_figure(original, stressed, ("Original", "Stressed"))
    assert len(figure.frames) == 30
    assert len(figure.data) == 8
    labels = {annotation.text for annotation in figure.layout.annotations}
    assert {"Original", "Stressed", "Left", "Right"}.issubset(labels)


def test_side_by_side_figure_rejects_wrong_shape():
    """Either comparison sequence with a wrong shape raises ValueError."""
    valid = np.zeros((30, 126))
    with pytest.raises(ValueError):
        side_by_side_figure(np.zeros((31, 126)), valid, ("A", "B"))
    with pytest.raises(ValueError):
        side_by_side_figure(valid, np.zeros((30, 127)), ("A", "B"))


def test_side_by_side_drawn_points_fit_ranges_and_starts_at_frame_zero():
    """All animation coordinates fit stable ranges and initial traces show frame 0."""
    rng = np.random.default_rng(12)
    original = rng.normal(0, 0.05, size=(30, 126))
    stressed = rng.normal(0, 0.06, size=(30, 126))
    original_copy = original.copy()
    stressed_copy = stressed.copy()
    figure = side_by_side_figure(original, stressed, ("Original", "Stressed"))
    x_min, x_max = figure.layout.xaxis.range
    y_first, y_second = figure.layout.yaxis.range
    y_min, y_max = min(y_first, y_second), max(y_first, y_second)

    assert figure.layout.sliders[0].active == 0
    assert any(len(trace.x) > 0 and len(trace.y) > 0 for trace in figure.data)
    for frame in figure.frames:
        for trace in frame.data:
            for x_value, y_value in zip(trace.x, trace.y):
                if x_value is not None and y_value is not None:
                    assert x_min <= x_value <= x_max
                    assert y_min <= y_value <= y_max
    np.testing.assert_array_equal(original, original_copy)
    np.testing.assert_array_equal(stressed, stressed_copy)


def test_every_animation_frame_has_same_trace_count_as_initial_data():
    """Empty-hand traces remain present so frame updates keep a stable mapping."""
    x = np.random.default_rng(13).normal(0, 0.05, size=(30, 126))
    figure = side_by_side_figure(x, x.copy(), ("Original", "Stressed"))
    assert all(len(frame.data) == len(figure.data) for frame in figure.frames)


def test_figure_hides_legend_ticks_and_axes_titles():
    """Artificial offsets are not presented as physical axes or legend entries."""
    x = np.random.default_rng(14).normal(0, 0.05, size=(30, 126))
    figure = skeleton_figure(x, "Replay")
    assert figure.layout.showlegend is False
    assert figure.layout.xaxis.showticklabels is False
    assert figure.layout.yaxis.showticklabels is False
    assert figure.layout.xaxis.title.text in (None, "")
    assert figure.layout.yaxis.title.text in (None, "")
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.yaxis.showgrid is False
    assert all(not trace.name for trace in figure.data)


def test_figure_keeps_inverted_equal_aspect_and_hand_colors():
    """The y axis is inverted and the two hands use distinct colors."""
    x = np.random.default_rng(15).normal(0, 0.05, size=(30, 126))
    figure = skeleton_figure(x, "Replay")
    assert figure.layout.yaxis.range[0] > figure.layout.yaxis.range[1]
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.scaleratio == 1
    left_color = figure.data[0].line.color
    right_color = figure.data[2].line.color
    assert left_color != right_color


def test_both_public_functions_leave_inputs_unchanged():
    """Figure creation does not mutate either source sequence."""
    original = np.random.default_rng(16).normal(0, 0.05, size=(30, 126))
    stressed = np.random.default_rng(17).normal(0, 0.05, size=(30, 126))
    original_copy = original.copy()
    stressed_copy = stressed.copy()
    skeleton_figure(original, "One")
    side_by_side_figure(original, stressed, ("Original", "Stressed"))
    np.testing.assert_array_equal(original, original_copy)
    np.testing.assert_array_equal(stressed, stressed_copy)


def test_realistic_replay_figures_serialize_to_dict_and_json():
    """Hand-shaped replay figures serialize without invalid Plotly properties."""
    samples, _ = make_mock_data(n=2, n_classes=3, seed=19)
    skeleton = skeleton_figure(samples[0], "Single sample")
    comparison = side_by_side_figure(samples[0], samples[1], ("Original", "Stressed"))
    for figure in (skeleton, comparison):
        assert figure.to_dict()
        assert figure.to_json()
