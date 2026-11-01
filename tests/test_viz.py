"""Tests for Plotly landmark visualizations."""

import numpy as np
import pytest

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
    assert len(figure.layout.annotations) == 2


def test_side_by_side_figure_rejects_wrong_shape():
    """Either comparison sequence with a wrong shape raises ValueError."""
    valid = np.zeros((30, 126))
    with pytest.raises(ValueError):
        side_by_side_figure(np.zeros((31, 126)), valid, ("A", "B"))
    with pytest.raises(ValueError):
        side_by_side_figure(valid, np.zeros((30, 127)), ("A", "B"))
