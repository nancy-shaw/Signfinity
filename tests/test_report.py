"""Tests for robustness report generation without external model or data files."""

import json

import numpy as np

import report


class DummyModel:
    """Predict one of two classes from the sign of the first feature."""

    def predict(self, X):
        """Return deterministic class predictions for the provided batch."""
        return (X[:, 0, 0] >= 0).astype(int)


def write_data(tmp_path, X, y):
    """Write arrays to temporary NumPy files and return their paths."""
    x_path = tmp_path / "features.npy"
    y_path = tmp_path / "labels.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    return x_path, y_path


def test_report_calculates_clean_perturbed_and_per_class_metrics(tmp_path, monkeypatch):
    """The report includes metrics for every registered perturbation."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(8, 30, 126)).astype(np.float32)
    y = (X[:, 0, 0] >= 0).astype(int)
    x_path, y_path = write_data(tmp_path, X, y)
    monkeypatch.setattr(report.adapters, "load_model", lambda _path: DummyModel())

    result = report.run_report("unused.pt", x_path, y_path, claimed_accuracy=1.0)

    assert "error" not in result
    assert result["baseline"]["accuracy"] == 1.0
    assert result["baseline_ok"] is True
    assert list(result["perturbations"]) == list(report.TESTS)
    assert set(result["per_class"]) == {str(int(class_id)) for class_id in np.unique(y)}
    assert all("clean_accuracy" in item for item in result["per_class"].values())
    assert all(
        set(item["perturbations"]) == set(report.TESTS)
        for item in result["per_class"].values()
    )
    assert result["robustness_score"] == result["mean_perturbed_accuracy"]
    assert result["worst_perturbation"]["name"] in report.TESTS
    assert result["worst_class"]["class_id"] in result["per_class"]
    json.dumps(result)


def test_claimed_accuracy_mismatch_sets_message(tmp_path, monkeypatch):
    """A disagreement beyond tolerance is clearly marked in the report."""
    rng = np.random.default_rng(21)
    X = rng.normal(size=(4, 30, 126)).astype(np.float32)
    y = (X[:, 0, 0] >= 0).astype(int)
    x_path, y_path = write_data(tmp_path, X, y)
    monkeypatch.setattr(report.adapters, "load_model", lambda _path: DummyModel())

    result = report.run_report("unused.pt", x_path, y_path, claimed_accuracy=0.0)

    assert result["baseline_ok"] is False
    assert result["message"] == "model or data does not match"


def test_bad_feature_shape_returns_error(tmp_path, monkeypatch):
    """Invalid feature dimensions are reported without propagating an error."""
    x_path, y_path = write_data(
        tmp_path,
        np.zeros((3, 12, 5), dtype=np.float32),
        np.zeros(3, dtype=int),
    )
    monkeypatch.setattr(report.adapters, "load_model", lambda _path: DummyModel())

    result = report.run_report("unused.pt", x_path, y_path)

    assert "error" in result
    assert "got (3, 12, 5)" in result["error"]


def test_model_load_failure_returns_error(tmp_path, monkeypatch):
    """An unloadable model becomes an error result."""
    monkeypatch.setattr(
        report.adapters,
        "load_model",
        lambda _path: (_ for _ in ()).throw(ValueError("bad model")),
    )

    result = report.run_report("bad-model.pt", "unused-x.npy", "unused-y.npy")

    assert result == {"error": "bad model"}
