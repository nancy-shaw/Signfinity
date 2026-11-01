"""Flask route tests using temporary random arrays and a dummy predictor."""

from io import BytesIO
import os

import numpy as np

import adapters
import app as flask_app
import report


class DummyModel:
    """Predict one of two classes using the first landmark value."""

    def predict(self, X):
        """Return deterministic predictions for each sequence."""
        return (X[:, 0, 0] >= 0).astype(int)


def npy_upload(values):
    """Encode an array as an in-memory NumPy upload."""
    content = BytesIO()
    np.save(content, values)
    content.seek(0)
    return content


def test_home_and_feature_pages_render():
    """Home and all three feature pages render successfully."""
    client = flask_app.app.test_client()

    assert client.get("/").status_code == 200
    for route in ("/robustness", "/advisor", "/landmarks"):
        assert client.get(route).status_code == 200


def test_robustness_page_renders_upload_form():
    """The robustness page serves the existing report upload form."""
    response = flask_app.app.test_client().get("/robustness")

    assert response.status_code == 200
    assert b"Sign Model Report Card" in response.data
    assert b"name=\"X_test\"" in response.data


def test_analyze_uploads_and_cleans_temporary_files(tmp_path, monkeypatch):
    """Valid uploads run the report and temporary files are removed afterward."""
    rng = np.random.default_rng(92)
    X = rng.normal(size=(18, 30, 126)).astype(np.float32)
    y = (X[:, 0, 0] >= 0).astype(int)
    monkeypatch.setattr(adapters, "load_model", lambda _path: DummyModel())
    original_run_report = report.run_report
    captured_paths = []

    def check_paths_then_run(model_path, x_path, y_path, claimed_accuracy=None):
        """Confirm uploaded paths exist during analysis, then run the report."""
        captured_paths.extend([model_path, x_path, y_path])
        assert all(os.path.isfile(path) for path in captured_paths)
        return original_run_report(
            model_path, x_path, y_path, claimed_accuracy=claimed_accuracy
        )

    monkeypatch.setattr(flask_app, "run_report", check_paths_then_run)
    flask_app.app.config["TESTING"] = True
    client = flask_app.app.test_client()

    response = client.post(
        "/analyze",
        data={
            "model": (BytesIO(b"dummy model"), "classifier.pt"),
            "X_test": (npy_upload(X), "features.npy"),
            "y_test": (npy_upload(y), "labels.npy"),
            "claimed_accuracy": "1.0",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    result = response.get_json()
    assert result["baseline"]["accuracy"] == 1.0
    assert result["baseline_ok"] is True
    assert set(result["perturbations"]) == set(report.TESTS)
    assert result["report_id"]
    assert len(captured_paths) == 3
    assert not any(os.path.exists(path) for path in captured_paths)

    card_response = client.get(f"/card/{result['report_id']}")
    assert card_response.status_code == 200
    assert card_response.headers["Content-Disposition"].endswith(
        'filename=robustness_card.html'
    )
    assert f"{result['robustness_score'] * 100:.2f}%".encode() in card_response.data
    assert b"@media print" in card_response.data


def test_unknown_card_id_returns_404():
    """Unknown in-memory report ids return a clear not-found response."""
    client = flask_app.app.test_client()

    response = client.get("/card/not-a-stored-report")

    assert response.status_code == 404
    assert b"Robustness card not found" in response.data


def test_analyze_rejects_unapproved_extensions():
    """Only the specified model and NumPy file extensions are accepted."""
    client = flask_app.app.test_client()

    response = client.post(
        "/analyze",
        data={
            "model": (BytesIO(b"source"), "unsafe.py"),
            "X_test": (npy_upload(np.zeros((1, 30, 126))), "features.npy"),
            "y_test": (npy_upload(np.zeros(1)), "labels.npy"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_returns_json_error_from_report(monkeypatch):
    """Report failures are returned as a JSON error response."""
    monkeypatch.setattr(flask_app, "run_report", lambda *_args, **_kwargs: {"error": "bad model"})
    client = flask_app.app.test_client()

    response = client.post(
        "/analyze",
        data={
            "model": (BytesIO(b"model"), "classifier.pt"),
            "X_test": (npy_upload(np.zeros((1, 30, 126))), "features.npy"),
            "y_test": (npy_upload(np.zeros(1)), "labels.npy"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 422
    assert response.get_json() == {"error": "bad model"}
