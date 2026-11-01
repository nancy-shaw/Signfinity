"""Serve the Sign Model robustness report interface."""

import os
import shutil
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from report import run_report


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("SIGNFINITY_MAX_CONTENT_LENGTH", 64 * 1024 * 1024)
)

MODEL_EXTENSIONS = {".pt", ".pth", ".onnx"}
DATA_EXTENSION = ".npy"


@app.get("/")
def index():
    """Serve the report upload page."""
    return render_template("index.html")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    """Return upload size failures as JSON."""
    return jsonify({"error": "Upload exceeds the configured size limit."}), 413


@app.post("/analyze")
def analyze():
    """Save validated uploads temporarily, run the report, and clean up."""
    uploads = {
        "model": request.files.get("model"),
        "X_test": request.files.get("X_test"),
        "y_test": request.files.get("y_test"),
    }
    missing = [name for name, upload in uploads.items() if upload is None or not upload.filename]
    if missing:
        return jsonify({"error": f"Missing required file field(s): {', '.join(missing)}."}), 400

    extensions = {name: Path(upload.filename).suffix.lower() for name, upload in uploads.items()}
    if extensions["model"] not in MODEL_EXTENSIONS:
        return jsonify({"error": "Model file must use .pt, .pth, or .onnx."}), 400
    invalid_data = [name for name in ("X_test", "y_test") if extensions[name] != DATA_EXTENSION]
    if invalid_data:
        return jsonify({"error": "X_test and y_test files must use .npy."}), 400

    claimed_accuracy = None
    claimed_value = request.form.get("claimed_accuracy", "").strip()
    if claimed_value:
        try:
            claimed_accuracy = float(claimed_value)
        except ValueError:
            return jsonify({"error": "claimed_accuracy must be a number."}), 400

    temp_dir = tempfile.mkdtemp(prefix="signfinity-report-")
    saved_paths = {}
    try:
        for name, upload in uploads.items():
            descriptor, temp_path = tempfile.mkstemp(
                suffix=extensions[name], dir=temp_dir
            )
            os.close(descriptor)
            saved_paths[name] = temp_path
            upload.save(temp_path)

        result = run_report(
            saved_paths["model"],
            saved_paths["X_test"],
            saved_paths["y_test"],
            claimed_accuracy=claimed_accuracy,
        )
        if "error" in result:
            return jsonify({"error": str(result["error"])}), 422
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    app.run()
