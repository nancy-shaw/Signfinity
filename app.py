"""Serve the Sign Model robustness report interface."""

import os
import secrets
import shutil
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from advisor import get_suggestions
from report import run_report


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("SIGNFINITY_MAX_CONTENT_LENGTH", 64 * 1024 * 1024)
)

MODEL_EXTENSIONS = {".pt", ".pth", ".onnx"}
DATA_EXTENSION = ".npy"
CARD_RESULTS = OrderedDict()
CARD_RESULTS_LOCK = threading.Lock()
CARD_RESULT_LIMIT = 20


def _percent(value):
    """Format a result accuracy or score as a percentage."""
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def _render_card(report, metadata):
    """Build a self-contained, print-ready HTML card from stored report data."""
    baseline = report["baseline"]
    baseline_accuracy = baseline["accuracy"]
    perturbations = sorted(
        report["perturbations"].items(), key=lambda item: item[1]["accuracy"]
    )
    perturbation_rows = []
    for name, values in perturbations:
        drop = (baseline_accuracy - values["accuracy"]) * 100
        perturbation_rows.append(
            "<tr>"
            f"<td>{escape(str(name))}</td>"
            f"<td>{_percent(values['accuracy'])}</td>"
            f"<td>{values['correct']} / {values['total']}</td>"
            f"<td>{drop:+.2f} pp</td>"
            "</tr>"
        )

    classes = []
    for class_id, values in report["per_class"].items():
        perturbation_accuracies = list(values["perturbations"].values())
        mean_accuracy = (
            sum(perturbation_accuracies) / len(perturbation_accuracies)
            if perturbation_accuracies
            else values["clean_accuracy"]
        )
        classes.append((mean_accuracy, str(class_id), values))
    classes.sort(key=lambda item: item[0])
    class_rows = []
    for mean_accuracy, class_id, values in classes[:10]:
        class_rows.append(
            "<tr>"
            f"<td>{escape(class_id)}</td>"
            f"<td>{values['n']}</td>"
            f"<td>{_percent(values['clean_accuracy'])}</td>"
            f"<td>{_percent(mean_accuracy)}</td>"
            "</tr>"
        )

    baseline_status = "Passed" if report["baseline_ok"] else "Did not match"
    status_detail = report.get("message", "")
    if not status_detail and "matches" not in baseline:
        status_detail = "No claimed accuracy was supplied for comparison."
    elif not status_detail and baseline.get("matches") is True:
        status_detail = "The measured accuracy is within the claimed tolerance."
    average_class_count = (
        sum(values["n"] for values in report["per_class"].values())
        / len(report["per_class"])
        if report["per_class"]
        else None
    )
    sample_note = (
        f"Per-class numbers use about {average_class_count:.1f} samples per class, so they are noisy."
        if average_class_count is not None
        else "Per-class sample counts are unavailable, so per-class results are noisy."
    )
    perturbation_count = len(report["perturbations"])
    worst_perturbation = report["worst_perturbation"]
    worst_class = report["worst_class"]
    worst_class_summary = (
        f"Class {escape(str(worst_class['class_id']))} "
        f"({_percent(worst_class['mean_perturbed_accuracy'])})"
        if worst_class is not None
        else "N/A"
    )
    model_filename = escape(metadata["model_filename"])
    generated_at = escape(metadata["generated_at"])
    message = escape(str(status_detail))

    improvement_section = ""
    suggestions = metadata.get("suggestions")
    if suggestions:
        suggestion_rows = []
        for suggestion in suggestions["suggestions"]:
            suggestion_rows.append(
                "<article class=\"suggestion\">"
                f"<p><strong>{escape(suggestion['priority'].title())}</strong></p>"
                f"<h3>{escape(suggestion['title'])}</h3>"
                f"<p><strong>Evidence:</strong> {escape(suggestion['evidence'])}</p>"
                f"<p><strong>What to change:</strong> {escape(suggestion['what_to_change'])}</p>"
                f"<p><strong>Expected effect:</strong> {escape(suggestion['expected_effect'])}</p>"
                "</article>"
            )
        caveat_list = "".join(f"<li>{escape(caveat)}</li>" for caveat in suggestions["caveats"])
        improvement_section = (
            "<h2>Improvement suggestions</h2>"
            f"<p>{escape(suggestions['summary'])}</p>"
            f"{''.join(suggestion_rows)}"
            f"<ul>{caveat_list}</ul>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign Model Robustness Card</title>
<style>
:root {{ color-scheme: light; font-family: Arial, Helvetica, sans-serif; color: #172522; background: #f4f7f5; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 36px 18px; }}
main {{ max-width: 960px; margin: 0 auto; padding: 36px; background: #fff; border: 1px solid #dce5e0; border-radius: 16px; }}
h1 {{ margin: 0; font-size: 2rem; }}
h2 {{ margin: 30px 0 12px; font-size: 1.2rem; }}
.meta, .muted {{ color: #586963; }}
.meta {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 20px; margin: 18px 0 24px; }}
.status {{ padding: 14px 16px; border-radius: 9px; background: {'#e7f5ee' if report['baseline_ok'] else '#fff0ed'}; color: {'#155c41' if report['baseline_ok'] else '#842d2d'}; }}
.metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.metric {{ padding: 15px; border: 1px solid #dce5e0; border-radius: 10px; }}
.metric span {{ display: block; margin-bottom: 7px; color: #586963; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
.metric strong {{ font-size: 1.25rem; overflow-wrap: anywhere; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 10px 11px; border-bottom: 1px solid #e7eeea; text-align: left; }}
th {{ color: #586963; font-size: .78rem; text-transform: uppercase; letter-spacing: .05em; }}
.note {{ padding: 14px 16px; border-left: 4px solid #39866a; background: #f1f6f3; line-height: 1.55; }}
.suggestion {{ margin: 14px 0; padding: 14px 16px; border: 1px solid #dce5e0; border-radius: 10px; break-inside: avoid; }}
.suggestion h3 {{ margin: 6px 0; }}
@media (max-width: 650px) {{ main {{ padding: 22px; }} .metrics {{ grid-template-columns: 1fr 1fr; }} .meta {{ grid-template-columns: 1fr; }} }}
@media print {{
  @page {{ margin: 14mm; }}
  :root, body {{ background: #fff; }}
  body {{ padding: 0; }}
  main {{ max-width: none; padding: 0; border: 0; border-radius: 0; }}
  .metric, .status, .note {{ break-inside: avoid; print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
  h2, table {{ break-inside: avoid; }}
}}
</style>
</head>
<body><main>
<header><h1>Sign Model Robustness Card</h1>
<div class="meta"><span>Generated: {generated_at}</span><span>Model: {model_filename}</span>
<span>Test samples: {baseline['total']}</span></div></header>
<p class="status"><strong>Baseline status: {baseline_status}</strong><br>{message}</p>
<section class="metrics" aria-label="Summary">
<div class="metric"><span>Baseline accuracy</span><strong>{_percent(baseline_accuracy)}</strong><div>{baseline['correct']} / {baseline['total']} correct</div></div>
<div class="metric"><span>Mean perturbed accuracy</span><strong>{_percent(report['mean_perturbed_accuracy'])}</strong></div>
<div class="metric"><span>Robustness score</span><strong>{_percent(report['robustness_score'])}</strong></div>
<div class="metric"><span>Worst perturbation</span><strong>{escape(str(worst_perturbation['name'])) if worst_perturbation else 'N/A'}</strong><div>{_percent(worst_perturbation['accuracy']) if worst_perturbation else 'N/A'}</div></div>
<div class="metric"><span>Worst class</span><strong>{worst_class_summary}</strong></div>
</section>
<h2>Accuracy by perturbation</h2>
<table><thead><tr><th>Perturbation</th><th>Accuracy</th><th>Correct</th><th>Drop vs baseline</th></tr></thead>
<tbody>{''.join(perturbation_rows)}</tbody></table>
<h2>10 worst classes</h2>
<table><thead><tr><th>Class</th><th>n</th><th>Clean accuracy</th><th>Mean perturbed accuracy</th></tr></thead>
<tbody>{''.join(class_rows)}</tbody></table>
<h2>How to read this</h2>
<p class="note">Higher accuracy means more samples were classified correctly. Compare each perturbation with the clean baseline; a larger drop means greater sensitivity. Per-class results are less reliable when a sign has few examples.</p>
{improvement_section}
<h2>Limitations</h2>
<p>{sample_note} Only {perturbation_count} perturbations are tested. Results apply only to the uploaded model and data.</p>
</main></body></html>"""


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
        model_filename = uploads["model"].filename.replace("\\", "/").rsplit("/", 1)[-1]
        report_id = secrets.token_urlsafe(18)
        stored_report = {
            "result": result,
            "model_filename": model_filename,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with CARD_RESULTS_LOCK:
            while report_id in CARD_RESULTS:
                report_id = secrets.token_urlsafe(18)
            CARD_RESULTS[report_id] = stored_report
            while len(CARD_RESULTS) > CARD_RESULT_LIMIT:
                CARD_RESULTS.popitem(last=False)
        response_data = dict(result)
        response_data["report_id"] = report_id
        return jsonify(response_data)
    except Exception as error:
        return jsonify({"error": str(error)}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/card/<report_id>")
def download_card(report_id):
    """Download a card for a report held in the server-side result store."""
    with CARD_RESULTS_LOCK:
        stored_report = CARD_RESULTS.get(report_id)
    if stored_report is None:
        return Response("Robustness card not found. Run an analysis to create a new card.", status=404, mimetype="text/plain")
    card_html = _render_card(stored_report["result"], stored_report)
    return send_file(
        BytesIO(card_html.encode("utf-8")),
        mimetype="text/html",
        as_attachment=True,
        download_name="robustness_card.html",
    )


@app.post("/advise")
def advise():
    """Generate suggestions from a server-stored report, not browser metrics."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    report_id = data.get("report_id")
    model_description = data.get("model_description", "")
    if not isinstance(report_id, str):
        return jsonify({"error": "report_id must be a string."}), 400
    if not isinstance(model_description, str):
        return jsonify({"error": "model_description must be a string."}), 400
    if len(model_description) > 3000:
        return jsonify({"error": "model_description must be 3000 characters or fewer."}), 400

    with CARD_RESULTS_LOCK:
        stored_report = CARD_RESULTS.get(report_id)
        result = stored_report["result"] if stored_report is not None else None
    if result is None:
        return jsonify({"error": "Unknown report_id."}), 404

    suggestions = get_suggestions(result, model_description)
    if "error" in suggestions:
        return jsonify({"error": suggestions["error"]}), 502
    with CARD_RESULTS_LOCK:
        current_report = CARD_RESULTS.get(report_id)
        if current_report is None:
            return jsonify({"error": "Unknown report_id."}), 404
        current_report["suggestions"] = suggestions
    return jsonify(suggestions)


if __name__ == "__main__":
    app.run()
