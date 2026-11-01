"""Tests for advisor validation and the report-id based Flask endpoint."""

import json
from types import SimpleNamespace

import advisor
import app as flask_app
import pytest


PERTURBATION_NAMES = [
    "Mirror (left hand)",
    "Slower (0.7x)",
    "Faster (1.3x)",
    "Noise",
    "Scale jitter",
    "Missing frames",
]


def report_data():
    """Build a compact report fixture with the production report schema."""
    baseline_accuracy = 0.8
    perturbations = {
        name: {
            "accuracy": baseline_accuracy - (index + 1) / 100,
            "correct": 79 - index,
            "total": 100,
            "per_class": {},
        }
        for index, name in enumerate(PERTURBATION_NAMES)
    }
    per_class = {
        str(class_id): {
            "n": 3,
            "clean_accuracy": baseline_accuracy,
            "perturbations": {
                name: values["accuracy"] - class_id / 1000
                for name, values in perturbations.items()
            },
        }
        for class_id in range(12)
    }
    return {
        "baseline": {"accuracy": baseline_accuracy, "correct": 80, "total": 100},
        "baseline_ok": True,
        "per_class": per_class,
        "perturbations": perturbations,
        "mean_perturbed_accuracy": 0.765,
        "robustness_score": 0.95625,
        "worst_perturbation": {"name": PERTURBATION_NAMES[0], "accuracy": 0.79},
        "worst_class": {"class_id": "11", "mean_perturbed_accuracy": 0.7845},
    }


def advisor_json():
    """Return valid advisor output with priorities deliberately out of order."""
    return json.dumps(
        {
            "summary": "Address the largest measured weaknesses first.",
            "suggestions": [
                {
                    "title": "Low item",
                    "evidence": "Measured evidence 1.",
                    "what_to_change": "Make a small change.",
                    "expected_effect": "Could help.",
                    "priority": "low",
                },
                {
                    "title": "High item",
                    "evidence": "Measured evidence 2.",
                    "what_to_change": "Review the data.",
                    "expected_effect": "Could improve robustness.",
                    "priority": "high",
                },
                {
                    "title": "Medium item",
                    "evidence": "Measured evidence 3.",
                    "what_to_change": "Add a check.",
                    "expected_effect": "Could reduce sensitivity.",
                    "priority": "medium",
                },
            ],
            "caveats": ["Per-class results are noisy."],
        }
    )


class FakeOpenAI:
    """Provide mocked chat completion responses and capture request details."""

    content = None
    error = None
    calls = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        """Capture a completion request and return a mocked JSON response."""
        type(self).calls.append(kwargs)
        if type(self).error is not None:
            raise type(self).error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=type(self).content))]
        )


def setup_fake(monkeypatch, content=None, error=None):
    """Configure credentials and a predictable mocked OpenAI client."""
    FakeOpenAI.content = content
    FakeOpenAI.error = error
    FakeOpenAI.calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(advisor, "OpenAI", FakeOpenAI)


def test_valid_json_is_parsed_and_prioritized(monkeypatch):
    """Valid model output is schema-checked and sorted by priority."""
    setup_fake(monkeypatch, advisor_json())

    result = advisor.get_suggestions(report_data(), "A small recurrent model")

    assert [item["priority"] for item in result["suggestions"]] == [
        "high",
        "medium",
        "low",
    ]
    assert set(result) == {"summary", "suggestions", "caveats"}
    request = FakeOpenAI.calls[0]
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload) == {"model_description", "report_summary"}
    assert len(payload["report_summary"]["worst_classes"]) == 10
    assert set(payload["report_summary"]["perturbations"]) == set(PERTURBATION_NAMES)
    assert set(payload["report_summary"]) == {
        "baseline_accuracy",
        "mean_perturbed_accuracy",
        "robustness_score",
        "perturbations",
        "worst_perturbation",
        "worst_class",
        "worst_classes",
    }


def test_invalid_json_retries_once_then_returns_error(monkeypatch):
    """Malformed responses are retried once and then returned as an error."""
    setup_fake(monkeypatch, "not json")

    result = advisor.get_suggestions(report_data(), "")

    assert result == {"error": "OpenAI returned invalid advisor JSON"}
    assert len(FakeOpenAI.calls) == 2


def test_missing_key_returns_required_error(monkeypatch):
    """A missing API key exits before constructing a client."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(advisor, "OpenAI", lambda **_kwargs: pytest.fail("client should not be created"))

    assert advisor.get_suggestions(report_data(), "") == {"error": "OPENAI_API_KEY not set"}


def test_api_exception_returns_error(monkeypatch):
    """API failures are caught and returned without raising into the app."""
    setup_fake(monkeypatch, error=RuntimeError("network unavailable"))

    result = advisor.get_suggestions(report_data(), "")

    assert result == {"error": "OpenAI API request failed"}
    assert len(FakeOpenAI.calls) == 1


def test_advise_unknown_report_id_returns_404(monkeypatch):
    """The advisor route rejects ids absent from the server-side store."""
    monkeypatch.setattr(
        flask_app,
        "get_suggestions",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not call advisor")),
    )
    client = flask_app.app.test_client()

    response = client.post(
        "/advise",
        json={"report_id": "not-a-live-report", "model_description": "description"},
    )

    assert response.status_code == 404
    assert response.get_json() == {"error": "Unknown report_id."}
