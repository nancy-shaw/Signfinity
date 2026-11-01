"""Generate evidence-based model improvement suggestions with OpenAI."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

SYSTEM_PROMPT = """You are an ML robustness reviewer for landmark-based sign language classifiers. Use only the numbers provided. Do not invent numbers or results. Every suggestion must cite the specific number(s) that motivate it. Suggestions are for a human engineer to apply; do not claim they were tested. Mention that per-class numbers use about 3 samples and are noisy. Write accuracies as percentages with at most 2 decimals. Return only JSON with this schema: {"summary": str, "suggestions": [{"title": str, "evidence": str, "what_to_change": str, "expected_effect": str, "priority": "high|medium|low"}], "caveats": [str]}. Include 3 to 5 suggestions, ordered by priority."""
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
REQUIRED_SUGGESTION_FIELDS = (
    "title",
    "evidence",
    "what_to_change",
    "expected_effect",
    "priority",
)


def _compact_summary(report):
    """Select only the report metrics needed for improvement advice."""
    baseline_accuracy = report["baseline"]["accuracy"]
    perturbations = {
        name: {
            "accuracy": values["accuracy"],
            "drop_percentage_points": (baseline_accuracy - values["accuracy"]) * 100,
        }
        for name, values in report["perturbations"].items()
    }
    classes = []
    for class_id, values in report["per_class"].items():
        accuracies = list(values["perturbations"].values())
        mean_accuracy = (
            sum(accuracies) / len(accuracies)
            if accuracies
            else values["clean_accuracy"]
        )
        classes.append(
            {
                "class_id": class_id,
                "n": values["n"],
                "clean_accuracy": values["clean_accuracy"],
                "mean_perturbed_accuracy": mean_accuracy,
            }
        )
    classes.sort(key=lambda item: item["mean_perturbed_accuracy"])
    return {
        "baseline_accuracy": baseline_accuracy,
        "mean_perturbed_accuracy": report["mean_perturbed_accuracy"],
        "robustness_score": report["robustness_score"],
        "perturbations": perturbations,
        "worst_perturbation": report["worst_perturbation"],
        "worst_class": report["worst_class"],
        "worst_classes": classes[:10],
    }


def _validated_response(content):
    """Parse advisor JSON and return only fields in the expected schema."""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    if not isinstance(data.get("summary"), str):
        raise ValueError("summary must be a string")
    suggestions = data.get("suggestions")
    if not isinstance(suggestions, list) or not 3 <= len(suggestions) <= 5:
        raise ValueError("suggestions must contain 3 to 5 items")

    checked_suggestions = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            raise ValueError("each suggestion must be an object")
        if any(not isinstance(suggestion.get(field), str) for field in REQUIRED_SUGGESTION_FIELDS):
            raise ValueError("suggestion fields must be strings")
        priority = suggestion["priority"]
        if priority not in PRIORITY_ORDER:
            raise ValueError("suggestion priority is invalid")
        checked_suggestions.append(
            {field: suggestion[field] for field in REQUIRED_SUGGESTION_FIELDS}
        )
    checked_suggestions.sort(key=lambda item: PRIORITY_ORDER[item["priority"]])

    caveats = data.get("caveats")
    if not isinstance(caveats, list) or any(not isinstance(item, str) for item in caveats):
        raise ValueError("caveats must be a list of strings")
    return {
        "summary": data["summary"],
        "suggestions": checked_suggestions,
        "caveats": caveats,
    }


def get_suggestions(report: dict, model_description: str) -> dict:
    """Request and validate practical suggestions for a compact report summary."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}
    model_name = os.getenv("OPENAI_MODEL")
    if not model_name:
        return {"error": "OPENAI_MODEL not set"}

    try:
        summary = _compact_summary(report)
        user_content = json.dumps(
            {
                "model_description": model_description,
                "report_summary": summary,
            },
            separators=(",", ":"),
        )
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=0)
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return _validated_response(content)
            except (json.JSONDecodeError, ValueError, TypeError, KeyError, IndexError, AttributeError):
                if attempt == 1:
                    return {"error": "OpenAI returned invalid advisor JSON"}
        return {"error": "OpenAI returned invalid advisor JSON"}
    except Exception:
        return {"error": "OpenAI API request failed"}
