"""Run a clean and perturbed robustness report for a sign model."""

import numpy as np

import adapters
from stress import TESTS


def _accuracy_by_class(predictions, labels, classes):
    """Calculate per-class counts and accuracies from predictions."""
    result = {}
    for class_id in classes:
        mask = labels == class_id
        count = int(np.count_nonzero(mask))
        correct = int(np.count_nonzero(predictions[mask] == labels[mask]))
        result[str(int(class_id))] = {
            "n": count,
            "correct": correct,
            "accuracy": correct / count if count else 0.0,
        }
    return result


def run_report(model_path, x_path, y_path, claimed_accuracy=None, tolerance=0.01):
    """Load inputs and return JSON-serializable clean and stress results."""
    try:
        model = adapters.load_model(model_path)
        X, y = adapters.load_data(x_path, y_path)
        y = np.asarray(y)
        baseline_predictions = np.asarray(model.predict(X))
        baseline = adapters.check_baseline(
            lambda _X: baseline_predictions,
            X,
            y,
            expected=claimed_accuracy,
            tolerance=tolerance,
        )
        baseline_ok = baseline.get("matches", True)
        classes = np.unique(y)
        clean_per_class = _accuracy_by_class(baseline_predictions, y, classes)

        perturbations = {}
        per_class = {
            class_id: {
                "n": values["n"],
                "clean_accuracy": values["accuracy"],
                "perturbations": {},
            }
            for class_id, values in clean_per_class.items()
        }

        for name, transform in TESTS.items():
            transformed_X = transform(X)
            predictions = np.asarray(model.predict(transformed_X))
            measured = adapters.check_baseline(lambda _X: predictions, transformed_X, y)
            class_results = _accuracy_by_class(predictions, y, classes)
            perturbations[name] = {
                "accuracy": measured["accuracy"],
                "correct": measured["correct"],
                "total": measured["total"],
                "per_class": class_results,
            }
            for class_id, values in class_results.items():
                per_class[class_id]["perturbations"][name] = values["accuracy"]

        perturbed_accuracies = [item["accuracy"] for item in perturbations.values()]
        mean_perturbed_accuracy = (
            float(np.mean(perturbed_accuracies)) if perturbed_accuracies else 0.0
        )
        robustness_score = (
            mean_perturbed_accuracy / baseline["accuracy"]
            if baseline["accuracy"]
            else None
        )
        worst_perturbation = (
            min(perturbations, key=lambda name: perturbations[name]["accuracy"])
            if perturbations
            else None
        )

        class_robustness = {
            class_id: (
                float(np.mean(list(values["perturbations"].values())))
                if values["perturbations"]
                else values["clean_accuracy"]
            )
            for class_id, values in per_class.items()
        }
        worst_class = min(class_robustness, key=class_robustness.get) if class_robustness else None

        result = {
            "baseline": baseline,
            "baseline_ok": bool(baseline_ok),
            "per_class": per_class,
            "perturbations": perturbations,
            "mean_perturbed_accuracy": mean_perturbed_accuracy,
            "robustness_score": robustness_score,
            "worst_perturbation": (
                {
                    "name": worst_perturbation,
                    "accuracy": perturbations[worst_perturbation]["accuracy"],
                }
                if worst_perturbation is not None
                else None
            ),
            "worst_class": (
                {
                    "class_id": worst_class,
                    "mean_perturbed_accuracy": class_robustness[worst_class],
                }
                if worst_class is not None
                else None
            ),
        }
        if not baseline_ok:
            result["message"] = "model or data does not match"
        return result
    except Exception as error:
        return {"error": str(error)}
