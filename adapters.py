"""Load sign classification models and evaluation data."""

from pathlib import Path

import numpy as np
import torch


class TorchPredictor:
    """Run a PyTorch model on CPU in batches and return predicted class ids."""

    def __init__(self, model, batch_size=64):
        self.model = model.to("cpu").eval()
        self.batch_size = batch_size

    def predict(self, X):
        """Predict class ids for a batch of normalized landmark sequences."""
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (30, 126):
            raise ValueError(f"X must have shape (N, 30, 126); got {values.shape}")

        predictions = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(values[start : start + self.batch_size]).to("cpu")
                outputs = self.model(batch)
                predictions.append(outputs.argmax(dim=-1).cpu().numpy())
        if not predictions:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(predictions).astype(int, copy=False)


def load_model(path):
    """Load a TorchScript .pt model or a BiLSTMAttention .pth state dict."""
    model_path = Path(path)
    extension = model_path.suffix.lower()
    if extension == ".pt":
        model = torch.jit.load(str(model_path), map_location="cpu")
    elif extension == ".pth":
        try:
            from model_def import BiLSTMAttention
        except ImportError as error:
            raise ImportError(
                "Loading .pth files requires BiLSTMAttention in model_def.py"
            ) from error
        model = BiLSTMAttention(126, 64, 1, 129, 0.65)
        state_dict = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise ValueError(
            f"Unsupported model format {model_path.suffix!r}; supported formats are .pt and .pth"
        )
    model.eval()
    return TorchPredictor(model)


def load_data(x_path, y_path):
    """Load and validate landmark features and integer class labels."""
    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 3 or X.shape[1:] != (30, 126):
        raise ValueError(f"X must have shape (N, 30, 126); got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"y must have shape ({X.shape[0]},) to match X; got {y.shape}"
        )
    return X.astype(np.float32, copy=False), y.astype(int, copy=False)


def check_baseline(predict_fn, X, y, expected=None, tolerance=0.005):
    """Compare model accuracy with an optional expected baseline."""
    predictions = np.asarray(predict_fn(X))
    labels = np.asarray(y)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"Predictions must have shape {labels.shape}; got {predictions.shape}"
        )
    correct = int(np.count_nonzero(predictions == labels))
    total = int(labels.size)
    accuracy = correct / total if total else 0.0
    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }
    if expected is not None:
        result["matches"] = abs(accuracy - expected) <= tolerance
    return result
