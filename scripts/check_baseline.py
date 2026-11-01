"""Print the accuracy of a model on the supplied test data."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from adapters import check_baseline, load_data, load_model


def main():
    """Load command-line inputs and report baseline accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="path to a .pt or .pth model")
    parser.add_argument("x_path", help="path to the test feature .npy file")
    parser.add_argument("y_path", help="path to the test label .npy file")
    args = parser.parse_args()

    model = load_model(args.model_path)
    X, y = load_data(args.x_path, args.y_path)
    result = check_baseline(model.predict, X, y)
    print(f"Baseline accuracy: {result['accuracy']:.4f}")
    print(f"Correct: {result['correct']}/{result['total']}")


if __name__ == "__main__":
    main()
