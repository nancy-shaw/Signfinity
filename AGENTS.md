# AGENTS.md

## Project

Sign Model Report Card: a Streamlit app that stress-tests a trained Indian Sign Language (ISL) landmark classifier. It changes the test data in six ways, re-runs the model, and shows accuracy drops, a readiness score and per-sign results with counts.

## Data

- X: numpy array, shape (N, 30, 126). 30 frames, 126 = 42 joints x 3 values.
- Features 0-62 = left hand, features 63-125 = right hand.
- Data is already wrist-relative and scale-normalized. Never normalize again.
- y: integer class ids 0-128 (129 classes), shape (N,).
- The model is used only through predict(X), which returns class ids, shape (N,).

## Results object (the screen depends only on this)

- baseline_acc, baseline_ci_low, baseline_ci_high (floats)
- tests: {test name: accuracy}
- score: int 0-100 (mean of test accuracy / baseline accuracy, times 100)
- per_class: {class id: {word, n, correct_before, correct_after: {test name: int}}}
- samples_per_class_train: {class id: int} (optional)
- summary: str

## Files and owners

app.py (B), stress.py and evaluate.py (A1), adapters.py, summary.py, export_card.py (A2), tests/, examples/, docs/.
Do not edit files you were not asked to touch.

## Commands

pip install -r requirements.txt / pytest -q / streamlit run app.py

## Rules

- Transforms take (N,30,126) and return (N,30,126). Never change the input in place.
- Mirror = negate x of every joint, then swap the left and right hand blocks.
- Randomness always takes a seed.
- No AI calls in stress.py or evaluate.py. AI is only in summary.py.
- Never edit or delete a test just to make it pass.
- Keys come only from environment variables (.env). Never print or commit them.
- Never commit data or model files.
- Small functions with a one-line docstring. Show the commands you ran and their output.
- Do not change the results object without asking.
