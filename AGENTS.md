# AGENTS.md

## Project

Sign Model Report Card: a Streamlit app that stress-tests a trained Indian Sign
Language (ISL) landmark classifier. It changes the test data in six ways,
re-runs the model, and shows accuracy drops, a readiness score and per-sign
results with counts. All numbers come from plain code.

## Data

- X: numpy float32 array, shape (N, 30, 126). 30 frames, 126 = 42 joints x 3
  values (x, y, z).
- Features 0-62 = left hand (joints 0-20), features 63-125 = right hand
  (joints 21-41). Reshape to (N, 30, 42, 3) to work per joint; x of joint j is
  at feature 3*j.
- Data is already wrist-relative and scale-normalized (wrist = 0). Never
  normalize again anywhere.
- A frame that is all zeros means the hand was not detected. Never fill
  dropped frames with zeros.
- y: integer class ids 0-128 (129 classes), shape (N,).
- Test set: 383 samples, about 3 per class. Per-class results are noisy, so
  always show n.

## Model

- BiLSTMAttention from model_def.py, built as BiLSTMAttention(126, 64, 1, 129, 0.65).
- Weights: a state_dict .pth file kept outside the repo. Eval mode,
  torch.no_grad, CPU, batches of 64, argmax of the outputs.
- No extra preprocessing in predict(). The data is already normalized.
- The model is used only through predict(X), which returns class ids, shape (N,).
- Known baseline on the full test set: 85.64% (328 of 383). The adapter must
  reproduce this exactly. If it does not, stop and find the cause first.

## The six tests (stress.py)

1. Mirror (left hand): negate x of every joint, then swap the left and right
   hand blocks. Not 1 - x.
2. Slower (0.7x) and 3. Faster (1.3x): resample the time axis by linear
   interpolation, then pad or crop back to 30 frames.
4. Noise: Gaussian noise, sigma 0.01, fixed seed.
5. Scale jitter: multiply all values by a factor (0.8 or 1.2). Do not
   renormalize. There is no shift test, because the data is wrist-relative.
6. Missing frames: remove about 15% of frames, then interpolate back to 30.

## Results object (the screen depends only on this)

- baseline_acc, baseline_ci_low, baseline_ci_high (floats). Range =
  1.96 * sqrt(p * (1 - p) / N).
- tests: {test name: accuracy}
- score: int 0-100 (mean of test accuracy / baseline accuracy, times 100)
- per_class: {class id: {word, n, correct_before, correct_after: {test name: int}}}
- samples_per_class_train: {class id: int} (optional, original train labels only)
- summary: str

## Folder layout

app.py, stress.py, evaluate.py, adapters.py, model_def.py, summary.py,
export_card.py, tests/, examples/, docs/, requirements.txt

## Files and owners

stress.py, evaluate.py, adapters.py, model_def.py, summary.py, export_card.py, results_io.py, pages/1_Stress_Tests.py (A)

## Commands

pip install -r requirements.txt / pytest -q / streamlit run app.py

## Rules

- Transforms take (N, 30, 126) and return (N, 30, 126). Never change the input
  in place.
- Randomness always takes a seed argument. Same seed, same output.
- No AI calls in stress.py or evaluate.py. AI is only in summary.py, and
  summary.py must work with no API key through a template fallback.
- Never edit or delete a test just to make it pass. If a test is wrong,
  explain why first.
- Keys come only from environment variables (.env). Never print or commit them.
- Never commit model files (*.pth) or the full data files. A small
  examples/sample.npz (at most 100 samples) is allowed.
- Small functions with a one-line docstring. Show the commands you ran and
  their output.
- Do not change the results object without asking.

## Definition of done

The code runs, pytest passes, the acceptance command in the task passes, and
the diff is small and readable.