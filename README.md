# Sign Model Report Card

Sign Model Report Card stress-tests a landmark-based sign language classifier on a supplied test set. It reruns the model after six controlled input perturbations and compares the results with a clean baseline. It summarizes overall robustness and highlights perturbation and class-level weaknesses.

## What it tests

| Perturbation | Why it matters |
| --- | --- |
| Mirror (left hand) | Checks whether predictions depend on handedness by mirroring the x coordinates and swapping hands. |
| Slower (0.7x) | Checks whether a sign is still recognized when its motion unfolds more slowly. |
| Faster (1.3x) | Checks sensitivity to faster signing and shorter motion timing. |
| Noise | Checks the effect of small landmark measurement jitter. |
| Scale jitter | Checks sensitivity to changes in landmark movement magnitude without renormalizing. |
| Missing frames | Checks whether recognition survives temporary hand-detection gaps. |

## Install and run

The following commands use Windows PowerShell and Python 3.14:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q --basetemp=.pytest_tmp
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

## What to upload

Upload a model file, `X_test.npy`, and `y_test.npy`. A claimed accuracy is optional and lets the app check whether the uploaded model and data match the expected baseline. The feature array must have shape `(N, 30, 126)`, and the label array must have shape `(N,)` with the same sample count. The features are already wrist-relative and scale-normalized; do not normalize them again.

The current adapter loads self-contained TorchScript `.pt` models and `.pth` state dictionaries. Prefer a self-contained TorchScript file for model portability. The upload form accepts `.onnx` by extension, but the current adapter cannot load ONNX yet. Loading a `.pth` file can execute serialized code; use trusted files only.

## What the Robustness Card contains

The card contains baseline accuracy and status, a robustness score, per-perturbation accuracy and accuracy drops, the worst perturbation, and the ten weakest classes with their sample counts. It includes a plain-language guide and limitations, and can be printed or saved as a PDF from the browser.

## Limitations

Per-class numbers use about three samples per class, so they are noisy. The report evaluates only the six listed perturbations. Results apply only to the uploaded model and data; they do not establish performance for other signers, recording conditions, or datasets.
