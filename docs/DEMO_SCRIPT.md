# Two-Minute Demo Script

## Flow

### 0:00–0:20 — Introduce the report

**Click:** Open the app page.

**Say:** “This report card checks whether a landmark-based sign classifier stays reliable when its inputs are perturbed. It first measures the clean baseline, then reruns the same model against six stress tests.”

### 0:20–0:40 — Upload the matching files

**Click:** Select the model file and its matching `X_test.npy` and `y_test.npy` files. Enter the claimed baseline accuracy, then click **Analyze model**.

**Say:** “The model and test arrays need to come from the same evaluation setup. The baseline check catches a mismatched model or data file before we interpret the stress results.”

### 0:40–1:05 — Read the baseline and summary

**Point at:** The baseline status and baseline card. For the bundled demonstration, the measured baseline is **85.64% (328/383)**.

**Say:** “This is the clean-data reference. If the claimed accuracy does not match within tolerance, the banner reports that the model or data may not match.”

### 1:05–1:30 — Compare perturbations

**Point at:** The perturbation chart and sorted table. The demonstration robustness score is **0.871**. The weakest test is **Mirror (left hand), at 22.98% accuracy**.

**Say:** “The score compares mean perturbed accuracy with the clean baseline. The mirror result is a large drop, which suggests this model is sensitive to handedness changes.”

### 1:30–1:45 — Review class details

**Point at:** The ten weakest classes and the sample-count column.

**Say:** “These class-level results are useful for finding candidates to investigate, but each class has only a few examples, so small differences can be noisy.”

### 1:45–2:00 — Download the card

**Click:** **Download Robustness Card**, then show the downloaded HTML or its print preview.

**Say:** “The card is a portable summary with the baseline, stress results, worst cases, and limitations. It can be printed or saved as a PDF.”

## Likely Judge Questions

**Why check the baseline?**

It confirms that the uploaded model and test labels reproduce the expected clean result before attributing changes to stress tests.

**Why must the model match the data?**

Accuracy is meaningful only when predictions are compared with labels from the same test set and the model has the expected input representation.

**How are uploaded files handled securely?**

Uploads use random temporary filenames and are deleted after analysis. The app accepts model and NumPy data extensions only and never executes uploaded Python files. A `.pth` file is a serialized model artifact and should come only from a trusted source.

**Why test only six perturbations?**

They are a small, interpretable set of changes to handedness, timing, landmark noise, scale, and frame availability. They are not a complete evaluation of real-world variation.

**What would you add next?**

I would add more signer and recording-condition slices, confidence and error analysis, and ONNX runtime support, then validate each addition against held-out data.
