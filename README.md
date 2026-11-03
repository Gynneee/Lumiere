# CineMatch

CineMatch is a browser-based photo color-matching tool. Choose a photo to edit and a reference photo, compare the before-and-after preview, refine the suggested Lightroom-style settings, and export the result.

The app runs locally in your browser. Your photos are processed on your device and are not sent to an inference API.

## Features

- Upload a source photo and a reference image (JPEG, PNG, WebP, or supported camera RAW formats such as ARW, CR2, CR3, NEF, and DNG).
- Compare the original and matched result with a draggable split preview.
- Inspect luminance histograms and dominant colors.
- Adjust white balance, tone, tone curve, HSL, color grading, and match strength.
- Optionally protect likely skin-colored pixels from aggressive HSL and grading color shifts.
- Apply manual radial barrel/pincushion distortion correction.
- Export the matched image as PNG or copy Lightroom-compatible preset XML and settings.
- Optionally run a locally trained ONNX model in the browser.
- Fall back to the built-in calculation-based matcher when no trained model is installed.

RAW files are developed locally by LibRaw into an RGB image that can be edited and exported as PNG. This does not modify or export the original sensor data. The manual distortion control is not camera/lens-profile based; automatic lens profiles and Lightroom-grade RAW rendering are not implemented. Optical correction is baked into PNG exports, not copied into XMP presets.

## Run locally

Requirements: Node.js and npm.

```powershell
npm install
npm run dev
```

Open the local address printed by Vite. The development server sets the cross-origin isolation headers required by the RAW decoder. Static deployments need to set `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` too.

To create a single-HTML app build:

```powershell
npm run build
```

The app page is written to `dist/Grade Match.html`; LibRaw's worker bundles are separate files in `dist/`, so keep the complete output directory together and serve it with the headers above to use RAW decoding. Use the development server when testing a local ONNX model.

## AI model and current status

The optional AI is a supervised neural network trained locally with Python and exported to ONNX. The browser runs ONNX inference using ONNX Runtime Web. The model accepts summaries of the source and reference images and predicts 50 Lightroom-style adjustment values.

The local model is trained from labeled source/reference pairs. Labels include known synthetic edits and, for selected pairs, the JavaScript color-matching solver's output. The browser runs both the ONNX model and the calculation matcher, then selects the result with the lower measured distance to the reference. This quality guard is not a claim of human-perceived accuracy. Portrait expansion and the skin-protection limitation are documented in [the training guide](docs/ai-training.md#portrait-focused-expansion-and-skin-tone-handling).

The earlier expanded candidate used 12,202 pairs: 11,106 synthetic known-setting pairs and 1,096 solver-labeled pairs, with 9,782 training / 2,420 validation examples across 3,006 / 752 disjoint source-photo groups. It completed 400 epochs and selected epoch 399. Validation MSE was 0.023078 and normalized MAE was 0.061148, versus 0.257654 for that run's zero-output baseline (76.3% lower MAE). It remains in `public\models\color_matcher_candidate.onnx`.

A later full-data candidate retains all 12,010 available flower pairs and adds every available portrait, landscape, architecture, and urban pair: 12,691 labeled pairs from 3,775 unique source photos. Its five category losses are weighted equally during training and validation so the much larger flower set does not dominate the objective. It trained on 10,141 pairs and validated on 2,550 source-disjoint pairs; early stopping ended training at epoch 270 and restored epoch 230. Macro category validation MSE was 0.025456, and macro normalized MAE was 0.070611 versus 0.253548 for a zero-output baseline (72.2% lower regression error, not 72.2% accuracy). The validation MAE by category was 0.0645 for architecture, 0.0712 for flowers, 0.0656 for landscapes, 0.0727 for portraits, and 0.0790 for urban scenes. These results measure prediction against synthetic and JavaScript-solver labels, not human-perceived color quality. Non-flower categories still have only 15-59 unique source photos apiece, and their validation sets contain just 3-12 source photos, so their category metrics are preliminary. This candidate is saved separately as `public\models\color_matcher_all_scenes_candidate.onnx`; the app's `color_matcher.onnx` was not replaced. The full Word/text report, per-epoch history, and visualizations are in `local-report\all_scenes_run_MODEL_TRAINING_REPORT.docx`, `.txt`, and `all_scenes_run_training_metrics*.png`; run metrics and curves are under `dataset\all_scenes_run\`.

The Commons acquisition was rate-limited (HTTP 429) after 50 portrait and 17 landscape photos; no new architecture or urban photos were obtained. Existing diverse-scene photos supplied those categories. The earlier 225-pair candidate used balanced source-photo sampling; the full-data run instead kept all available examples and balanced category loss. The candidate runs used different validation data, so their metrics should not be compared as if they were measured on the same test set. All datasets, candidate models, and reports are Git-ignored and are not included in a fresh clone. See the [earlier expanded-run curve](docs/training_curves.svg) and [full-data training guide](docs/ai-training.md#full-available-mixed-scene-run).

For training details, dataset format, and limitations, see [docs/ai-training.md](docs/ai-training.md).

### Train a compact local model

Requirements: Python 3.12 or newer. From PowerShell in the project folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
python prepare_synthetic_training.py
python train_color_matcher.py --manifest dataset\flower_match\train.jsonl --output public\models\color_matcher.onnx
```

The preparation script downloads TensorFlow's 218 MB Flower Photos archive and generates three synthetic edits of each of 3,670 flower photos. Training and validation use separate source photos. The labels are synthetic, not expert retouching or proof of general photo-style matching. The photos are CC BY 2.0; keep the generated attribution file with the local dataset and review the license before redistribution. The dataset and trained model are ignored by Git.

For your own dataset, each JSONL training example needs a source image, a reference image, and the desired adjustment settings. See the training guide for the full schema and instructions.

## How matching works

1. The app extracts color and tone statistics from the source and reference images.
2. If a local ONNX model is available, it predicts adjustment settings from the two image summaries.
3. Otherwise, the built-in calculation-based matcher estimates settings from the image statistics.
4. The selected adjustments are applied to a browser canvas for preview and image export.

The model does not generate pixels directly. It predicts adjustment controls; the app applies those controls to the source image.

## Project files

- `Grade Match.html` — app interface.
- `script.js` and `styles.css` — browser behavior and styling.
- `train_color_matcher.py` — local neural-network training and ONNX export.
- `prepare_expanded_manifest.py` — combines available local manifests and keeps identical source photos in one data split.
- `prepare_synthetic_training.py` — creates labeled synthetic examples from the compact flower-photo dataset.
- `prepare_diverse_training.py` — obtains openly licensed scene photos and generates diverse synthetic pairs.
- `export_js_teacher_labels.py` — locally collects adjustment labels from the JavaScript solver.
- `docs/ai-training.md` — dataset, training, and model instructions.

## License

No project license is specified in this repository. The downloaded flower photos are licensed under CC BY 2.0; preserve their photographer credits from `dataset/flower_match/Flower_Photos_LICENSE.txt` and the selected-image credits in `dataset/flower_match/ATTRIBUTION.txt`.
