# Industrial Visual Defect Detection

**Train visual anomaly detectors using normal product images. Inspect unusual regions, evaluate failures, and compare compact models locally.**

Industrial Visual Defect Detection is an open-source research workbench for industrial visual inspection. It includes a PyTorch teacher–student feature-distillation detector, a ResNet18 nearest-neighbor feature-bank baseline, a VisA dataset adapter, normal-only threshold calibration, evaluation metrics, a FastAPI service, and a React inspection workbench.

> The repository does not currently include VisA data, trained weights, or benchmark results. The dashboard shows an honest setup state until a model is trained and calibrated. Anomaly scores are not probabilities; heatmaps show model discrepancy, not defect cause.

## How it works

The trainable detector keeps an ImageNet-pretrained ResNet18 teacher frozen and trains a smaller depthwise-separable student to reproduce its layer1, layer2, and layer3 features on normal images. At inference, the mean multi-scale normalized feature discrepancy forms the anomaly map; the image score is the mean of its highest-scoring 1% of pixels. A held-out normal calibration partition sets the image-level quantile threshold.

The PatchCore-style baseline path uses frozen ResNet18 features and a deterministic, batch-stratified random feature-bank subset. It is a bounded random coreset variant, not the greedy k-center PatchCore algorithm. Fit it with `industrial-visual-defect-detection fit-baseline --data-root data/VisA --category pcb1 --run runs/pcb1-patchcore`, then calibrate and evaluate with the corresponding commands above. Compare evaluated runs with `industrial-visual-defect-detection compare --runs runs/pcb1-student runs/pcb1-patchcore` to produce paired metrics and visual error comparisons.

## Quick start

Python 3.11+ is required. Install the project and development tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Use an official VisA extraction under `data/VisA`, or explicitly initiate its download (large archive; requires free disk space):

```bash
industrial-visual-defect-detection data download --data-root data/VisA
```

Then inspect and prepare the category:

```bash
industrial-visual-defect-detection doctor
industrial-visual-defect-detection data inspect --data-root data/VisA --category pcb1
industrial-visual-defect-detection data prepare --data-root data/VisA --category pcb1
industrial-visual-defect-detection train --config configs/student-pcb1.yaml
industrial-visual-defect-detection calibrate --data-root data/VisA --category pcb1 --run runs/pcb1-student --normal-quantile 0.99
industrial-visual-defect-detection evaluate --data-root data/VisA --category pcb1 --run runs/pcb1-student --split test
industrial-visual-defect-detection predict --run runs/pcb1-student --image /path/to/product-image.jpg --output runs/prediction.npy
```

The first training run may download torchvision's ResNet18 ImageNet weights. Dataset downloading is not automatic. `data prepare` writes a deterministic manifest using official train/test assignments and reserves normal training images for calibration. It does not copy or modify source data.

Run the API and dashboard in separate terminals:

```bash
uvicorn industrial_visual_defect_detection.api.app:app --reload
cd apps/web && pnpm install && pnpm run dev
```

Set `INDUSTRIAL_VISUAL_DEFECT_DETECTION_RUNS` in the API process if model runs live outside `./runs`. The API accepts image uploads up to 10 MiB. Set `VITE_API_URL` in `apps/web/.env.local` to override the dashboard's default API origin.

## Dataset and scientific protocol

Industrial Visual Defect Detection supports the 12 VisA one-class categories: pcb1–pcb4, candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, and pipe_fryum. Obtain the dataset from the [official AWS registry](https://registry.opendata.aws/visa/) and follow the distributed [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). Cite the dataset paper, [“SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation”](https://arxiv.org/abs/2207.14315), when reporting results. Do not commit the dataset.

The primary split uses only official normal training records to fit the student or baseline. A seeded subset of those normal records calibrates separate image and pixel thresholds; the official test partition remains held out. Test anomalies do not contribute to fitting or calibration. AUROC and average precision are threshold-independent; confusion metrics and segmentation metrics use their respective normal-only calibrated thresholds. A normal-only quantile is not a guarantee of future false-positive rate.

## Architecture

```text
Normal training images → frozen ResNet18 teacher ─┐
                       → trainable compact student ├→ feature loss / student update
                                                   └→ calibrated inference → map + score
Normal training images → frozen ResNet18 features → bounded random memory bank
```

See [architecture](docs/architecture.md), [dataset protocol](docs/dataset.md), [training](docs/model-training.md), [evaluation](docs/evaluation.md), [reproducibility](docs/reproducibility.md), [API](docs/api.md), and [limitations](docs/limitations.md).

## Data and benchmark evidence

VisA images, trained weights, and benchmark results are not included. Prepare the dataset and run the documented training and evaluation workflow to generate category-specific results. Synthetic fixtures exercise software paths only; they are not evidence of industrial performance. Run `python -m pytest`, `python -m ruff check src tests`, and `python -m mypy --ignore-missing-imports src/industrial_visual_defect_detection` to validate a local installation; CI runs these checks and builds the web workbench.

## License

The Industrial Visual Defect Detection code is MIT licensed. VisA images and annotations remain governed by the dataset's own terms.
