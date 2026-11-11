# Dataset protocol

Use the official VisA distribution and its one-class `split_csv/1cls.csv`. An explicit `industrial-visual-defect-detection data download` command streams the official archive with progress, checks available disk space, blocks unsafe archive paths and links, and records computed local archive and split-file hashes without calling them official checksums. Alternatively, use a supplied extraction. The adapter recognizes official `object`, `set`, `label`, `image_path`, and `mask_path` fields, hashes image bytes, rejects duplicate image content, and retains official train/test membership. The seeded calibration subset comes only from official normal training images. Remaining normal training images form the optimization split; anomalous training rows, if present in a category definition, are excluded from primary experiments. Test records retain their official assignment. The official split preparation script documents these fields and binary mask interpretation: [Amazon Science spot-diff](https://github.com/amazon-science/spot-diff/blob/main/utils/prepare_data.py).

The dataset adapter resolves paths from the split CSV relative to the data root and validates that they remain inside it. `industrial-visual-defect-detection data inspect` reports record counts and integrity findings; `industrial-visual-defect-detection data prepare` writes the report and refuses to create a manifest when validation errors are present. Preflight checks decode images and masks, compare original image/mask dimensions, report missing anomaly masks, and summarize source dimensions and modes. Images are converted to RGB and resized; masks use nearest-neighbor resizing and any nonzero source value is treated as anomalous. Missing masks on normal images are treated as all-zero masks during segmentation evaluation.

Never commit VisA images. The AWS Open Data Registry identifies the dataset license as CC BY 4.0 and provides the recommended citation. Retain attribution and check the license terms before redistributing images or derived artifacts.

Recommended citation:

> Zou, Yang, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. “SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.” arXiv:2207.14315 (2022).

Dataset record: [AWS Open Data Registry](https://registry.opendata.aws/visa/). Paper: [arXiv:2207.14315](https://arxiv.org/abs/2207.14315). License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
