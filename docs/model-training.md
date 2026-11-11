# Model training

`industrial-visual-defect-detection train` fits the student with configurable AdamW, Adam, or SGD optimization (AdamW by default), finite-loss checks, gradient clipping, deterministic seeds, train and normal calibration losses, and best/latest safetensors weights. Training uses normal images only. `--resume` reloads the latest student weights and continues for the number of epochs requested; optimizer state is not currently restored, so this is a weight-level continuation rather than exact optimizer resume.

The teacher uses torchvision ResNet18 ImageNet1K_V1 weights and stays frozen. The feature objective for scale `l` is `L_l = mean((normalize(T_l(x)) - normalize(S_l(x)))^2)` and `L = (L_1 + L_2 + L_3)/3`. Input normalization uses ImageNet channel means and standard deviations. Inference score is the average of the largest 1% of map pixels.

`industrial-visual-defect-detection calibrate --normal-quantile 0.99` computes the requested empirical quantile over normal calibration image scores and a separate 99.5th percentile over normal calibration pixel scores. Both are stored per run. The pixel threshold uses many spatially correlated pixels, so its empirical quantile is a descriptive operating point rather than a statistically independent false-positive guarantee. Neither value guarantees future production rates.
