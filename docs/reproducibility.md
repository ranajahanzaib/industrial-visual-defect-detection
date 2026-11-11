# Reproducibility

Set a seed with `--seed`; data sampling, Python, NumPy, and PyTorch random generators are seeded. Each run manifest records image paths, hashes, split assignments, category, and protocol. Run metadata records model configuration, device, loss history, calibration data count, threshold method, quantile, and installed runtime versions. Checkpoints use safetensors; metadata is JSON. `industrial-visual-defect-detection reproduce --run ...` re-fits, re-calibrates, and re-evaluates into a fresh output directory using the stored manifest and configuration.

Reproducibility is bounded by hardware and library versions. Nondeterministic accelerator kernels may vary, and exact resume is not currently supported because optimizer state is not saved. The dataset manifest records local hashes but these are not official dataset checksums.
