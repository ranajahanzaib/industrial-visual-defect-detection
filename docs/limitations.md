# Limitations

- Performance depends on category-specific normal data, camera geometry, lighting, and product presentation; cross-category generalization is not implied.
- Normal-data contamination, domain shifts, subtle defects, and alignment variation can impair detection.
- A feature-discrepancy heatmap is not a causal explanation and may not trace defect boundaries accurately.
- Scores are not probabilities. A normal-only image-score quantile does not guarantee a future false-positive rate.
- Pixel AUROC, average precision, Dice, and IoU are calculated where masks are available. Image-level AUROC and average-precision bootstrap intervals and paired run comparisons are implemented; pixel confidence intervals, PRO, and subgroup analysis are not. Pixel thresholds use a spatially correlated normal-only quantile and should not be read as a guaranteed pixel false-positive rate.
- The baseline uses random coreset sampling and should not be represented as the original greedy PatchCore method.
- CPU and MPS runtime performance has not been characterized in published benchmarks. The pretrained teacher uses torchvision ImageNet weights, which may need to be downloaded on first use.
- Upload inference is local-first, but the prototype API has no authentication and is not an industrial deployment service.
