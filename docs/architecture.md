# Architecture

## Student detector

The frozen torchvision ResNet18 teacher emits spatial features from layer1, layer2, and layer3. A compact convolutional student downsamples through depthwise-separable blocks and projects its outputs to 64, 128, and 256 channels. For each scale, both feature tensors are L2-normalized along channels and trained with mean squared error. The total loss is the arithmetic mean of the three scale losses.

At inference, the same normalized squared feature discrepancy is averaged over channels. Each scale is resized to the input grid and normalized using per-scale mean and standard deviation measured on the held-out normal calibration subset; the positive z-score excess is then averaged across scales. Calibration images are separate from student fitting. An independent upper quantile of the resulting normal maps sets the pixel threshold. The image score is the mean of the highest-scoring configurable fraction of pixels (1% by default), and its image threshold is calibrated independently. These quantiles describe the calibration distribution and do not guarantee future error rates.

## Baseline

The `PatchMemory` class extracts concatenated ResNet18 layer1–3 patch embeddings, aligns them to the layer2 grid, and selects up to `max_points` vectors with seeded per-batch random sampling. Inference computes nearest-bank Euclidean distance in query chunks. This bounds query memory; the bank still occupies memory proportional to its size. The selected random subset is explicitly not the greedy coreset used in the original PatchCore paper.

## Flow

```text
                Normal Training Images
                         │
                  Preprocessing
                         │
              ┌──────────┴──────────┐
              │                     │
       Frozen Teacher         Trainable Student
              │                     │
       Teacher Features       Student Features
              └──────────┬──────────┘
                         │
              normalized multi-scale MSE
                         │
                  Updated Student

Test image → teacher + student → feature discrepancy → map → top-1% score
                                                        ↓
                                       normal-only threshold → pass / flag
```
