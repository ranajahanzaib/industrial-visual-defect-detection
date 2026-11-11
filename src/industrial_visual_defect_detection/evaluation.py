from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def image_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    result: dict[str, Any] = {"threshold": float(threshold), "accuracy": float(accuracy_score(y, pred)),
      "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
      "f1": float(f1_score(y, pred, zero_division=0)), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
      "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
      "true_positive_rate": float(tp / (tp + fn)) if tp + fn else None,
      "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
      "average_precision": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None}
    return result


def segmentation_metrics(target: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    t = np.asarray(target, dtype=bool).ravel()
    p = np.asarray(pred, dtype=bool).ravel()
    inter = int((t & p).sum())
    union = int((t | p).sum())
    return {"dice": float(2 * inter / (int(t.sum()) + int(p.sum()))) if t.sum() + p.sum() else 1.0,
            "iou": float(inter / union) if union else 1.0}


def pixel_metrics(target: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(target, dtype=bool).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    pred = s >= threshold
    counts = segmentation_metrics(y, pred)
    counts.update({"threshold": float(threshold), "pixel_count": int(y.size),
                   "positive_pixel_count": int(y.sum()),
                   "auroc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
                   "average_precision": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else None})
    return counts


def bootstrap_image_intervals(labels: list[int], scores: list[float], *, replicates: int = 500, seed: int = 42) -> dict[str, Any]:
    """Percentile intervals resampling images (never pixels) as paired units."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) < 2 or replicates < 1 or len(np.unique(y)) < 2:
        return {"replicates_requested": replicates, "valid_replicates": 0, "auroc_95_ci": None, "average_precision_95_ci": None, "unit": "image"}
    rng = np.random.default_rng(seed)
    aurocs, aps = [], []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2: continue
        aurocs.append(roc_auc_score(y[indices], s[indices]))
        aps.append(average_precision_score(y[indices], s[indices]))
    def interval(values: list[float]) -> list[float] | None:
        return [float(np.quantile(values, .025)), float(np.quantile(values, .975))] if values else None
    return {"replicates_requested": replicates, "valid_replicates": len(aurocs), "auroc_95_ci": interval(aurocs), "average_precision_95_ci": interval(aps), "unit": "image", "seed": seed}
