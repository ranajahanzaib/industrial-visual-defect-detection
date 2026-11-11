from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import precision_recall_curve, roc_curve

from industrial_visual_defect_detection.utils import atomic_json


def write_evaluation(run: Path, predictions: list[dict[str, Any]], metrics: dict[str, Any]) -> Path:
    report_dir = Path("reports") / run.name
    artifact_dir = report_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pixel_threshold = json.loads((run / "metadata.json").read_text()).get("pixel_threshold")
    for row in predictions:
        try:
            amap = np.load(row["heatmap_path"])
            with Image.open(row["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                heat = np.clip(amap / max(float(pixel_threshold or 1.0) * 2, 1e-8), 0, 1)
                map_image = Image.fromarray(np.uint8(heat * 255)).resize(image.size, Image.Resampling.BILINEAR)
                heat = np.asarray(map_image, dtype=np.float32) / 255
                base = np.asarray(image, dtype=np.float32)
                alpha = np.clip(heat * .62, 0, .62)[..., None]
                overlay_color = np.zeros_like(base); overlay_color[..., 0] = 255; overlay_color[..., 1] = 105; overlay_color[..., 2] = 20
                overlay = Image.fromarray(np.uint8(base * (1 - alpha) + overlay_color * alpha))
                overlay_path = artifact_dir / f"{row['sample_id']}-overlay.png"
                heat_path = artifact_dir / f"{row['sample_id']}-heatmap.png"
                heat_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
                heat_rgb[..., 0] = np.uint8(np.clip(heat * 2, 0, 1) * 255)
                heat_rgb[..., 1] = np.uint8(np.clip((heat - .25) * 1.5, 0, 1) * 220)
                heat_rgb[..., 2] = np.uint8(np.clip((heat - .75) * 4, 0, 1) * 255)
                Image.fromarray(heat_rgb).save(heat_path)
                overlay.save(overlay_path)
                row["overlay_path"] = str(overlay_path)
                row["heatmap_image_path"] = str(heat_path)
        except (OSError, ValueError, KeyError):
            row["overlay_path"] = None
            row["heatmap_image_path"] = None
    jsonl = report_dir / "predictions.jsonl"
    jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions))
    csv_fields = ["sample_id", "category", "label", "score", "threshold", "pixel_threshold", "prediction", "outcome", "latency_ms", "preprocessing_ms", "teacher_ms", "student_ms", "map_generation_ms", "patchcore_inference_ms", "image_path", "mask_path", "predicted_mask_path", "heatmap_path", "heatmap_image_path", "overlay_path"]
    with (report_dir / "evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(predictions)
    latencies = [float(p["latency_ms"]) for p in predictions]
    latency = {"count": len(latencies), "warmup_images": 3 if predictions else 0,
               "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
               "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
               "mean_ms": statistics.fmean(latencies) if latencies else None,
               "throughput_images_per_second": 1000 / statistics.fmean(latencies) if latencies and statistics.fmean(latencies) > 0 else None,
               "device": json.loads((run / "metadata.json").read_text()).get("device", "unknown"),
               "latency_scope": "single-image inference; synchronized around model execution; first three sample inferences warmed up separately"}
    run_metadata = json.loads((run / "metadata.json").read_text())
    for stage in ("preprocessing_ms", "teacher_ms", "student_ms", "map_generation_ms", "patchcore_inference_ms"):
        values = [float(p[stage]) for p in predictions if p.get(stage) is not None]
        latency[f"{stage}_mean"] = statistics.fmean(values) if values else None
    latency.update({"fit_seconds": run_metadata.get("training_seconds", run_metadata.get("fit_seconds")), "artifact_bytes": sum(p.stat().st_size for p in run.iterdir() if p.is_file() and p.suffix == ".safetensors"), "trainable_parameters": run_metadata.get("trainable_parameters", 0), "peak_memory_bytes": None, "peak_memory_note": "not measured by this local evaluation path"})
    atomic_json(report_dir / "latency-summary.json", latency)
    atomic_json(report_dir / "metrics.json", metrics)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [int(p["label"]) for p in predictions]
    scores = [float(p["score"]) for p in predictions]
    threshold = metrics.get("threshold")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    fig, ax = plt.subplots(figsize=(4, 3)); ax.imshow(cm, cmap="Greens"); ax.set_xticks([0, 1], ["Pass", "Flag"]); ax.set_yticks([0, 1], ["Normal", "Anomaly"]); ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth");
    for (i, j), value in np.ndenumerate(np.asarray(cm)): ax.text(j, i, str(value), ha="center", va="center")
    fig.tight_layout(); fig.savefig(report_dir / "confusion-matrix.png", dpi=140); plt.close(fig)
    if len(set(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, scores); fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(fpr, tpr, label=f"AUROC {metrics.get('auroc'):.3f}"); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "roc-curve.png", dpi=140); plt.close(fig)
        precision, recall, _ = precision_recall_curve(labels, scores); fig, ax = plt.subplots(figsize=(4, 3)); ax.plot(recall, precision, label=f"AP {metrics.get('average_precision'):.3f}"); ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "precision-recall.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3));
    for cls, name, hist_color in ((0, "Normal", "#44a66c"), (1, "Anomaly", "#dc8f32")):
        values = [s for y, s in zip(labels, scores) if y == cls]
        if values: ax.hist(values, bins=24, alpha=.55, label=name, color=hist_color)
    if threshold is not None: ax.axvline(threshold, linestyle="--", color="#b6443d", label="Normal calibration threshold")
    ax.set(xlabel="Anomaly score", ylabel="Images"); ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "score-distribution.png", dpi=140); plt.close(fig)
    metadata = run_metadata
    title = html.escape(f"Industrial Visual Defect Detection evaluation · {run.name}")
    body = f"<h1>{title}</h1><p>Category: {html.escape(str(metrics.get('category')))} · split: {html.escape(str(metrics.get('split')))} · samples: {metrics.get('sample_count')}</p><pre>{html.escape(json.dumps(metrics, indent=2))}</pre><p>Model: {html.escape(str(metadata.get('model')))} · device: {html.escape(str(metadata.get('device')))}</p>"
    curve_images = "".join(f"<h2>{label}</h2><img src='{filename}'>" for label, filename in (("ROC", "roc-curve.png"), ("Precision–recall", "precision-recall.png")) if (report_dir / filename).exists())
    (report_dir / "report.html").write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title>{body}<h2>Confusion matrix</h2><img src='confusion-matrix.png'>{curve_images}<h2>Score distribution</h2><img src='score-distribution.png'><p><a href='metrics.json'>Metrics JSON</a> · <a href='evaluation.csv'>Prediction CSV</a> · <a href='predictions.jsonl'>Prediction JSONL</a> · <a href='latency-summary.json'>Latency summary</a></p>")
    return report_dir
