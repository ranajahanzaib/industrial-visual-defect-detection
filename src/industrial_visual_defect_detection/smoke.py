"""Synthetic software-path check; all outputs are explicitly non-benchmark evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file

from industrial_visual_defect_detection.api.schemas import PredictionResponse
from industrial_visual_defect_detection.data import image_transform
from industrial_visual_defect_detection.inference import Detector
from industrial_visual_defect_detection.models import Student, Teacher, distillation_loss, feature_map_scales, feature_maps
from industrial_visual_defect_detection.utils import atomic_json, seed_everything


def smoke(out: Path, seed: int = 42, size: int = 64) -> dict[str, Any]:
    if size < 32: raise ValueError("smoke input size must be at least 32")
    seed_everything(seed)
    teacher, student = Teacher(pretrained=False).cpu(), Student().cpu()
    transform = image_transform(size)
    normal = Image.new("RGB", (size, size), (126, 132, 120))
    x = torch.stack((transform(normal), transform(normal))).cpu()
    before = next(student.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    target, pred = teacher(x), student(x)
    loss, _ = distillation_loss(target, pred)
    if not torch.isfinite(loss): raise FloatingPointError("synthetic smoke loss is non-finite")
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    changed = not torch.equal(before, next(student.parameters()).detach())
    with torch.inference_mode():
        raw_scales = feature_map_scales(teacher(x), student(x), (size, size))
    stats = []
    for fmap in raw_scales:
        values = fmap.flatten()
        stats.append({"mean": float(values.mean()), "std": max(float(values.std(unbiased=False)), 1e-8)})
    with torch.inference_mode():
        normal_map = feature_maps(teacher(x[:1]), student(x[:1]), (size, size), stats)[0, 0]
        pixel_threshold = float(normal_map.flatten().quantile(.995))
        normal_score = float(normal_map.flatten().topk(max(1, int(normal_map.numel() * .01))).values.mean())
    metadata = {"model": "synthetic-smoke-only", "pretrained_teacher": False, "teacher": "randomly initialized ResNet18 (synthetic software validation only)", "size": size, "score_top_fraction": .01, "map_normalization": stats, "threshold": normal_score, "pixel_threshold": pixel_threshold, "seed": seed, "device": "cpu", "category": "synthetic", "training_images": 2, "calibration_images": 1, "history": [{"epoch": 1, "train_loss": float(loss.detach()), "calibration_loss": float(loss.detach())}], "evidence": "software validation only; not industrial performance"}
    out.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}, str(out / "best.safetensors"))
    save_file({k: v.detach().cpu().contiguous() for k, v in teacher.state_dict().items()}, str(out / "teacher.safetensors"))
    atomic_json(out / "metadata.json", metadata)
    test_image = normal.copy()
    draw = ImageDraw.Draw(test_image)
    draw.rectangle((size // 3, size // 3, size // 2, size // 2), fill=(245, 30, 30))
    detector = Detector(out, "cpu")
    result = detector.predict(test_image)
    response = PredictionResponse(**(result | {"anomaly_map": result["anomaly_map"].tolist()}))
    if not changed or response.image_width != size or response.image_height != size or np.asarray(result["anomaly_map"]).shape != (size, size) or not np.isfinite(result["score"]):
        raise RuntimeError("synthetic image → preprocessing → model → score → heatmap → response pipeline failed")
    np.save(out / "synthetic-anomaly-map.npy", result["anomaly_map"])
    report = {"model": metadata["model"], "seed": seed, "loss": float(loss.detach()), "score": response.score, "threshold": response.threshold, "feature_map_shape": list(np.asarray(result["anomaly_map"]).shape), "response_valid": True, "evidence": metadata["evidence"]}
    atomic_json(out / "smoke.json", report)
    return report
