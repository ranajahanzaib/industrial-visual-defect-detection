from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file

from industrial_visual_defect_detection.data import image_transform
from industrial_visual_defect_detection.models import PatchMemory, Student, Teacher, feature_maps
from industrial_visual_defect_detection.utils import choose_device


class Detector:
    def __init__(self, run: Path, device_name: str = "auto"):
        self.run = run
        self.device = choose_device(device_name)
        self.metadata: dict[str, Any] = json.loads((run / "metadata.json").read_text())
        self.is_patchcore = self.metadata.get("model", "").startswith("patchcore")
        if self.is_patchcore:
            self.memory = PatchMemory().to(self.device).eval()
            self.memory.bank = load_file(str(run / "memory.safetensors"), device=str(self.device))["bank"]
        else:
            pretrained = bool(self.metadata.get("pretrained_teacher", True))
            self.teacher, self.student = Teacher(pretrained).to(self.device).eval(), Student().to(self.device).eval()
            if (run / "teacher.safetensors").exists():
                self.teacher.load_state_dict(load_file(str(run / "teacher.safetensors"), device=str(self.device)))
            self.student.load_state_dict(load_file(str(run / "best.safetensors"), device=str(self.device)))
        self.transform = image_transform(int(self.metadata["size"]))

    def _synchronize(self) -> None:
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)
        elif self.device.type == "mps": torch.mps.synchronize()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict[str, Any]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        self._synchronize()
        started = time.perf_counter()
        x = self.transform(image).unsqueeze(0).to(self.device)
        self._synchronize()
        preprocessing_ms = (time.perf_counter() - started) * 1000
        stage_start = time.perf_counter()
        teacher_ms = student_ms = map_ms = patchcore_ms = None
        if self.is_patchcore:
            amap = self.memory.predict_map(x)[0, 0]
            self._synchronize(); patchcore_ms = (time.perf_counter() - stage_start) * 1000
        else:
            teacher_features = self.teacher(x)
            self._synchronize(); teacher_ms = (time.perf_counter() - stage_start) * 1000
            stage_start = time.perf_counter()
            student_features = self.student(x)
            self._synchronize(); student_ms = (time.perf_counter() - stage_start) * 1000
            stage_start = time.perf_counter()
            amap = feature_maps(teacher_features, student_features, x.shape[-2:], self.metadata.get("map_normalization"))[0, 0]
            self._synchronize(); map_ms = (time.perf_counter() - stage_start) * 1000
        fraction = float(self.metadata.get("score_top_fraction", .01))
        score = float(amap.flatten().topk(max(1, int(amap.numel() * fraction))).values.mean().cpu())
        self._synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        threshold = self.metadata.get("threshold")
        pixel_threshold = self.metadata.get("pixel_threshold")
        return {"score": score, "threshold": threshold, "pixel_threshold": pixel_threshold, "decision": "flag" if threshold is not None and score >= threshold else ("pass" if threshold is not None else "uncalibrated"), "latency_ms": elapsed, "preprocessing_ms": preprocessing_ms, "teacher_ms": teacher_ms, "student_ms": student_ms, "map_generation_ms": map_ms, "patchcore_inference_ms": patchcore_ms, "model": self.metadata.get("model"), "category": self.metadata.get("category"), "image_width": image.width, "image_height": image.height, "heatmap_reference": "inline:anomaly_map", "overlay_reference": None, "predicted_mask": (amap >= float(pixel_threshold)).cpu().numpy().astype(bool).tolist() if pixel_threshold is not None else None, "anomaly_map": amap.cpu().numpy().astype(np.float32)}
