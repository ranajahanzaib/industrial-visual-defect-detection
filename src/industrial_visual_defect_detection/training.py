from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from industrial_visual_defect_detection.data import VisaDataset
from industrial_visual_defect_detection.models import (
    PatchMemory,
    Student,
    Teacher,
    distillation_loss,
    feature_map_scales,
    feature_maps,
)
from industrial_visual_defect_detection.utils import (
    atomic_json,
    choose_device,
    environment,
    seed_everything,
)


def _epoch(loader: DataLoader, teacher: Teacher, student: Student, device: torch.device, optimizer: torch.optim.Optimizer | None) -> tuple[float, list[float]]:
    train = optimizer is not None
    student.train(train)
    totals, scales, n = 0., [0., 0., 0.], 0
    for x, _, _, _ in loader:
        x = x.to(device)
        with torch.set_grad_enabled(train):
            t = teacher(x)
            s = student(x)
            loss, per_scale = distillation_loss(t, s)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
        batch = x.shape[0]
        totals += float(loss.detach()) * batch
        for i, v in enumerate(per_scale): scales[i] += float(v.detach()) * batch
        n += batch
    return totals / max(n, 1), [v / max(n, 1) for v in scales]


def train_student(records: list[dict[str, Any]], out: Path, *, size: int = 256, epochs: int = 10, batch_size: int = 8, lr: float = 1e-3, optimizer_name: str = "AdamW", seed: int = 42, device_name: str = "auto", workers: int = 0, resume: bool = False, score_top_fraction: float = .01, pretrained_teacher: bool = True) -> Path:
    if not 0 < score_top_fraction <= 1: raise ValueError("score_top_fraction must be in (0, 1]")
    if size < 32 or epochs < 1 or batch_size < 1 or lr <= 0 or workers < 0: raise ValueError("size>=32, epochs>=1, batch_size>=1, learning_rate>0, and workers>=0 are required")
    optimizers = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    if optimizer_name not in optimizers: raise ValueError(f"Unsupported optimizer {optimizer_name!r}; choose from {', '.join(optimizers)}")
    if not resume and (out / "latest.safetensors").exists(): raise FileExistsError(f"Run already has a checkpoint: {out}; choose a new run ID or pass --resume")
    seed_everything(seed)
    device = choose_device(device_name)
    training_started = time.perf_counter()
    train_ds, cal_ds = VisaDataset(records, "train", size), VisaDataset(records, "calibration", size)
    if not train_ds or not cal_ds: raise ValueError("Training and normal calibration splits must both contain images")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    cal_dl = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    teacher, student = Teacher(pretrained_teacher).to(device), Student().to(device)
    trainable_parameters = sum(parameter.numel() for parameter in student.parameters() if parameter.requires_grad)
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    optimizer = optimizers[optimizer_name](student.parameters(), lr=lr)
    out.mkdir(parents=True, exist_ok=True)
    epoch0 = 0
    history: list[dict[str, Any]] = []
    best = float("inf")
    if resume and (out / "latest.safetensors").exists():
        student.load_state_dict(load_file(str(out / "latest.safetensors"), device=str(device)))
        meta = json.loads((out / "metadata.json").read_text())
        history = meta.get("history", [])
        epoch0 = len(history)
        best = float(meta.get("best_calibration_loss", float("inf")))
    for epoch in range(epoch0, epochs):
        started = time.perf_counter()
        train_loss, train_scales = _epoch(train_dl, teacher, student, device, optimizer)
        cal_loss, cal_scales = _epoch(cal_dl, teacher, student, device, None)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "calibration_loss": cal_loss,
               "train_scale_losses": train_scales, "calibration_scale_losses": cal_scales,
               "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.perf_counter() - started}
        history.append(row)
        state = {k: v.detach().cpu().contiguous() for k, v in student.state_dict().items()}
        save_file(state, str(out / "latest.safetensors"))
        if cal_loss < best:
            best = cal_loss
            save_file(state, str(out / "best.safetensors"))
        split_id = hashlib.sha256("".join(sorted(r["sha256"] for r in records)).encode()).hexdigest()
        atomic_json(out / "metadata.json", {"model": "student-distillation", "teacher": "torchvision ResNet18 ImageNet1K_V1" if pretrained_teacher else "torchvision ResNet18 random initialization (synthetic validation only)", "pretrained_teacher": pretrained_teacher, "trainable_parameters": trainable_parameters, "teacher_parameters": teacher_parameters, "size": size, "preprocessing": "RGB, resize, ImageNet normalization", "score_top_fraction": score_top_fraction, "training_config": {"epochs": epochs, "batch_size": batch_size, "learning_rate": lr, "optimizer": optimizer_name, "workers": workers}, "seed": seed, "device": str(device), "environment": environment(), "split_id": split_id, "train_images": len(train_ds), "calibration_images": len(cal_ds), "training_seconds": time.perf_counter() - training_started, "history": history, "best_calibration_loss": best, "threshold": None})
        print(f"Epoch {epoch+1:02d}/{epochs} train={train_loss:.5f} calibration={cal_loss:.5f} lr={lr:g} elapsed={row['seconds']:.1f}s")
    return out


def fit_patchcore(records: list[dict[str, Any]], out: Path, *, size: int = 256, batch_size: int = 8, max_points: int = 10000, sampling_ratio: float = 1.0, seed: int = 42, device_name: str = "auto", workers: int = 0, score_top_fraction: float = .01) -> Path:
    from safetensors.torch import save_file
    from torch.utils.data import DataLoader

    from industrial_visual_defect_detection.data import VisaDataset
    from industrial_visual_defect_detection.utils import atomic_json
    if not 0 < score_top_fraction <= 1: raise ValueError("score_top_fraction must be in (0, 1]")
    if max_points <= 0: raise ValueError("max_points must be positive")
    if not 0 < sampling_ratio <= 1: raise ValueError("sampling_ratio must be in (0, 1]")
    if size < 32 or batch_size < 1 or workers < 0: raise ValueError("size>=32, batch_size>=1, and workers>=0 are required")
    if (out / "memory.safetensors").exists(): raise FileExistsError(f"Run already has a memory bank: {out}; choose a new run ID")
    started = time.perf_counter()
    device = choose_device(device_name)
    dataset = VisaDataset(records, "train", size)
    if not dataset: raise ValueError("No normal training records are available for memory-bank fitting")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    grid = max(1, size // 8)
    effective_points = max(1, min(max_points, int(len(dataset) * grid * grid * sampling_ratio)))
    model = PatchMemory(max_points=effective_points).to(device).eval()
    model.fit(loader, device=device, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    save_file({"bank": model.bank.detach().cpu().contiguous()}, str(out / "memory.safetensors"))
    split_id = hashlib.sha256("".join(sorted(r["sha256"] for r in records)).encode()).hexdigest()
    atomic_json(out / "metadata.json", {"model": "patchcore-style-random-memory", "teacher": "torchvision ResNet18 ImageNet1K_V1", "preprocessing": "RGB, resize, ImageNet normalization", "trainable_parameters": 0, "teacher_parameters": sum(parameter.numel() for parameter in model.backbone.parameters()), "size": size, "score_top_fraction": score_top_fraction, "fit_config": {"batch_size": batch_size, "memory_bank_size": max_points, "sampling_ratio": sampling_ratio, "workers": workers}, "seed": seed, "device": str(device), "environment": environment(), "split_id": split_id, "train_images": len(dataset), "memory_points": len(model.bank), "memory_bytes": model.bank.numel() * model.bank.element_size(), "fit_seconds": time.perf_counter() - started, "sampling": "seeded stratified random patch subset per training batch", "threshold": None})
    return out


@torch.no_grad()
def calibrate_student(run: Path, records: list[dict[str, Any]], quantile: float = .99, batch_size: int = 8, device_name: str = "auto", pixel_quantile: float = .995) -> float:
    if not 0 < quantile < 1: raise ValueError("normal quantile must be between 0 and 1")
    if not 0 < pixel_quantile < 1: raise ValueError("pixel quantile must be between 0 and 1")
    device = choose_device(device_name)
    meta = json.loads((run / "metadata.json").read_text())
    teacher, student = Teacher(True).to(device).eval(), Student().to(device).eval()
    student.load_state_dict(load_file(str(run / "best.safetensors"), device=str(device)))
    dl = DataLoader(VisaDataset(records, "calibration", int(meta["size"])), batch_size=batch_size)
    moments = [{"sum": 0., "sum_squares": 0., "count": 0} for _ in range(3)]
    for x, _, _, _ in dl:
        x = x.to(device)
        for scale, fmap in enumerate(feature_map_scales(teacher(x), student(x), x.shape[-2:])):
            values = fmap.detach().double()
            moments[scale]["sum"] += float(values.sum())
            moments[scale]["sum_squares"] += float(values.square().sum())
            moments[scale]["count"] += values.numel()
    scale_stats = []
    for moment in moments:
        mean = moment["sum"] / moment["count"]
        variance = max(moment["sum_squares"] / moment["count"] - mean * mean, 0.)
        scale_stats.append({"mean": mean, "std": max(variance ** .5, 1e-8)})
    scores, pixel_values = [], []
    for x, _, _, _ in dl:
        x = x.to(device)
        amap = feature_maps(teacher(x), student(x), x.shape[-2:], scale_stats)
        pixel_values.append(amap.flatten().cpu())
        fraction = float(meta.get("score_top_fraction", .01))
        scores += amap.flatten(1).topk(max(1, int(amap.shape[-1] * amap.shape[-2] * fraction)), dim=1).values.mean(1).cpu().tolist()
    if not scores: raise ValueError("No normal calibration images available")
    threshold = float(torch.tensor(scores).quantile(quantile))
    pixel_threshold = float(torch.cat(pixel_values).quantile(pixel_quantile))
    meta.update({"threshold": threshold, "threshold_method": "normal calibration image-score quantile", "normal_quantile": quantile, "pixel_threshold": pixel_threshold, "pixel_threshold_method": "normal calibration pixel-score quantile", "pixel_normal_quantile": pixel_quantile, "calibration_config": {"normal_quantile": quantile, "pixel_normal_quantile": pixel_quantile}, "map_normalization": scale_stats, "map_fusion": "mean of per-scale positive standardized discrepancies from normal calibration mean", "calibration_dataset_id": meta.get("split_id"), "calibration_created_at_utc": datetime.now(UTC).isoformat(), "calibration_scores": {"count": len(scores), "min": min(scores), "median": float(torch.tensor(scores).median()), "max": max(scores)}, "calibration_pixels": sum(x.numel() for x in pixel_values)})
    atomic_json(run / "metadata.json", meta)
    atomic_json(run / "calibration.json", {k: meta[k] for k in ("threshold", "threshold_method", "normal_quantile", "pixel_threshold", "pixel_threshold_method", "pixel_normal_quantile", "calibration_dataset_id", "calibration_created_at_utc", "calibration_scores", "calibration_pixels", "map_normalization", "map_fusion")})
    return threshold


@torch.no_grad()
def calibrate_patchcore(run: Path, records: list[dict[str, Any]], quantile: float = .99, batch_size: int = 8, device_name: str = "auto", pixel_quantile: float = .995) -> float:
    from safetensors.torch import load_file

    from industrial_visual_defect_detection.data import VisaDataset
    if not 0 < quantile < 1: raise ValueError("normal quantile must be between 0 and 1")
    if not 0 < pixel_quantile < 1: raise ValueError("pixel quantile must be between 0 and 1")
    device = choose_device(device_name)
    meta = json.loads((run / "metadata.json").read_text())
    model = PatchMemory().to(device).eval()
    model.bank = load_file(str(run / "memory.safetensors"), device=str(device))["bank"]
    loader = DataLoader(VisaDataset(records, "calibration", int(meta["size"])), batch_size=batch_size)
    scores, pixel_values = [], []
    for x, _, _, _ in loader:
        amap = model.predict_map(x.to(device))
        pixel_values.append(amap.flatten().cpu())
        k = max(1, int(amap.shape[-1] * amap.shape[-2] * float(meta.get("score_top_fraction", .01))))
        scores += amap.flatten(1).topk(k, dim=1).values.mean(1).cpu().tolist()
    if not scores: raise ValueError("No normal calibration images available")
    threshold = float(torch.tensor(scores).quantile(quantile))
    pixel_threshold = float(torch.cat(pixel_values).quantile(pixel_quantile))
    meta.update({"threshold": threshold, "threshold_method": "normal calibration image-score quantile", "normal_quantile": quantile, "pixel_threshold": pixel_threshold, "pixel_threshold_method": "normal calibration pixel-score quantile", "pixel_normal_quantile": pixel_quantile, "calibration_config": {"normal_quantile": quantile, "pixel_normal_quantile": pixel_quantile}, "calibration_dataset_id": meta.get("split_id"), "calibration_created_at_utc": datetime.now(UTC).isoformat(), "calibration_scores": {"count": len(scores), "min": min(scores), "median": float(torch.tensor(scores).median()), "max": max(scores)}, "calibration_pixels": sum(x.numel() for x in pixel_values)})
    atomic_json(run / "metadata.json", meta)
    atomic_json(run / "calibration.json", {k: meta[k] for k in ("threshold", "threshold_method", "normal_quantile", "pixel_threshold", "pixel_threshold_method", "pixel_normal_quantile", "calibration_dataset_id", "calibration_created_at_utc", "calibration_scores", "calibration_pixels")})
    return threshold
