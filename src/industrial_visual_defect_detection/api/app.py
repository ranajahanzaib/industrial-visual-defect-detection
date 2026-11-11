from __future__ import annotations

import csv
import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from industrial_visual_defect_detection.api.schemas import (
    HealthResponse,
    PredictionResponse,
    RunSummary,
)
from industrial_visual_defect_detection.data import VISA_CATEGORIES, visa_samples
from industrial_visual_defect_detection.inference import Detector

RUNS = Path(os.environ.get("INDUSTRIAL_VISUAL_DEFECT_DETECTION_RUNS", "runs"))
app = FastAPI(title="Industrial Visual Defect Detection API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])
_detectors: OrderedDict[str, Detector] = OrderedDict()
_dataset_cache: dict[str, list] = {}
MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


@app.get("/health")
@app.get("/api/health")
def health() -> HealthResponse: return HealthResponse(status="ok")


@app.get("/api/system")
def system() -> dict:
    return {"python_device": "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu", "torch": torch.__version__, "dataset_root": str(Path("data/VisA").resolve()), "dataset_available": (Path("data/VisA/split_csv/1cls.csv").is_file()), "run_count": len(list_runs())}


@app.get("/runs")
@app.get("/api/runs")
def list_runs() -> list[RunSummary]:
    if not RUNS.exists(): return []
    return [RunSummary(id=p.name, model=json.loads((p / "metadata.json").read_text()).get("model", "unknown")) for p in RUNS.iterdir() if p.is_dir() and (p / "metadata.json").exists() and ((p / "best.safetensors").exists() or (p / "memory.safetensors").exists())]


def get_detector(run_id: str, run: Path) -> Detector:
    detector = _detectors.get(run_id)
    if detector is None:
        detector = Detector(run); _detectors[run_id] = detector
        while len(_detectors) > 2: _detectors.popitem(last=False)
    else:
        _detectors.move_to_end(run_id)
    return detector


def get_samples(category: str):
    if category not in _dataset_cache:
        _dataset_cache[category] = visa_samples(Path("data/VisA"), category)
    return _dataset_cache[category]


@app.get("/runs/{run_id}/evaluation")
@app.get("/api/runs/{run_id}/metrics")
@app.get("/api/runs/{run_id}/evaluation")
def evaluation(run_id: str) -> dict:
    path = (RUNS / run_id / "evaluation.json").resolve()
    if RUNS.resolve() not in path.parents or not path.is_file(): raise HTTPException(404, "Evaluation not found")
    return json.loads(path.read_text())


@app.get("/api/runs/{run_id}")
def run_details(run_id: str) -> dict:
    path = (RUNS / run_id / "metadata.json").resolve()
    if RUNS.resolve() not in path.parents or not path.is_file(): raise HTTPException(404, "Run not found")
    return json.loads(path.read_text())


@app.get("/api/runs/{run_id}/history")
def run_history(run_id: str) -> dict:
    meta = run_details(run_id)
    return {"history": meta.get("history", []), "model": meta.get("model"), "category": meta.get("category")}


@app.get("/api/runs/{run_id}/predictions")
def run_predictions(run_id: str, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)) -> dict:
    path = (Path("reports") / run_id / "predictions.jsonl").resolve()
    if Path("reports").resolve() not in path.parents or not path.is_file(): raise HTTPException(404, "Predictions are unavailable; evaluate this run first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    start = (page - 1) * page_size
    selected = []
    for row in rows[start:start + page_size]:
        selected.append({k: v for k, v in row.items() if k not in {"image_path", "mask_path", "heatmap_path", "heatmap_image_path", "overlay_path", "predicted_mask_path"}} | {"image_url": f"/api/runs/{run_id}/samples/{row['sample_id']}/image", "mask_url": f"/api/runs/{run_id}/samples/{row['sample_id']}/mask" if row.get("mask_path") else None, "heatmap_url": f"/api/artifacts/{run_id}/{row['sample_id']}-heatmap.png", "overlay_url": f"/api/artifacts/{run_id}/{row['sample_id']}-overlay.png", "predicted_mask_url": f"/api/artifacts/{run_id}/{row['sample_id']}-predicted-mask.png" if row.get("predicted_mask_path") else None})
    return {"run_id": run_id, "page": page, "page_size": page_size, "total": len(rows), "predictions": selected}


def prediction_record(run_id: str, sample_id: str) -> dict:
    path = (Path("reports") / run_id / "predictions.jsonl").resolve()
    if Path("reports").resolve() not in path.parents or not path.is_file(): raise HTTPException(404, "Predictions not found")
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("sample_id") == sample_id: return row
    raise HTTPException(404, "Sample prediction not found")


@app.get("/api/runs/{run_id}/samples/{sample_id}/image")
def prediction_image(run_id: str, sample_id: str):
    row = prediction_record(run_id, sample_id)
    path = Path(row["image_path"]).resolve()
    dataset_root = Path(run_details(run_id).get("dataset_root", "data/VisA")).resolve()
    if dataset_root not in path.parents or not path.is_file(): raise HTTPException(404, "Image file is unavailable")
    return FileResponse(path)


@app.get("/api/runs/{run_id}/samples/{sample_id}/mask")
def prediction_mask(run_id: str, sample_id: str):
    row = prediction_record(run_id, sample_id)
    if not row.get("mask_path"): raise HTTPException(404, "Ground-truth mask is unavailable")
    path = Path(row["mask_path"]).resolve()
    dataset_root = Path(run_details(run_id).get("dataset_root", "data/VisA")).resolve()
    if dataset_root not in path.parents or not path.is_file(): raise HTTPException(404, "Mask file is unavailable")
    return FileResponse(path)


@app.get("/api/comparisons")
def comparisons() -> list[dict]:
    root = Path("reports")
    rows = []
    for path in root.glob("comparison-*/comparison.json"):
        disagreement_path = path.parent / "disagreement.json"
        rows.append(json.loads(path.read_text()) | {"id": path.parent.name, "disagreement": json.loads(disagreement_path.read_text()) if disagreement_path.is_file() else None})
    return rows


@app.get("/api/comparisons/{comparison_id}/{artifact_id}")
def comparison_artifact(comparison_id: str, artifact_id: str):
    if artifact_id not in {"roc-curve.png", "accuracy-vs-latency.png", "accuracy-vs-storage.png", "precision-recall.png", "score-distribution.png"}:
        raise HTTPException(404, "Comparison artifact not found")
    path = (Path("reports") / comparison_id / artifact_id).resolve()
    root = Path("reports").resolve()
    if root not in path.parents or not path.is_file(): raise HTTPException(404, "Comparison artifact not found")
    return FileResponse(path)


@app.get("/api/comparisons/{comparison_id}/disagreement/{sample_id}.png")
def comparison_disagreement(comparison_id: str, sample_id: str):
    if len(sample_id) != 16 or any(c not in "0123456789abcdef" for c in sample_id): raise HTTPException(404, "Comparison artifact not found")
    path = (Path("reports") / comparison_id / "visual-disagreement" / f"{sample_id}.png").resolve()
    root = Path("reports").resolve()
    if root not in path.parents or not path.is_file(): raise HTTPException(404, "Comparison artifact not found")
    return FileResponse(path)


@app.get("/api/artifacts/{run_id}/{artifact_id}")
def artifact(run_id: str, artifact_id: str):
    allowed = {"roc-curve.png", "precision-recall.png", "score-distribution.png", "confusion-matrix.png"}
    is_sample_artifact = len(artifact_id) in {20, 28, 35} and all(c in "0123456789abcdef" for c in artifact_id[:16]) and artifact_id[16:] in {"-overlay.png", "-heatmap.png", ".npy", "-predicted-mask.png"}
    if artifact_id not in allowed and not is_sample_artifact: raise HTTPException(404, "Artifact not found")
    reports_root = Path("reports").resolve()
    root = (reports_root / run_id).resolve()
    path = (root / ("artifacts" if is_sample_artifact else "") / artifact_id).resolve()
    if reports_root not in root.parents or root not in path.parents or not path.is_file(): raise HTTPException(404, "Artifact not found")
    return FileResponse(path)


@app.get("/api/datasets")
def datasets() -> dict:
    csv_path = Path("data/VisA/split_csv/1cls.csv")
    root = Path("data/VisA")
    counts = {c: {"train_normal": 0, "test_normal": 0, "test_anomaly": 0} for c in VISA_CATEGORIES}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                category = row.get("object")
                if category not in counts: continue
                key = (row.get("set") or row.get("split") or "").lower() + "_" + ("normal" if (row.get("label") or "").lower() == "normal" else "anomaly")
                if key in counts[category]: counts[category][key] += 1
    return {"dataset": "VisA", "available": csv_path.is_file(), "categories": [{"id": c, "available": csv_path.is_file() and (root / c).is_dir(), **counts[c]} for c in VISA_CATEGORIES], "split_csv": str(csv_path) if csv_path.is_file() else None, "attribution": "Amazon Science VisA, CC BY 4.0"}


@app.get("/api/datasets/{category}/samples")
def dataset_samples(category: str, page: int = Query(1, ge=1), page_size: int = Query(40, ge=1, le=200)) -> dict:
    if category not in VISA_CATEGORIES: raise HTTPException(404, "Unknown VisA category")
    try: records = get_samples(category)
    except FileNotFoundError: raise HTTPException(404, "VisA category is not installed")
    start = (page - 1) * page_size
    selected = records[start:start + page_size]
    return {"category": category, "page": page, "page_size": page_size, "total": len(records), "samples": [{"id": s.sha256[:16], "label": s.label, "official_split": s.split, "defect": s.defect, "has_mask": s.mask is not None, "image_url": f"/api/datasets/{category}/samples/{s.sha256[:16]}", "mask_url": f"/api/datasets/{category}/samples/{s.sha256[:16]}/mask" if s.mask else None, "width": Image.open(s.path).width, "height": Image.open(s.path).height} for s in selected]}


@app.get("/api/datasets/{category}/samples/{sample_id}")
def dataset_sample_image(category: str, sample_id: str):
    if category not in VISA_CATEGORIES: raise HTTPException(404, "Unknown VisA category")
    try: records = get_samples(category)
    except FileNotFoundError: raise HTTPException(404, "VisA category is not installed")
    sample = next((s for s in records if s.sha256.startswith(sample_id)), None)
    if sample is None: raise HTTPException(404, "Sample not found")
    return FileResponse(sample.path)


@app.get("/api/datasets/{category}/samples/{sample_id}/mask")
def dataset_sample_mask(category: str, sample_id: str):
    if category not in VISA_CATEGORIES: raise HTTPException(404, "Unknown VisA category")
    try: records = get_samples(category)
    except FileNotFoundError: raise HTTPException(404, "VisA category is not installed")
    sample = next((s for s in records if s.sha256.startswith(sample_id)), None)
    if sample is None or sample.mask is None: raise HTTPException(404, "Sample mask is unavailable")
    return FileResponse(sample.mask)


@app.post("/runs/{run_id}/predict")
@app.post("/api/runs/{run_id}/predict")
@app.post("/api/inference")
async def predict(run_id: str, file: UploadFile = File(...)) -> PredictionResponse:
    image = await decode_upload(file)
    run = (RUNS / run_id).resolve()
    if RUNS.resolve() not in run.parents or not (run / "metadata.json").is_file(): raise HTTPException(404, "Model run not found")
    try:
        detector = get_detector(run_id, run)
        result = detector.predict(image)
    except Exception as exc: raise HTTPException(422, f"Inference failed: {exc}") from exc
    result["anomaly_map"] = result["anomaly_map"].tolist()
    return PredictionResponse(**result)


async def decode_upload(file: UploadFile) -> Image.Image:
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD: raise HTTPException(413, "Image exceeds 10 MiB limit")
    try:
        image = Image.open(io.BytesIO(raw)); image.verify()
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}: raise HTTPException(415, "Unsupported image format")
        if image.width * image.height > MAX_PIXELS: raise HTTPException(413, "Decoded image exceeds 25 megapixels")
        return ImageOps.exif_transpose(image).convert("RGB")
    except HTTPException:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(400, "Invalid or unsafe image")


@app.post("/api/inference/compare")
async def compare_inference(run_ids: list[str], file: UploadFile = File(...)) -> dict:
    if len(run_ids) < 2 or len(run_ids) > 3: raise HTTPException(422, "Provide two or three run IDs")
    image = await decode_upload(file)
    results = {}
    for run_id in run_ids:
        run = (RUNS / run_id).resolve()
        if RUNS.resolve() not in run.parents or not (run / "metadata.json").is_file(): raise HTTPException(404, f"Run {run_id} not found")
        try:
            detector = get_detector(run_id, run)
            result = detector.predict(image)
            result["anomaly_map"] = result["anomaly_map"].tolist()
            results[run_id] = result
        except Exception as exc: raise HTTPException(422, f"Inference failed for {run_id}: {exc}") from exc
    return {"results": results}
