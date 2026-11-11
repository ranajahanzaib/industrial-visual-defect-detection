from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class RunSummary(BaseModel):
    id: str
    model: str


class PredictionResponse(BaseModel):
    score: float
    threshold: float | None
    pixel_threshold: float | None = None
    decision: str
    latency_ms: float
    preprocessing_ms: float | None = None
    teacher_ms: float | None = None
    student_ms: float | None = None
    map_generation_ms: float | None = None
    patchcore_inference_ms: float | None = None
    model: str | None = None
    category: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    heatmap_reference: str | None = None
    overlay_reference: str | None = None
    predicted_mask: list[list[bool]] | None = None
    anomaly_map: list[list[float]]


class EvaluationResponse(BaseModel):
    category: str | None = None
    split: str | None = None
    sample_count: int | None = None
    metrics: dict[str, Any] = {}
