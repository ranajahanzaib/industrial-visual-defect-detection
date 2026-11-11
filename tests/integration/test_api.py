from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import industrial_visual_defect_detection.api.app as api_module
from industrial_visual_defect_detection.api.app import app


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


def test_health_and_empty_run_list(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_module, "RUNS", tmp_path / "runs")
    client = TestClient(app)
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/runs").json() == []


def test_upload_errors_and_valid_inference(monkeypatch, tmp_path: Path):
    runs = tmp_path / "runs"
    run = runs / "test-run"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text("{}")
    (run / "best.safetensors").write_bytes(b"test")
    monkeypatch.setattr(api_module, "RUNS", runs)

    class StubDetector:
        def __init__(self, *_): pass
        def predict(self, image):
            return {"score": .4, "threshold": .3, "pixel_threshold": .2, "decision": "flag", "latency_ms": 2.0, "model": "stub", "anomaly_map": np.ones((2, 2), dtype=np.float32)}

    monkeypatch.setattr(api_module, "Detector", StubDetector)
    client = TestClient(app)
    invalid = client.post("/api/runs/test-run/predict", files={"file": ("bad.png", b"nope", "image/png")})
    assert invalid.status_code == 400
    missing = client.post("/api/runs/missing/predict", files={"file": ("image.png", _png(), "image/png")})
    assert missing.status_code == 404
    valid = client.post("/api/inference?run_id=test-run", files={"file": ("image.png", _png(), "image/png")})
    assert valid.status_code == 200
    assert valid.json()["decision"] == "flag"
    assert valid.json()["anomaly_map"] == [[1.0, 1.0], [1.0, 1.0]]


def test_upload_size_limit(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(api_module, "RUNS", tmp_path / "runs")
    client = TestClient(app)
    response = client.post("/api/runs/no-run/predict", files={"file": ("big.png", b"x" * (10 * 1024 * 1024 + 1), "image/png")})
    assert response.status_code == 413
