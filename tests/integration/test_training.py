from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from industrial_visual_defect_detection.training import train_student


def test_tiny_normal_training_saves_and_resumes_safetensors(tmp_path: Path):
    records = []
    for index, split in enumerate(("train", "train", "calibration")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (64, 64), (index * 25, 50, 120)).save(path)
        records.append({"path": str(path), "mask": None, "label": 0, "split": split,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    run = tmp_path / "student-run"
    train_student(records, run, size=64, epochs=1, batch_size=1, seed=3, device_name="cpu", pretrained_teacher=False)
    metadata = json.loads((run / "metadata.json").read_text())
    assert len(metadata["history"]) == 1
    assert (run / "best.safetensors").is_file()
    train_student(records, run, size=64, epochs=2, batch_size=1, seed=3, device_name="cpu", resume=True, pretrained_teacher=False)
    metadata = json.loads((run / "metadata.json").read_text())
    assert len(metadata["history"]) == 2
