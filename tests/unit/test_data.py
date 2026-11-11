from pathlib import Path

from PIL import Image

from industrial_visual_defect_detection.data import (
    Sample,
    create_manifest,
    validate_samples,
    visa_samples,
)


def test_normal_calibration_is_deterministic_and_disjoint(tmp_path: Path):
    samples = []
    for i in range(12):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), (i, 0, 0)).save(image)
        import hashlib
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        samples.append(Sample(image, None, 0, "train", "normal", digest))
    first = create_manifest(samples, seed=7, calibration_fraction=.25)
    second = create_manifest(samples, seed=7, calibration_fraction=.25)
    assert first == second
    train = {r["sha256"] for r in first["records"] if r["split"] == "train"}
    cal = {r["sha256"] for r in first["records"] if r["split"] == "calibration"}
    assert train.isdisjoint(cal)
    assert len(cal) == 3


def test_official_csv_assignments_and_mask_alignment(tmp_path: Path):
    root = tmp_path / "VisA"
    (root / "split_csv").mkdir(parents=True)
    records = ["object,set,label,image_path,mask_path"]
    for i, (split, label, name) in enumerate([("train", "normal", "n1"), ("train", "normal", "n2"), ("test", "normal", "n3"), ("test", "anomaly", "a1")]):
        image = root / "pcb1" / "Data" / "Images" / name
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10), (10 + i, 20, 30)).save(image.with_suffix(".png"))
        mask_rel = ""
        if label == "anomaly":
            mask = root / "pcb1" / "Data" / "Masks" / "a1.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (12, 10), 255).save(mask)
            mask_rel = str(mask.relative_to(root))
        records.append(f"pcb1,{split},{label},{image.with_suffix('.png').relative_to(root)},{mask_rel}")
    (root / "split_csv" / "1cls.csv").write_text("\n".join(records) + "\n")
    samples = visa_samples(root, "pcb1")
    assert [s.split for s in samples] == ["train", "train", "test", "test"]
    report = validate_samples(samples)
    assert report["error_count"] == 0


def test_validation_reports_corrupt_image(tmp_path: Path):
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"not an image")
    sample = Sample(corrupt, None, 0, "train", "normal", "abc")
    report = validate_samples([sample])
    assert report["error_count"] == 1
    assert "unreadable image" in report["errors"][0]["error"]
