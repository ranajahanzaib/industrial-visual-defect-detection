from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

VISA_CATEGORIES = ("pcb1", "pcb2", "pcb3", "pcb4", "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pipe_fryum")
VISA_ARCHIVE_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"


def image_transform(size: int):
    if size < 1: raise ValueError("image size must be positive")
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


@dataclass(frozen=True)
class Sample:
    path: Path
    mask: Path | None
    label: int
    split: str
    defect: str
    sha256: str


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def visa_samples(root: Path, category: str) -> list[Sample]:
    """Read official VisA split CSVs; preserve official test assignments."""
    root = root.expanduser().resolve()
    candidates = [root / "split_csv" / "1cls.csv", root / "1cls.csv"]
    csv_path = next((p for p in candidates if p.is_file()), None)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find VisA split_csv/1cls.csv under {root}")
    records: list[Sample] = []
    seen: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category_value = row.get("object") or row.get("category") or row.get("object_name")
            if category_value != category:
                continue
            rel = row.get("image") or row.get("image_path") or row.get("path")
            if not rel:
                continue
            image = (root / rel).resolve()
            if not image.is_file():
                # Some extracted layouts have the category as the dataset root.
                image = (root / category / rel).resolve()
            if not image.is_file():
                raise FileNotFoundError(f"Image referenced by official split is missing: {rel}")
            if root not in image.parents:
                raise ValueError(f"Split CSV image path escapes data root: {rel}")
            mask_rel = row.get(" mask") or row.get("mask") or row.get("mask_path") or ""
            mask = (root / mask_rel).resolve() if mask_rel else None
            if mask and root not in mask.parents:
                raise ValueError(f"Split CSV mask path escapes data root: {mask_rel}")
            if mask and not mask.is_file():
                mask = None
            split_value = (row.get("split") or row.get("set") or "").lower()
            if split_value not in {"train", "test"}:
                raise ValueError(f"Unsupported official split value {split_value!r} for {rel}")
            split = split_value
            label_val = row.get("label") or row.get("is_anomaly") or ""
            label_normalized = str(label_val).lower()
            if label_normalized not in {"0", "1", "normal", "anomaly", "anomalous", "true", "false"}:
                raise ValueError(f"Unsupported annotation label {label_val!r} for {rel}")
            label = int(label_normalized in {"1", "anomaly", "anomalous", "true"})
            defect = row.get("defect") or row.get("anomaly_type") or ("normal" if not label else "unknown")
            digest = _sha(image)
            if digest in seen:
                raise ValueError(f"Duplicate image content in split: {image} duplicates {seen[digest]}")
            seen[digest] = str(image)
            records.append(Sample(image, mask, label, split, defect, digest))
    if not records:
        raise ValueError(f"No samples for category {category}; check the official split CSV and data root")
    return records


def validate_samples(samples: list[Sample]) -> dict[str, Any]:
    """Preflight image decoding and annotation alignment without dropping bad records."""
    errors: list[dict[str, str]] = []
    dimensions: dict[str, int] = {}
    source_modes: dict[str, int] = {}
    for sample in samples:
        try:
            with Image.open(sample.path) as image:
                image.verify()
            with Image.open(sample.path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                dimensions[f"{oriented.width}x{oriented.height}"] = dimensions.get(f"{oriented.width}x{oriented.height}", 0) + 1
                source_modes[oriented.mode] = source_modes.get(oriented.mode, 0) + 1
                image_size = oriented.size
        except (OSError, UnidentifiedImageError) as exc:
            errors.append({"path": str(sample.path), "error": f"unreadable image: {exc}"})
            continue
        if sample.mask:
            try:
                with Image.open(sample.mask) as mask:
                    mask.verify()
                with Image.open(sample.mask) as mask:
                    if mask.size != image_size:
                        errors.append({"path": str(sample.path), "error": f"mask dimensions {mask.size} do not match image dimensions {image_size}"})
            except (OSError, UnidentifiedImageError) as exc:
                errors.append({"path": str(sample.path), "error": f"unreadable mask {sample.mask}: {exc}"})
        elif sample.label:
            errors.append({"path": str(sample.path), "error": "anomalous sample has no mask"})
    return {"record_count": len(samples), "valid_count": max(0, len(samples) - len(errors)), "error_count": len(errors), "dimensions": dimensions, "source_modes": source_modes, "errors": errors, "image_channels": "converted to RGB by the shared preprocessing pipeline", "mask_interpretation": "all source mask values > 0 indicate anomaly"}


def download_visa(root: Path) -> dict[str, Any]:
    """Explicitly download the official archive with progress and safe extraction."""
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"Refusing to mix a new archive into non-empty data root: {root}")
    archive = root.parent / "VisA_20220922.tar.part"
    stage = Path(tempfile.mkdtemp(prefix=".industrial_visual_defect_detection-visa-", dir=root.parent))
    try:
        with urllib.request.urlopen(VISA_ARCHIVE_URL, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            required = (total * 2 if total else 4_000_000_000) + 1_000_000_000
            free = shutil.disk_usage(root.parent).free
            if free < required:
                raise OSError(f"Insufficient free disk space: need approximately {required / 1e9:.1f} GB, have {free / 1e9:.1f} GB")
            digest = hashlib.sha256()
            received = 0
            with archive.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk); digest.update(chunk); received += len(chunk)
                    if total: print(f"Downloaded {received / total * 100:5.1f}% ({received / 1e9:.2f}/{total / 1e9:.2f} GB)", end="\r", flush=True)
                    else: print(f"Downloaded {received / 1e9:.2f} GB", end="\r", flush=True)
        print()
        checksum = digest.hexdigest()
        with tarfile.open(archive, "r:*") as tar:
            members = tar.getmembers()
            if not members: raise ValueError("Official archive is empty")
            prefixes = {PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}
            expanded_size = sum(member.size for member in members if member.isfile())
            if shutil.disk_usage(root.parent).free < expanded_size + 1_000_000_000:
                raise OSError(f"Insufficient free space to safely extract archive ({expanded_size / 1e9:.1f} GB plus 1 GB reserve)")
            for member in members:
                target = (stage / member.name).resolve()
                if stage not in target.parents and target != stage:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Unsafe archive member type: {member.name}")
                member.mode = 0o755 if member.isdir() else 0o644
            tar.extractall(stage, members=members)
        single_prefix = next(iter(prefixes)) if len(prefixes) == 1 else None
        dataset_stage = stage / single_prefix if single_prefix and (stage / single_prefix).is_dir() else stage
        split_path = dataset_stage / "split_csv" / "1cls.csv"
        if not split_path.exists():
            split_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(VISA_SPLIT_URL, split_path)
        result = {"dataset": "VisA", "archive_url": VISA_ARCHIVE_URL, "archive_bytes": received, "computed_local_sha256": checksum, "checksum_is_official": False, "split_url": VISA_SPLIT_URL, "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(), "downloaded_at_utc": datetime.now(UTC).isoformat(), "archive_samples": len(members)}
        (dataset_stage / "download-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        root.rmdir()
        dataset_stage.rename(root)
    except Exception:
        archive.unlink(missing_ok=True)
        shutil.rmtree(stage, ignore_errors=True)
        raise
    archive.unlink(missing_ok=True)
    if stage.exists(): shutil.rmtree(stage, ignore_errors=True)
    return result


def create_manifest(samples: list[Sample], seed: int = 42, calibration_fraction: float = .12) -> dict[str, Any]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    normal_train = [s for s in samples if s.split == "train" and s.label == 0]
    if len(normal_train) < 2:
        raise ValueError("At least two official normal training images are required")
    shuffled = sorted(normal_train, key=lambda s: s.sha256)
    random.Random(seed).shuffle(shuffled)
    n_cal = max(1, round(len(shuffled) * calibration_fraction))
    calibration = {s.sha256 for s in shuffled[:n_cal]}
    rows = []
    for s in samples:
        if s.sha256 in calibration:
            part = "calibration"
        elif s.split == "train" and s.label == 0:
            part = "train"
        elif s.split == "test":
            part = "test"
        else:
            part = "excluded"
        rows.append({"path": str(s.path), "mask": str(s.mask) if s.mask else None, "label": s.label, "split": part, "official_split": s.split, "defect": s.defect, "sha256": s.sha256})
    ids: dict[str, str] = {}
    for row in rows:
        sha256 = str(row["sha256"])
        split = str(row["split"])
        old = ids.setdefault(sha256, split)
        if old != split:
            raise ValueError("Leakage: identical image content assigned across partitions")
    return {"protocol": "official 1cls split; normal-only deterministic calibration holdout", "seed": seed, "category": next(iter(samples)).path.parent.name, "records": rows}


class VisaDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int, str]]):
    def __init__(self, records: list[dict[str, Any]], split: str, size: int = 256):
        self.records = [r for r in records if r["split"] == split]
        self.transform = image_transform(size)
        self.mask_resize = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        rec = self.records[index]
        try:
            image = ImageOps.exif_transpose(Image.open(rec["path"])).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Unreadable image: {rec['path']}") from exc
        x = self.transform(image)
        if rec["mask"]:
            source_mask = ImageOps.exif_transpose(Image.open(rec["mask"])).convert("L")
            if source_mask.size != image.size:
                raise ValueError(f"Image/mask dimensions do not align: {rec['path']}")
            mask = self.mask_resize(source_mask)
            y = torch.from_numpy((np.asarray(mask) > 0).astype("float32")).unsqueeze(0)
        else:
            y = torch.zeros((1, x.shape[-2], x.shape[-1]), dtype=torch.float32)
        return x, y, int(rec["label"]), rec["path"]
