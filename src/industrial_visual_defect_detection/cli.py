from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from industrial_visual_defect_detection.data import (
    VISA_CATEGORIES,
    create_manifest,
    download_visa,
    validate_samples,
    visa_samples,
)
from industrial_visual_defect_detection.evaluation import (
    bootstrap_image_intervals,
    image_metrics,
    pixel_metrics,
)
from industrial_visual_defect_detection.inference import Detector
from industrial_visual_defect_detection.reporting import write_evaluation
from industrial_visual_defect_detection.smoke import smoke
from industrial_visual_defect_detection.training import (
    calibrate_patchcore,
    calibrate_student,
    fit_patchcore,
    train_student,
)
from industrial_visual_defect_detection.utils import atomic_json, choose_device


def _records(run: Path, root: Path, category: str, seed: int = 42, fraction: float = .12) -> list[dict[str, Any]]:
    manifest = run / "manifest.json"
    if manifest.exists(): return json.loads(manifest.read_text())["records"]
    m = create_manifest(visa_samples(root, category), seed, fraction)
    m["category"] = category
    atomic_json(manifest, m)
    return m["records"]


def _store_context(run: Path, root: Path, category: str, fraction: float, normal_quantile: float = .99, pixel_quantile: float = .995) -> None:
    path = run / "metadata.json"
    metadata = json.loads(path.read_text())
    split_csv = root / "split_csv" / "1cls.csv"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip() or None
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=False, capture_output=True, text=True).stdout.strip())
    metadata.update({"run_id": run.name, "category": category, "dataset": "Amazon VisA", "dataset_source": "official Amazon Science VisA one-class split", "dataset_root": str(root.resolve()), "split_csv_sha256": hashlib.sha256(split_csv.read_bytes()).hexdigest() if split_csv.is_file() else None, "calibration_fraction": fraction, "calibration_config": {"normal_quantile": normal_quantile, "pixel_normal_quantile": pixel_quantile}, "code_commit": commit, "working_tree_dirty_at_run": dirty, "created_at_utc": datetime.now(UTC).isoformat()})
    atomic_json(path, metadata)
    config = {"run_id": run.name, "model": metadata.get("model"), "category": category, "dataset_root": str(root.resolve()), "calibration_fraction": fraction, "calibration": metadata["calibration_config"], "size": metadata.get("size"), "score_top_fraction": metadata.get("score_top_fraction"), "seed": metadata.get("seed"), "device": metadata.get("device"), "training": metadata.get("training_config", metadata.get("fit_config"))}
    atomic_json(run / "config.json", config)
    if metadata.get("history"):
        (run / "history.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metadata["history"]))


def main() -> None:
    p = argparse.ArgumentParser(prog="industrial-visual-defect-detection", description="Local-first visual anomaly detection")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    q = sub.add_parser("smoke", help="run a synthetic software-path check without VisA")
    q.add_argument("--output", type=Path, default=Path("reports/synthetic-smoke")); q.add_argument("--seed", type=int, default=42); q.add_argument("--size", type=int, default=64); q.add_argument("--config", type=Path)
    data = sub.add_parser("data").add_subparsers(dest="data_cmd", required=True)
    for action in ("inspect", "prepare"):
        q = data.add_parser(action); q.add_argument("--dataset", choices=("visa",), default="visa"); q.add_argument("--data-root", type=Path, default=Path("data/VisA")); q.add_argument("--category", default="pcb1", choices=VISA_CATEGORIES); q.add_argument("--seed", type=int, default=42); q.add_argument("--calibration-fraction", type=float, default=.12); q.add_argument("--output", type=Path, default=Path("runs/data"))
    download = data.add_parser("download", help="explicitly download the official VisA archive"); download.add_argument("--data-root", type=Path, default=Path("data/VisA"))
    for name in ("train", "fit-baseline", "calibrate", "evaluate", "predict", "report", "reproduce"):
        q = sub.add_parser(name); q.add_argument("--category", default="pcb1", choices=VISA_CATEGORIES); q.add_argument("--data-root", type=Path, default=Path("data/VisA")); q.add_argument("--run", type=Path, default=Path("runs/pcb1-student")); q.add_argument("--device", default="auto")
        if name == "train":
            q.add_argument("--config", type=Path)
            q.add_argument("--model", choices=("student",), default="student")
            q.add_argument("--epochs", type=int, default=10); q.add_argument("--batch-size", type=int, default=8); q.add_argument("--size", type=int, default=256); q.add_argument("--lr", type=float, default=1e-3); q.add_argument("--optimizer", choices=("AdamW", "Adam", "SGD"), default="AdamW"); q.add_argument("--seed", type=int, default=42); q.add_argument("--workers", type=int, default=0); q.add_argument("--resume", action="store_true"); q.add_argument("--score-top-fraction", type=float, default=.01)
        if name == "fit-baseline": q.add_argument("--config", type=Path); q.add_argument("--model", choices=("patchcore",), default="patchcore"); q.add_argument("--size", type=int, default=256); q.add_argument("--batch-size", type=int, default=8); q.add_argument("--memory-bank-size", type=int, default=10000); q.add_argument("--sampling-ratio", type=float, default=1.0); q.add_argument("--seed", type=int, default=42); q.add_argument("--workers", type=int, default=0); q.add_argument("--score-top-fraction", type=float, default=.01)
        if name == "calibrate": q.add_argument("--normal-quantile", type=float); q.add_argument("--pixel-normal-quantile", type=float)
        if name == "predict": q.add_argument("--image", type=Path, required=True); q.add_argument("--output", type=Path)
        if name == "reproduce": q.add_argument("--output", type=Path)
        if name in ("evaluate", "report"): q.add_argument("--split", choices=("test",), default="test")
    compare = sub.add_parser("compare", help="paired comparison of evaluated runs")
    compare.add_argument("--runs", nargs="+", type=Path, required=True); compare.add_argument("--output", type=Path)
    serve = sub.add_parser("serve", help="start the local FastAPI inference service")
    serve.add_argument("--host", default="127.0.0.1"); serve.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.cmd in ("train", "fit-baseline") and args.config:
        import yaml
        config = yaml.safe_load(args.config.read_text())
        explicit = set(sys.argv[1:])
        dataset, training, model = config.get("dataset", {}), config.get("training", {}), config.get("model", {})
        mappings = {"--data-root": (dataset, "root", "data_root", Path), "--category": (dataset, "category", "category", str), "--epochs": (training, "epochs", "epochs", int), "--batch-size": (training, "batch_size", "batch_size", int), "--size": (model, "input_size", "size", int), "--lr": (training, "learning_rate", "lr", float), "--optimizer": (training, "optimizer", "optimizer", str), "--seed": (training, "seed", "seed", int), "--workers": (training, "workers", "workers", int), "--device": (training, "device", "device", str), "--run": (config.get("artifacts", {}), "directory", "run", Path), "--score-top-fraction": (config.get("inference", {}), "score_top_fraction", "score_top_fraction", float), "--memory-bank-size": (model, "memory_bank_size", "memory_bank_size", int), "--sampling-ratio": (model, "sampling_ratio", "sampling_ratio", float)}
        for flag, (section, key, target, cast) in mappings.items():
            if hasattr(args, target) and flag not in explicit and key in section:
                setattr(args, target, cast(section[key]))
        fraction = float(dataset.get("calibration_fraction", .12))
        normal_quantile = float(config.get("calibration", {}).get("normal_quantile", .99))
        pixel_normal_quantile = float(config.get("calibration", {}).get("pixel_normal_quantile", .995))
    else:
        fraction = .12
        normal_quantile, pixel_normal_quantile = .99, .995
    if args.cmd == "smoke":
        if args.config:
            import yaml
            config = yaml.safe_load(args.config.read_text())
            if config.get("dataset", {}).get("kind") != "synthetic": raise SystemExit("Smoke config must declare dataset.kind: synthetic")
            args.seed = int(config.get("training", {}).get("seed", args.seed))
            args.size = int(config.get("dataset", {}).get("image_size", args.size))
        print(json.dumps(smoke(args.output, args.seed, args.size), indent=2)); return
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("industrial_visual_defect_detection.api.app:app", host=args.host, port=args.port, reload=False); return
    if args.cmd == "compare":
        if len(args.runs) < 2: raise SystemExit("Provide at least two evaluated runs")
        bundles = []
        for run in args.runs:
            prediction_path = Path("reports") / run.name / "predictions.jsonl"
            evaluation_path, metadata_path = run / "evaluation.json", run / "metadata.json"
            if not prediction_path.is_file() or not evaluation_path.is_file() or not metadata_path.is_file(): raise SystemExit(f"Run {run} must be evaluated first (missing report artifacts)")
            predictions = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
            metadata = json.loads(metadata_path.read_text()); metrics = json.loads(evaluation_path.read_text())
            bundles.append((run, predictions, metadata, metrics))
        category_splits = {(b[3].get("category"), b[3].get("split")) for b in bundles}
        if len(category_splits) != 1: raise SystemExit("Runs must use the same category and evaluation split")
        if len({b[2].get("size") for b in bundles}) != 1 or len({b[2].get("preprocessing") for b in bundles}) != 1:
            raise SystemExit("Runs must use the same input resolution and preprocessing for a fair comparison")
        reference = {p["sample_id"]: int(p["label"]) for p in bundles[0][1]}
        for run, predictions, _, _ in bundles[1:]:
            current = {p["sample_id"]: int(p["label"]) for p in predictions}
            if current != reference: raise SystemExit(f"Run {run} did not evaluate on the identical held-out sample IDs and labels")
        outdir = args.output or Path("reports") / ("comparison-" + "-".join(b[0].name for b in bundles))
        outdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for run, predictions, metadata, metrics in bundles:
            artifact_bytes = sum(p.stat().st_size for p in run.iterdir() if p.is_file() and (p.suffix == ".safetensors" or p.name == "metadata.json"))
            latency_path = Path("reports") / run.name / "latency-summary.json"
            latency = json.loads(latency_path.read_text()) if latency_path.exists() else {}
            row = {"run_id": run.name, "model": metadata.get("model"), "category": metrics.get("category"), "sample_count": metrics.get("sample_count"), "auroc": metrics.get("auroc"), "average_precision": metrics.get("average_precision"), "pixel_metrics": metrics.get("pixel_metrics"), "false_positive_rate": metrics.get("false_positive_rate"), "recall": metrics.get("recall"), "latency_p50_ms": latency.get("p50_ms"), "latency_p95_ms": latency.get("p95_ms"), "throughput_images_per_second": latency.get("throughput_images_per_second"), "artifact_bytes": artifact_bytes, "fit_seconds": metadata.get("training_seconds", metadata.get("fit_seconds")), "trainable_parameters": metadata.get("trainable_parameters", 0), "pretrained_teacher": metadata.get("teacher")}
            rows.append(row)
        atomic_json(outdir / "comparison.json", {"protocol": "paired official test sample IDs; category-specific; continuous scores are model-specific", "runs": rows})
        with (outdir / "comparison.csv").open("w", newline="") as f:
            import csv
            fields = [k for k in rows[0] if k != "pixel_metrics"]
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows({k: v for k, v in row.items() if k in fields} for row in rows)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(5, 4))
        for run, pred, _, metric in bundles:
            y = [int(p["label"]) for p in pred]; s = [float(p["score"]) for p in pred]
            if len(set(y)) == 2:
                fpr, tpr, _ = roc_curve(y, s); ax.plot(fpr, tpr, label=f"{run.name} · {metric.get('auroc'):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "roc-curve.png", dpi=140); plt.close(fig)
        from sklearn.metrics import precision_recall_curve
        fig, ax = plt.subplots(figsize=(5, 4))
        for run, pred, _, metric in bundles:
            y = [int(p["label"]) for p in pred]; s = [float(p["score"]) for p in pred]
            if len(set(y)) == 2:
                precision, recall, _ = precision_recall_curve(y, s); ax.plot(recall, precision, label=f"{run.name} · {metric.get('average_precision'):.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "precision-recall.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for row in rows:
            if row["latency_p50_ms"] is not None and row["auroc"] is not None: ax.scatter(row["latency_p50_ms"], row["auroc"], label=row["run_id"])
        ax.set(xlabel="Median latency (ms)", ylabel="Image AUROC", ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "accuracy-vs-latency.png", dpi=140); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for row in rows:
            if row["artifact_bytes"] and row["auroc"] is not None: ax.scatter(row["artifact_bytes"] / 1e6, row["auroc"], label=row["run_id"])
        ax.set(xlabel="Model artifact size (MB)", ylabel="Image AUROC", ylim=(0, 1)); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "accuracy-vs-storage.png", dpi=140); plt.close(fig)
        fig, axes = plt.subplots(1, len(bundles), figsize=(5 * len(bundles), 4), squeeze=False)
        for axis, (run, pred, _, _) in zip(axes[0], bundles):
            for cls, label, color in ((0, "Normal", "#44a66c"), (1, "Anomaly", "#dc8f32")):
                values = [float(p["score"]) for p in pred if int(p["label"]) == cls]
                if values: axis.hist(values, bins=20, alpha=.55, label=label, color=color)
            axis.set_title(run.name); axis.set_xlabel("Model-specific score"); axis.set_ylabel("Images"); axis.legend()
        fig.tight_layout(); fig.savefig(outdir / "score-distribution.png", dpi=140); plt.close(fig)
        disagreement = [sid for sid, label in reference.items() if len({int(next(p["prediction"] for p in bundle[1] if p["sample_id"] == sid)) for bundle in bundles}) > 1]
        galleries = []
        if disagreement:
            gallery_dir = outdir / "visual-disagreement"; gallery_dir.mkdir(exist_ok=True)
            for sid in disagreement[:8]:
                fig, axes = plt.subplots(1, len(bundles) + 1, figsize=(4 * (len(bundles) + 1), 4))
                first = next(p for p in bundles[0][1] if p["sample_id"] == sid)
                with Image.open(first["image_path"]) as image: axes[0].imshow(image.convert("RGB"))
                axes[0].set_title(f"Original · y={reference[sid]}"); axes[0].axis("off")
                for index, (run, pred, metadata, _) in enumerate(bundles, start=1):
                    entry = next(p for p in pred if p["sample_id"] == sid)
                    with Image.open(entry["image_path"]) as image: axes[index].imshow(image.convert("RGB"))
                    amap = np.load(entry["heatmap_path"])
                    scale = max(float(metadata.get("pixel_threshold") or 1.) * 2, 1e-8)
                    axes[index].imshow(amap, cmap="inferno", alpha=np.clip(amap / scale, 0, 1) * .65)
                    axes[index].set_title(f"{run.name}\nscore={entry['score']:.3f}, pred={entry['prediction']}"); axes[index].axis("off")
                fig.tight_layout(); gallery_path = gallery_dir / f"{sid}.png"; fig.savefig(gallery_path, dpi=120); plt.close(fig); galleries.append(str(gallery_path))
        atomic_json(outdir / "disagreement.json", {"samples": [{"sample_id": sid, "label": label, "predictions": {b[0].name: next(int(p["prediction"]) for p in b[1] if p["sample_id"] == sid) for b in bundles}} for sid, label in reference.items()], "visual_disagreement_artifacts": galleries})
        table_html = "<table><tr><th>Run</th><th>Model</th><th>AUROC</th><th>AP</th><th>FPR</th><th>Recall</th><th>Median ms</th><th>Artifact MB</th></tr>" + "".join(f"<tr><td>{row['run_id']}</td><td>{row['model']}</td><td>{row['auroc']}</td><td>{row['average_precision']}</td><td>{row['false_positive_rate']}</td><td>{row['recall']}</td><td>{row['latency_p50_ms']}</td><td>{row['artifact_bytes']/1e6:.2f}</td></tr>" for row in rows) + "</table>"
        (outdir / "report.html").write_text(f"<!doctype html><meta charset='utf-8'><title>Industrial Visual Defect Detection paired comparison</title><h1>Paired model comparison</h1><p>{next(iter(category_splits))} · same held-out image IDs · raw scores are model-specific</p>{table_html}<h2>ROC</h2><img src='roc-curve.png'><h2>Precision–recall</h2><img src='precision-recall.png'><h2>Score distributions · model-specific scales</h2><img src='score-distribution.png'><h2>Accuracy vs latency</h2><img src='accuracy-vs-latency.png'><h2>Accuracy vs storage</h2><img src='accuracy-vs-storage.png'><h2>Visual disagreements</h2>" + "".join(f"<img src='visual-disagreement/{Path(g).name}'>" for g in galleries))
        print(json.dumps({"comparison_dir": str(outdir), "runs": rows}, indent=2)); return
    if args.cmd == "doctor":
        import cv2
        import torchvision
        try:
            import psutil
            available_memory = int(psutil.virtual_memory().available)
        except ImportError:
            try: available_memory = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (ValueError, OSError, AttributeError): available_memory = None
        print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__, "opencv": cv2.__version__, "architecture": platform.machine(), "selected_device": str(choose_device()), "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()), "dataset_root": str(Path("data/VisA").resolve()), "dataset_present": Path("data/VisA/split_csv/1cls.csv").is_file(), "pretrained_resnet18_cached": any(Path(torch.hub.get_dir(), "checkpoints").glob("resnet18-f37072fd.pth")), "available_memory_bytes": available_memory}, indent=2)); return
    if args.cmd == "data":
        if args.data_cmd == "download":
            print(json.dumps(download_visa(args.data_root), indent=2)); return
        samples = visa_samples(args.data_root, args.category)
        integrity = validate_samples(samples)
        manifest = create_manifest(samples, args.seed, args.calibration_fraction)
        manifest["category"] = args.category
        counts = {s: sum(r["split"] == s for r in manifest["records"]) for s in ("train", "calibration", "test", "excluded")}
        print(json.dumps({"category": args.category, "samples": len(samples), "counts": counts, "integrity": integrity}, indent=2))
        if args.data_cmd == "prepare":
            atomic_json(args.output / args.category / "integrity.json", integrity)
            if integrity["error_count"]: raise SystemExit(f"Dataset integrity check found {integrity['error_count']} error(s); see {args.output / args.category / 'integrity.json'}")
            atomic_json(args.output / args.category / "manifest.json", manifest); print(f"Manifest: {args.output / args.category / 'manifest.json'}")
        return
    if args.cmd == "train":
        records = _records(args.run, args.data_root, args.category, args.seed, fraction)
        train_student(records, args.run, size=args.size, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, optimizer_name=args.optimizer, seed=args.seed, device_name=args.device, workers=args.workers, resume=args.resume, score_top_fraction=args.score_top_fraction)
        _store_context(args.run, args.data_root, args.category, fraction, normal_quantile, pixel_normal_quantile); return
    if args.cmd == "calibrate":
        records = _records(args.run, args.data_root, args.category)
        meta = json.loads((args.run / "metadata.json").read_text())
        calibration_cfg = meta.get("calibration_config", {})
        normal_q = args.normal_quantile if args.normal_quantile is not None else float(calibration_cfg.get("normal_quantile", .99))
        pixel_q = args.pixel_normal_quantile if args.pixel_normal_quantile is not None else float(calibration_cfg.get("pixel_normal_quantile", .995))
        if (args.run / "memory.safetensors").exists():
            threshold = calibrate_patchcore(args.run, records, normal_q, device_name=args.device, pixel_quantile=pixel_q)
        else:
            threshold = calibrate_student(args.run, records, normal_q, device_name=args.device, pixel_quantile=pixel_q)
        run_config_path = args.run / "config.json"
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text()); run_config["calibration"] = {"normal_quantile": normal_q, "pixel_normal_quantile": pixel_q}; atomic_json(run_config_path, run_config)
        print(f"Normal calibration threshold: {threshold:.6f}"); return
    if args.cmd == "predict":
        result = Detector(args.run, args.device).predict(Image.open(args.image))
        amap = result.pop("anomaly_map")
        if args.output:
            np.save(args.output, amap)
        print(json.dumps(result, indent=2)); return
    if args.cmd in ("evaluate", "report"):
        records = _records(args.run, args.data_root, args.category)
        detector = Detector(args.run, args.device)
        if detector.metadata.get("threshold") is None: raise SystemExit("Run `industrial-visual-defect-detection calibrate` before evaluation so thresholded metrics use normal-only calibration")
        labels, scores, predictions = [], [], []
        pixel_targets, pixel_scores = [], []
        from industrial_visual_defect_detection.data import VisaDataset
        test_rows = [row for row in records if row["split"] == args.split]
        test_dataset = VisaDataset(records, args.split, int(detector.metadata["size"]))
        for _ in range(min(3, len(test_rows))):
            with Image.open(test_rows[0]["path"]) as source: detector.predict(source.copy())
        report_dir = Path("reports") / args.run.name
        artifact_dir = report_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for item_index, row in enumerate(test_rows):
            with Image.open(row["path"]) as source: out = detector.predict(source.copy())
            labels.append(row["label"]); scores.append(out["score"])
            predicted = int(out["score"] >= detector.metadata["threshold"])
            outcome = ("true_positive" if predicted else "false_negative") if row["label"] else ("false_positive" if predicted else "true_negative")
            map_path = artifact_dir / f"{row['sha256'][:16]}.npy"
            np.save(map_path, out["anomaly_map"])
            binary_path = None
            if detector.metadata.get("pixel_threshold") is not None:
                binary_path = artifact_dir / f"{row['sha256'][:16]}-predicted-mask.png"
                Image.fromarray((out["anomaly_map"] >= detector.metadata["pixel_threshold"]).astype(np.uint8) * 255).save(binary_path)
            predictions.append({"sample_id": row["sha256"][:16], "category": args.category, "label": row["label"], "score": out["score"], "threshold": detector.metadata["threshold"], "pixel_threshold": detector.metadata.get("pixel_threshold"), "prediction": predicted, "outcome": outcome, "latency_ms": out["latency_ms"], "preprocessing_ms": out["preprocessing_ms"], "teacher_ms": out["teacher_ms"], "student_ms": out["student_ms"], "map_generation_ms": out["map_generation_ms"], "patchcore_inference_ms": out["patchcore_inference_ms"], "image_path": row["path"], "mask_path": row["mask"], "heatmap_path": str(map_path), "predicted_mask_path": str(binary_path) if binary_path else None})
            if row["label"] == 1 and row["mask"] is None: continue
            _, mask, _, _ = test_dataset[item_index]
            pixel_targets.append(mask.numpy()[0].astype(bool))
            pixel_scores.append(out["anomaly_map"])
        result = image_metrics(labels, scores, detector.metadata.get("threshold", float("inf")))
        result["confidence_intervals"] = bootstrap_image_intervals(labels, scores)
        pixel_threshold = detector.metadata.get("pixel_threshold")
        result["pixel_metrics"] = pixel_metrics(np.stack(pixel_targets), np.stack(pixel_scores), pixel_threshold) if pixel_threshold is not None and pixel_targets else None
        result.update({"category": args.category, "split": args.split, "sample_count": len(labels), "pixel_samples": len(pixel_targets), "pixel_samples_without_mask_excluded": sum(r["split"] == args.split and r["label"] == 1 and r["mask"] is None for r in records), "protocol": "official VisA test split; image and pixel thresholds calibrated independently from normal-only calibration"})
        atomic_json(args.run / "evaluation.json", result)
        report_dir = write_evaluation(args.run, predictions, result)
        print(json.dumps({"metrics": result, "report_dir": str(report_dir)}, indent=2)); return
    if args.cmd == "fit-baseline":
        records = _records(args.run, args.data_root, args.category, args.seed)
        fit_patchcore(records, args.run, size=args.size, batch_size=args.batch_size, max_points=args.memory_bank_size, sampling_ratio=args.sampling_ratio, seed=args.seed, device_name=args.device, workers=args.workers, score_top_fraction=args.score_top_fraction)
        _store_context(args.run, args.data_root, args.category, fraction, normal_quantile, pixel_normal_quantile)
        print(f"Saved PatchCore-style memory bank to {args.run / 'memory.safetensors'}"); return
    if args.cmd == "reproduce":
        source_run = args.run
        meta_path = source_run / "metadata.json"
        if not meta_path.is_file(): raise SystemExit(f"Run metadata not found: {meta_path}")
        metadata = json.loads(meta_path.read_text())
        manifest_path = source_run / "manifest.json"
        if not manifest_path.is_file(): raise SystemExit("Reproduction requires the original run's split manifest")
        records = json.loads(manifest_path.read_text())["records"]
        args.run = args.output or Path("runs") / f"{metadata.get('model','run')}-{metadata['category']}-reproduction-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        args.run.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, args.run / "manifest.json")
        if metadata.get("model") == "student-distillation":
            cfg = metadata.get("training_config", {})
            train_student(records, args.run, size=int(metadata["size"]), epochs=int(cfg.get("epochs", 10)), batch_size=int(cfg.get("batch_size", 8)), lr=float(cfg.get("learning_rate", .001)), optimizer_name=str(cfg.get("optimizer", "AdamW")), seed=int(metadata.get("seed", 42)), device_name=args.device, workers=int(cfg.get("workers", 0)), score_top_fraction=float(metadata.get("score_top_fraction", .01)))
        elif str(metadata.get("model", "")).startswith("patchcore"):
            cfg = metadata.get("fit_config", {})
            fit_patchcore(records, args.run, size=int(metadata["size"]), batch_size=int(cfg.get("batch_size", 8)), max_points=int(cfg.get("memory_bank_size", 10000)), sampling_ratio=float(cfg.get("sampling_ratio", 1.)), seed=int(metadata.get("seed", 42)), device_name=args.device, workers=int(cfg.get("workers", 0)), score_top_fraction=float(metadata.get("score_top_fraction", .01)))
        else: raise SystemExit(f"Unsupported model type in run metadata: {metadata.get('model')}")
        calibration_cfg = metadata.get("calibration_config", {})
        normal_q, pixel_q = float(calibration_cfg.get("normal_quantile", metadata.get("normal_quantile", .99))), float(calibration_cfg.get("pixel_normal_quantile", metadata.get("pixel_normal_quantile", .995)))
        _store_context(args.run, Path(metadata["dataset_root"]), metadata["category"], float(metadata.get("calibration_fraction", .12)), normal_q, pixel_q)
        if (args.run / "memory.safetensors").exists(): calibrate_patchcore(args.run, records, normal_q, device_name=args.device, pixel_quantile=pixel_q)
        else: calibrate_student(args.run, records, normal_q, device_name=args.device, pixel_quantile=pixel_q)
        subprocess.run([sys.executable, "-m", "industrial_visual_defect_detection.cli", "evaluate", "--run", str(args.run), "--data-root", metadata["dataset_root"], "--category", metadata["category"], "--device", args.device], check=True)
        print(f"Reproduced training, calibration, and evaluation into {args.run}"); return


if __name__ == "__main__": main()
