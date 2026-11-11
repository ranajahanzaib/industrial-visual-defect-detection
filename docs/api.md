# API

Start locally with `uvicorn industrial_visual_defect_detection.api.app:app --reload`. Set `INDUSTRIAL_VISUAL_DEFECT_DETECTION_RUNS` to the artifact directory. Routes are available under `/api`; selected older short routes remain aliases.

- `GET /api/health` and `/api/system` report service and local device/dataset status.
- `GET /api/runs` lists usable student and feature-memory runs; run metadata, history, metrics, and paginated prediction records are available under `/api/runs/{run_id}`.
- `POST /api/runs/{run_id}/predict` accepts multipart `file` up to 10 MiB and returns score, calibrated decision, stage timings, model identifier, and anomaly map. `POST /api/inference?run_id={run_id}` is an upload-oriented alias. `POST /api/inference/compare` scores one upload with two or three runs.
- `GET /api/datasets` and `/api/datasets/{category}/samples` expose VisA records. Run sample image/mask and generated report artifacts use contained, allowlisted routes.
- `GET /api/comparisons` lists paired offline evaluation reports and their comparison artifacts.

The API is local development software with no authentication. Keep it bound to localhost unless access controls are added.
