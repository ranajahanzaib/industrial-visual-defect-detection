import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  Cpu,
  Database,
  Gauge,
  Layers3,
  ShieldCheck,
} from "lucide-react";
import React from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import "./workbench.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const TrainingLossChart = React.lazy(() => import("./TrainingLossChart"));
type Run = { id: string; model: string };
type Prediction = {
  sample_id: string;
  category: string;
  label: number;
  score: number;
  threshold: number;
  prediction: number;
  outcome: string;
  latency_ms: number;
  image_url: string;
  mask_url: string | null;
  heatmap_url: string;
  overlay_url: string;
  predicted_mask_url: string | null;
};
const views = [
  { id: "inspection", label: "Inspection", icon: Activity },
  { id: "comparison", label: "Model comparison", icon: BarChart3 },
  { id: "experiments", label: "Training & experiments", icon: Cpu },
  { id: "evaluation", label: "Evaluation", icon: Gauge },
  { id: "datasets", label: "Dataset explorer", icon: Database },
] as const;
type View = (typeof views)[number]["id"];
async function getJson<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok)
    throw new Error(
      (await r.json().catch(() => ({ detail: "Request failed" }))).detail ||
        "Request failed",
    );
  return r.json();
}

export function App() {
  const [view, setView] = React.useState<View>("inspection");
  const [run, setRun] = React.useState("");
  const [file, setFile] = React.useState<File>();
  const [result, setResult] = React.useState<any>();
  const [error, setError] = React.useState("");
  const [selected, setSelected] = React.useState<Prediction>();
  const [opacity, setOpacity] = React.useState(0.68);
  const [datasetCategory, setDatasetCategory] = React.useState("pcb1");
  const [datasetPage, setDatasetPage] = React.useState(1);
  const [predictionPage, setPredictionPage] = React.useState(1);
  const [outcomeFilter, setOutcomeFilter] = React.useState("all");
  const [predictionSort, setPredictionSort] = React.useState("score_desc");
  const [exploreThreshold, setExploreThreshold] = React.useState<number>();
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: () => getJson<Run[]>("/api/runs"),
    retry: false,
  });
  const predictions = useQuery({
    queryKey: ["predictions", run, predictionPage],
    queryFn: () =>
      getJson<{ predictions: Prediction[]; total: number }>(
        `/api/runs/${encodeURIComponent(run)}/predictions?page=${predictionPage}&page_size=100`,
      ),
    enabled: !!run,
    retry: false,
  });
  const metrics = useQuery({
    queryKey: ["metrics", run],
    queryFn: () => getJson<any>(`/api/runs/${encodeURIComponent(run)}/metrics`),
    enabled: !!run,
    retry: false,
  });
  const metadata = useQuery({
    queryKey: ["metadata", run],
    queryFn: () => getJson<any>(`/api/runs/${encodeURIComponent(run)}`),
    enabled: !!run,
    retry: false,
  });
  const history = useQuery({
    queryKey: ["history", run],
    queryFn: () =>
      getJson<{ history: any[] }>(
        `/api/runs/${encodeURIComponent(run)}/history`,
      ),
    enabled: !!run,
    retry: false,
  });
  const dataset = useQuery({
    queryKey: ["datasets"],
    queryFn: () => getJson<any>("/api/datasets"),
    enabled: view === "datasets",
    retry: false,
  });
  const datasetSamples = useQuery({
    queryKey: ["dataset-samples", datasetCategory, datasetPage],
    queryFn: () =>
      getJson<any>(
        `/api/datasets/${datasetCategory}/samples?page=${datasetPage}&page_size=24`,
      ),
    enabled:
      view === "datasets" &&
      !!dataset.data?.categories.find((c: any) => c.id === datasetCategory)
        ?.available,
    retry: false,
  });
  const comparisons = useQuery({
    queryKey: ["comparisons"],
    queryFn: () => getJson<any[]>("/api/comparisons"),
    enabled: view === "comparison",
    retry: false,
  });
  async function inspect() {
    if (!file || !run) return;
    setError("");
    setResult(undefined);
    const body = new FormData();
    body.append("file", file);
    try {
      const r = await fetch(
        `${API}/api/inference?run_id=${encodeURIComponent(run)}`,
        { method: "POST", body },
      );
      if (!r.ok) throw new Error((await r.json()).detail || "Inference failed");
      const data = await r.json();
      setResult(data);
      setExploreThreshold(data.threshold ?? undefined);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Inference failed");
    }
  }
  const pagePredictions = (predictions.data?.predictions || [])
    .filter((p) => outcomeFilter === "all" || p.outcome === outcomeFilter)
    .sort((a, b) =>
      predictionSort === "score_asc"
        ? a.score - b.score
        : predictionSort === "margin"
          ? Math.abs(a.score - a.threshold) - Math.abs(b.score - b.threshold)
          : b.score - a.score,
    );
  const chosen =
    predictions.data?.predictions.find(
      (p) => p.sample_id === selected?.sample_id,
    ) || selected;
  const focusScore = result?.score ?? chosen?.score;
  const focusThreshold = result?.threshold ?? chosen?.threshold;
  const focusFlag =
    focusScore !== undefined &&
    focusThreshold !== undefined &&
    focusScore >= (exploreThreshold ?? focusThreshold);
  return (
    <main>
      <header>
        <div className="brand">
          <div className="logo">
            <Layers3 size={20} />
          </div>
          <div>
            <b>
              VISION<span>QC</span>
            </b>
            <small>QUALITY INTELLIGENCE / LOCAL WORKBENCH</small>
          </div>
        </div>
        <div className="top-status">
          <i className={runs.isError ? "offline" : ""} />
          {runs.isError ? "API OFFLINE" : "LOCAL SYSTEM"}
        </div>
      </header>
      <section className="intro">
        <div>
          <div className="eyebrow">
            INDUSTRIAL INSPECTION <span>•</span> RESEARCH WORKBENCH
          </div>
          <h1>
            See what the model
            <br />
            <em>sees differently.</em>
          </h1>
          <p>
            Train on normal products, surface unusual visual regions, and
            inspect where the detector succeeds or fails.
          </p>
        </div>
        <div className="intro-stat">
          <span>LOCAL MODEL RUNS</span>
          <b>{runs.data?.length || 0} AVAILABLE</b>
          <small>
            {runs.data?.length
              ? "Select a calibrated run to inspect"
              : "Prepare data, train, then calibrate a model"}
          </small>
        </div>
      </section>
      <nav>
        {views.map((v) => (
          <button
            key={v.id}
            className={view === v.id ? "active" : ""}
            onClick={() => setView(v.id)}
          >
            <v.icon size={15} />
            {v.label}
          </button>
        ))}
        <span>SESSION / LOCAL</span>
      </nav>
      {view === "inspection" && (
        <section className="grid">
          <article className="panel inspector">
            <div className="panel-head">
              <div>
                <label>01 / INSPECTION</label>
                <h2>
                  {chosen ? "Held-out sample review" : "Inspect an image"}
                </h2>
              </div>
              <span className="pill">
                {chosen ? "EVALUATED SAMPLE" : "UPLOAD INFERENCE"}
              </span>
            </div>
            <div className="controls">
              <label>
                MODEL RUN
                <select
                  value={run}
                  onChange={(e) => {
                    setRun(e.target.value);
                    setSelected(undefined);
                    setResult(undefined);
                    setPredictionPage(1);
                  }}
                >
                  <option value="">Select a run</option>
                  {runs.data?.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.id} · {r.model}
                    </option>
                  ))}
                </select>
              </label>
              {!chosen && (
                <>
                  <label className="upload">
                    IMAGE INPUT
                    <input
                      type="file"
                      accept="image/*"
                      onChange={(e) => setFile(e.target.files?.[0])}
                    />
                    <span>{file?.name || "Choose an image to inspect"}</span>
                  </label>
                  <button
                    className="inspect"
                    onClick={inspect}
                    disabled={!file || !run}
                  >
                    Run inspection <ArrowUpRight size={16} />
                  </button>
                </>
              )}
              {error && <div className="error">{error}</div>}
            </div>
            {chosen ? (
              <div className="sample-detail">
                <div className="sample-panes">
                  <figure>
                    <img
                      src={`${API}${chosen.image_url}`}
                      alt={`Original ${chosen.label ? "anomalous" : "normal"} sample`}
                    />
                    <figcaption>
                      ORIGINAL · GROUND TRUTH{" "}
                      {chosen.label ? "ANOMALOUS" : "NORMAL"}
                    </figcaption>
                  </figure>
                  <figure>
                    <img
                      src={`${API}${chosen.heatmap_url}`}
                      alt="Anomaly heatmap"
                    />
                    <figcaption>ANOMALY HEATMAP</figcaption>
                  </figure>
                  <figure>
                    <img
                      src={`${API}${chosen.overlay_url}`}
                      alt="Heatmap overlay on the original image"
                    />
                    <figcaption>HEATMAP OVERLAY</figcaption>
                  </figure>
                  <figure>
                    {chosen.mask_url ? (
                      <img
                        src={`${API}${chosen.mask_url}`}
                        alt="Ground-truth anomaly mask"
                      />
                    ) : (
                      <div className="empty">No stored mask</div>
                    )}
                    <figcaption>GROUND-TRUTH MASK</figcaption>
                  </figure>
                  <figure>
                    {chosen.predicted_mask_url ? (
                      <img
                        src={`${API}${chosen.predicted_mask_url}`}
                        alt="Predicted binary anomaly mask"
                      />
                    ) : (
                      <div className="empty">No calibrated mask</div>
                    )}
                    <figcaption>PREDICTED BINARY MASK</figcaption>
                  </figure>
                </div>
                <button
                  className="subtle"
                  onClick={() => setSelected(undefined)}
                >
                  Return to upload inspection
                </button>
              </div>
            ) : (
              <div className="image-view">
                {result && file ? (
                  <Heatmap
                    file={file}
                    map={result.anomaly_map}
                    threshold={result.pixel_threshold}
                    opacity={opacity}
                  />
                ) : (
                  <div className="empty">
                    <div className="target">⌖</div>
                    <b>Awaiting image input</b>
                    <span>
                      Select a calibrated run and upload an inspection image.
                    </span>
                  </div>
                )}
                {result && (
                  <div className="overlay">
                    FEATURE DISCREPANCY · SCORE {result.score.toFixed(4)}
                  </div>
                )}
              </div>
            )}
            {(result || chosen) && (
              <div className="controls lower">
                {result && (
                  <label>
                    OVERLAY OPACITY
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step=".02"
                      value={opacity}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                    />
                  </label>
                )}
                <label>
                  EXPLORATORY THRESHOLD
                  <input
                    type="range"
                    min="0"
                    max={Math.max(
                      (focusScore || 0) * 2,
                      (focusThreshold || 0) * 2,
                      0.001,
                    )}
                    step=".001"
                    value={exploreThreshold ?? focusThreshold ?? 0}
                    onChange={(e) =>
                      setExploreThreshold(Number(e.target.value))
                    }
                  />
                </label>
                <span className="explore-label">EXPLORATORY ONLY</span>
              </div>
            )}
            <div className="sample-list">
              <div className="panel-head">
                <div>
                  <label>HELD-OUT PREDICTIONS</label>
                  <h2>Review model outcomes</h2>
                </div>
                <span className="pill">
                  {predictions.data?.total || 0} SAMPLES
                </span>
              </div>
              <div className="samples">
                <div className="gallery-controls">
                  <select
                    aria-label="Filter predictions"
                    value={outcomeFilter}
                    onChange={(e) => setOutcomeFilter(e.target.value)}
                  >
                    <option value="all">All outcomes</option>
                    <option value="false_positive">False positives</option>
                    <option value="false_negative">False negatives</option>
                    <option value="true_positive">True positives</option>
                    <option value="true_negative">True negatives</option>
                  </select>
                  <select
                    aria-label="Sort predictions"
                    value={predictionSort}
                    onChange={(e) => setPredictionSort(e.target.value)}
                  >
                    <option value="score_desc">Highest score</option>
                    <option value="score_asc">Lowest score</option>
                    <option value="margin">Closest to threshold</option>
                  </select>
                </div>
                <div className="samples">
                  {pagePredictions.map((p) => (
                    <button
                      key={p.sample_id}
                      className={`sample ${selected?.sample_id === p.sample_id ? "selected" : ""}`}
                      onClick={() => {
                        setSelected(p);
                        setResult(undefined);
                        setExploreThreshold(p.threshold);
                      }}
                    >
                      <span className={`outcome ${p.outcome}`} />
                      <b>{p.sample_id}</b>
                      <small>{p.outcome.replace("_", " ").toUpperCase()}</small>
                      <strong>{p.score.toFixed(4)}</strong>
                    </button>
                  ))}
                  {predictions.isError && (
                    <p className="muted">
                      Evaluate this run to create its sample gallery.
                    </p>
                  )}
                </div>
                {predictions.data && (
                  <div className="pager">
                    <button
                      disabled={predictionPage <= 1}
                      onClick={() => setPredictionPage(predictionPage - 1)}
                    >
                      Previous
                    </button>
                    <span>
                      PAGE {predictionPage} · {predictions.data.total} TEST
                      SAMPLES
                    </span>
                    <button
                      disabled={predictionPage * 100 >= predictions.data.total}
                      onClick={() => setPredictionPage(predictionPage + 1)}
                    >
                      Next
                    </button>
                  </div>
                )}
              </div>
            </div>
          </article>
          <aside>
            <article className="panel verdict">
              <label>02 / DECISION</label>
              <div className={`verdict-icon ${focusFlag ? "warn" : ""}`}>
                {focusFlag ? <AlertTriangle /> : <ShieldCheck />}
              </div>
              <h2>
                {focusScore !== undefined
                  ? result?.decision === "uncalibrated"
                    ? "UNCALIBRATED"
                    : focusFlag
                      ? "FLAG"
                      : "PASS"
                  : "NO RESULT"}
              </h2>
              <p>
                {focusScore !== undefined
                  ? "Exploratory threshold changes do not modify saved benchmark results."
                  : "Run an inspection or select a test sample to see its decision."}
              </p>
              <Metric
                label="ANOMALY SCORE"
                value={focusScore?.toFixed(4) || "—"}
              />
              <Metric
                label="CALIBRATED THRESHOLD"
                value={focusThreshold?.toFixed(4) || "—"}
              />
              <Metric
                label="INFERENCE TIME"
                value={
                  result
                    ? `${result.latency_ms.toFixed(1)} ms`
                    : chosen
                      ? `${chosen.latency_ms.toFixed(1)} ms`
                      : "—"
                }
              />
            </article>
            <article className="note">
              <b>READING THE SIGNAL</b>
              <p>
                Scores are not probabilities. Heatmaps show model discrepancy;
                they do not identify the physical cause of a defect.
              </p>
            </article>
          </aside>
        </section>
      )}
      {view === "evaluation" && (
        <section className="panel page-panel">
          <div className="panel-head">
            <div>
              <label>03 / BENCHMARK</label>
              <h2>Evaluation results</h2>
            </div>
            <span className="pill">
              {metrics.data?.split?.toUpperCase() || "NO REPORT"}
            </span>
          </div>
          {metrics.data ? (
            <>
              <div className="metric-grid">
                <Metric
                  label="IMAGE AUROC"
                  value={format(metrics.data.auroc)}
                />
                <Metric
                  label="AVERAGE PRECISION"
                  value={format(metrics.data.average_precision)}
                />
                <Metric
                  label="FALSE-POSITIVE RATE"
                  value={format(metrics.data.false_positive_rate)}
                />
                <Metric label="RECALL" value={format(metrics.data.recall)} />
                <Metric
                  label="PIXEL AUROC"
                  value={format(metrics.data.pixel_metrics?.auroc)}
                />
                <Metric
                  label="PIXEL DICE"
                  value={format(metrics.data.pixel_metrics?.dice)}
                />
                <Metric
                  label="PIXEL IOU"
                  value={format(metrics.data.pixel_metrics?.iou)}
                />
                <Metric
                  label="SAMPLES"
                  value={String(metrics.data.sample_count)}
                />
              </div>
              <div className="charts">
                <img
                  alt="Image-level ROC curve"
                  onError={(e) => {
                    e.currentTarget.hidden = true;
                  }}
                  src={`${API}/api/artifacts/${run}/roc-curve.png`}
                />
                <img
                  alt="Image-level precision-recall curve"
                  onError={(e) => {
                    e.currentTarget.hidden = true;
                  }}
                  src={`${API}/api/artifacts/${run}/precision-recall.png`}
                />
                <img
                  alt="Image anomaly score distribution"
                  src={`${API}/api/artifacts/${run}/score-distribution.png`}
                />
                <img
                  alt="Image classification confusion matrix"
                  src={`${API}/api/artifacts/${run}/confusion-matrix.png`}
                />
              </div>
              <a
                className="download"
                href={`${API}/api/runs/${run}/metrics`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open metrics JSON ↗
              </a>
            </>
          ) : (
            <p className="muted">
              Choose a run with a completed held-out evaluation report.
            </p>
          )}
        </section>
      )}
      {view === "experiments" && (
        <section className="panel page-panel">
          <div className="panel-head">
            <div>
              <label>04 / EXPERIMENTS</label>
              <h2>Training and model details</h2>
            </div>
            <span className="pill">
              {metadata.data?.device || "SELECT A RUN"}
            </span>
          </div>
          {metadata.data ? (
            <>
              <div className="metric-grid">
                <Metric label="MODEL" value={metadata.data.model} />
                <Metric label="CATEGORY" value={metadata.data.category} />
                <Metric
                  label="TRAIN IMAGES"
                  value={String(metadata.data.train_images)}
                />
                <Metric
                  label="CALIBRATION IMAGES"
                  value={String(metadata.data.calibration_images)}
                />
                <Metric
                  label="TRAINABLE PARAMETERS"
                  value={
                    metadata.data.trainable_parameters?.toLocaleString() || "0"
                  }
                />
                <Metric
                  label="FIT TIME"
                  value={`${(metadata.data.training_seconds || metadata.data.fit_seconds || 0).toFixed(1)} s`}
                />
                <Metric
                  label="MEMORY BANK"
                  value={metadata.data.memory_points?.toLocaleString() || "—"}
                />
                <Metric
                  label="ARTIFACT SIZE"
                  value={formatBytes(metadata.data.memory_bytes)}
                />
              </div>
              {history.data?.history?.length ? (
                <div className="chart">
                  <React.Suspense
                    fallback={<p className="muted">Loading training curves…</p>}
                  >
                    <TrainingLossChart data={history.data.history} />
                  </React.Suspense>
                </div>
              ) : (
                <p className="muted">
                  Feature memory fitting does not use gradient training epochs.
                </p>
              )}
              <pre className="config">
                {JSON.stringify(metadata.data, null, 2)}
              </pre>
            </>
          ) : (
            <p className="muted">
              Choose a completed run to inspect its actual metadata and loss
              history.
            </p>
          )}
        </section>
      )}
      {view === "comparison" && (
        <section className="panel page-panel">
          <div className="panel-head">
            <div>
              <label>05 / MODEL COMPARISON</label>
              <h2>Paired held-out results</h2>
            </div>
            <span className="pill">
              {comparisons.data?.length || 0} COMPARISONS
            </span>
          </div>
          {comparisons.data?.length ? (
            comparisons.data.map((c: any) => (
              <div className="comparison" key={c.id}>
                <p>{c.protocol}</p>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>RUN</th>
                        <th>MODEL</th>
                        <th>AUROC</th>
                        <th>AVG PRECISION</th>
                        <th>PIXEL AUROC</th>
                        <th>PIXEL AP</th>
                        <th>FPR</th>
                        <th>RECALL</th>
                        <th>P50 MS</th>
                        <th>ARTIFACT</th>
                      </tr>
                    </thead>
                    <tbody>
                      {c.runs.map((r: any) => (
                        <tr key={r.run_id}>
                          <td>{r.run_id}</td>
                          <td>{r.model}</td>
                          <td>{format(r.auroc)}</td>
                          <td>{format(r.average_precision)}</td>
                          <td>{format(r.pixel_metrics?.auroc)}</td>
                          <td>{format(r.pixel_metrics?.average_precision)}</td>
                          <td>{format(r.false_positive_rate)}</td>
                          <td>{format(r.recall)}</td>
                          <td>{format(r.latency_p50_ms)}</td>
                          <td>{formatBytes(r.artifact_bytes)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="charts">
                  <img
                    alt="Paired image-level ROC curve"
                    onError={(e) => {
                      e.currentTarget.hidden = true;
                    }}
                    src={`${API}/api/comparisons/${c.id}/roc-curve.png`}
                  />{" "}
                  <img
                    alt="Paired image-level precision-recall curve"
                    onError={(e) => {
                      e.currentTarget.hidden = true;
                    }}
                    src={`${API}/api/comparisons/${c.id}/precision-recall.png`}
                  />{" "}
                  <img
                    alt="Compared anomaly score distributions"
                    src={`${API}/api/comparisons/${c.id}/score-distribution.png`}
                  />{" "}
                  <img
                    alt="Accuracy compared with inference latency"
                    src={`${API}/api/comparisons/${c.id}/accuracy-vs-latency.png`}
                  />{" "}
                  <img
                    alt="Accuracy compared with model artifact size"
                    src={`${API}/api/comparisons/${c.id}/accuracy-vs-storage.png`}
                  />{" "}
                </div>
                <div className="disagreement-gallery">
                  {c.disagreement?.visual_disagreement_artifacts?.map(
                    (path: string) => (
                      <img
                        key={path}
                        alt={`Model prediction disagreement for ${path.split("/").pop()}`}
                        src={`${API}/api/comparisons/${c.id}/disagreement/${path.split("/").pop()}`}
                      />
                    ),
                  )}
                </div>
              </div>
            ))
          ) : (
            <p className="muted">
              Evaluate two models on the same category, then run{" "}
              <code>
                industrial-visual-defect-detection compare --runs RUN_A RUN_B
              </code>
              .
            </p>
          )}
        </section>
      )}
      {view === "datasets" && (
        <section className="panel page-panel">
          <div className="panel-head">
            <div>
              <label>06 / DATASET</label>
              <h2>Amazon VisA explorer</h2>
            </div>
            <span className="pill">
              {dataset.data?.available
                ? "SPLIT FILE FOUND"
                : "DATA NOT INSTALLED"}
            </span>
          </div>
          {dataset.data ? (
            <>
              <p className="muted">
                Source: VisA · CC BY 4.0 · category-specific one-class test
                partitions. Opening this explorer does not download data.
              </p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>CATEGORY</th>
                      <th>LOCAL DATA</th>
                      <th>TRAIN NORMAL</th>
                      <th>TEST NORMAL</th>
                      <th>TEST ANOMALY</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.data.categories.map((c: any) => (
                      <tr key={c.id}>
                        <td>{c.id}</td>
                        <td>{c.available ? "AVAILABLE" : "NOT INSTALLED"}</td>
                        <td>{c.train_normal || 0}</td>
                        <td>{c.test_normal || 0}</td>
                        <td>{c.test_anomaly || 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <label className="category-select">
                BROWSE CATEGORY
                <select
                  value={datasetCategory}
                  onChange={(e) => {
                    setDatasetCategory(e.target.value);
                    setDatasetPage(1);
                  }}
                >
                  {dataset.data.categories.map((c: any) => (
                    <option key={c.id} value={c.id}>
                      {c.id}
                      {c.available ? "" : " · not installed"}
                    </option>
                  ))}
                </select>
              </label>
              <div className="dataset-gallery">
                {datasetSamples.data?.samples.map((sample: any) => (
                  <article key={sample.id}>
                    <img
                      src={`${API}${sample.image_url}`}
                      loading="lazy"
                      alt={`${sample.label ? "Anomalous" : "Normal"} VisA sample ${sample.id}`}
                    />{" "}
                    <div>
                      <b>{sample.id}</b>
                      <span>
                        {sample.official_split.toUpperCase()} ·{" "}
                        {sample.label ? "ANOMALY" : "NORMAL"}
                      </span>
                    </div>
                    {sample.mask_url && (
                      <img
                        className="mask-preview"
                        src={`${API}${sample.mask_url}`}
                        loading="lazy"
                        alt="Ground-truth mask"
                      />
                    )}
                  </article>
                ))}
              </div>
              {datasetSamples.data && (
                <div className="pager">
                  <button
                    disabled={datasetPage <= 1}
                    onClick={() => setDatasetPage(datasetPage - 1)}
                  >
                    Previous
                  </button>
                  <span>
                    PAGE {datasetPage} /{" "}
                    {Math.ceil(datasetSamples.data.total / 24)}
                  </span>
                  <button
                    disabled={datasetPage * 24 >= datasetSamples.data.total}
                    onClick={() => setDatasetPage(datasetPage + 1)}
                  >
                    Next
                  </button>
                </div>
              )}
            </>
          ) : (
            <p className="muted">
              Dataset status unavailable. Start the API and prepare VisA
              locally.
            </p>
          )}
        </section>
      )}
      <footer>
        <span>
          INDUSTRIAL VISUAL DEFECT DETECTION / LOCAL RESEARCH WORKBENCH
        </span>
        <span>Model outputs require category-specific evaluation.</span>
      </footer>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}
function format(v: any): string {
  return v === null || v === undefined
    ? "—"
    : typeof v === "number"
      ? v.toFixed(4)
      : String(v);
}
function formatBytes(v: any): string {
  return !v
    ? "—"
    : v > 1e6
      ? `${(v / 1e6).toFixed(1)} MB`
      : `${(v / 1e3).toFixed(0)} KB`;
}
function Heatmap({
  file,
  map,
  threshold,
  opacity,
}: {
  file: File;
  map: number[][];
  threshold: number | null;
  opacity: number;
}) {
  const ref = React.useRef<HTMLCanvasElement>(null);
  React.useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const image = new Image();
    const url = URL.createObjectURL(file);
    image.onload = () => {
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(image, 0, 0);
      const h = map.length,
        w = map[0]?.length || 0;
      if (!w) return;
      const scale = Math.max((threshold || 0) * 2, 1e-8);
      const layer = document.createElement("canvas");
      layer.width = w;
      layer.height = h;
      const lc = layer.getContext("2d")!,
        pixels = lc.createImageData(w, h);
      for (let y = 0; y < h; y++)
        for (let x = 0; x < w; x++) {
          const v = Math.max(0, Math.min(1, map[y][x] / scale)),
            i = (y * w + x) * 4;
          pixels.data[i] = 255;
          pixels.data[i + 1] = Math.round(105 * (1 - v));
          pixels.data[i + 2] = 20;
          pixels.data[i + 3] = Math.round(255 * v);
        }
      lc.putImageData(pixels, 0, 0);
      ctx.globalAlpha = opacity;
      ctx.drawImage(layer, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    };
    image.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file, map, threshold, opacity]);
  return (
    <canvas
      ref={ref}
      role="img"
      aria-label="Uploaded image with anomaly heatmap overlay"
      style={{ maxWidth: "100%", maxHeight: 510, objectFit: "contain" }}
    />
  );
}
const rootElement = document.getElementById("root");
if (rootElement) {
  createRoot(rootElement).render(
    <React.StrictMode>
      <QueryClientProvider client={new QueryClient()}>
        <App />
      </QueryClientProvider>
    </React.StrictMode>,
  );
}
