import { useEffect, useState } from "react";
import axios from "axios";
import QRCode from "qrcode";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const DETECTION_CONFIG = {
  image: { endpoint: "/detect-deepfake-image", title: "Image Detection" },
  video: { endpoint: "/detect-deepfake-video", title: "Video Detection" },
  audio: { endpoint: "/detect-deepfake-audio", title: "Audio Detection" },
};

export default function App() {
  const [files, setFiles] = useState({
    image: null,
    video: null,
    audio: null,
  });
  const [results, setResults] = useState({
    image: null,
    video: null,
    audio: null,
  });
  const [loading, setLoading] = useState({
    image: false,
    video: false,
    audio: false,
  });
  const [fusion, setFusion] = useState(null);
  const [logs, setLogs] = useState([]);

  // QR upload state
  const [qrModalOpen, setQrModalOpen] = useState(false);
  const [qrTargetType, setQrTargetType] = useState("image"); // image | video | audio
  const [qrBaseUrl, setQrBaseUrl] = useState(() => localStorage.getItem("qrBaseUrl") || API_BASE);
  const [qrCreating, setQrCreating] = useState(false);
  const [qrSession, setQrSession] = useState(null); // { session_id, upload_page_url, qr_png_url, ... }
  const [qrError, setQrError] = useState(null);

  // Local previews for selected files (do not upload again; just create temporary URLs).
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState(null);

  useEffect(() => {
    if (!files.image) {
      setImagePreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(files.image);
    setImagePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [files.image]);

  useEffect(() => {
    if (!files.video) {
      setVideoPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(files.video);
    setVideoPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [files.video]);

  useEffect(() => {
    if (!files.audio) {
      setAudioPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(files.audio);
    setAudioPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [files.audio]);

  const addLog = (source, status, message, payload = null) => {
    const entry = {
      id: `${Date.now()}-${Math.random()}`,
      time: new Date().toLocaleTimeString(),
      source,
      status,
      message,
      payload,
    };
    setLogs((prev) => [entry, ...prev].slice(0, 20));
  };

  const setSelectedFile = (type, file) => {
    setFiles((prev) => ({ ...prev, [type]: file ?? null }));
    setResults((prev) => ({ ...prev, [type]: null }));
    setFusion(null);
  };

  const openQrFor = (type) => {
    setQrTargetType(type);
    setQrError(null);
    setQrSession(null);
    setQrModalOpen(true);
  };

  const createQrSession = async () => {
    try {
      setQrCreating(true);
      setQrError(null);
      localStorage.setItem("qrBaseUrl", qrBaseUrl);
      const res = await axios.post(`${API_BASE}/qr-upload/sessions`, {
        base_url: qrBaseUrl,
        ttl_seconds: 10 * 60,
      });
      setQrSession(res.data);
    } catch (e) {
      console.error(e);
      setQrError(e?.response?.data?.detail || e?.message || "Failed to create QR session.");
    } finally {
      setQrCreating(false);
    }
  };

  useEffect(() => {
    if (!qrModalOpen || !qrSession?.session_id) return;
    let alive = true;
    const sessionId = qrSession.session_id;

    const tick = async () => {
      try {
        const st = await axios.get(`${API_BASE}/qr-upload/sessions/${sessionId}/status`);
        if (!alive) return;
        if (st.data?.state === "uploaded") {
          // Fetch the uploaded file, convert to File, and set into the chosen type
          const dl = await axios.get(`${API_BASE}/qr-upload/sessions/${sessionId}/download`, {
            responseType: "blob",
          });
          if (!alive) return;
          const filename =
            st.data?.filename ||
            dl?.headers?.["content-disposition"]?.split("filename=")?.[1]?.replaceAll('"', "") ||
            "upload";
          const blob = dl.data;
          const f = new File([blob], filename, { type: blob.type || "application/octet-stream" });
          setSelectedFile(qrTargetType, f);
          setQrModalOpen(false);
          addLog("qr", "success", `QR upload received for ${qrTargetType}`, {
            filename,
            size_bytes: st.data?.size_bytes ?? null,
          });
        }
      } catch (err) {
        // ignore transient polling errors
      }
    };

    const interval = setInterval(tick, 1200);
    tick();
    return () => {
      alive = false;
      clearInterval(interval);
    };
  }, [qrModalOpen, qrSession?.session_id, qrTargetType]);

  const runDetection = async (type) => {
    const file = files[type];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = DETECTION_CONFIG[type]?.endpoint;
    if (!endpoint) return;

    try {
      setLoading((prev) => ({ ...prev, [type]: true }));
      addLog(type, "request", `Calling ${endpoint}`);
      const res = await axios.post(`${API_BASE}${endpoint}`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        // Video inference/model warm-up can take longer on first run.
        timeout: type === "video" ? 600000 : 90000,
      });

      setResults((prev) => ({ ...prev, [type]: res.data }));
      setFusion(null);
      addLog(type, "success", "Detection completed", res.data);
    } catch (e) {
      console.error(e);
      const backendError =
        e?.response?.data ??
        (e?.code === "ECONNABORTED"
          ? "Request timed out. Backend is taking too long."
          : e?.message ?? "Unknown error");
      addLog(type, "error", "Detection failed", backendError);
      if (e?.message === "Network Error") {
        alert("Network Error: backend restarted/crashed or is unreachable. Try again in a few seconds.");
      } else {
        alert("Detection failed. Check logs panel for backend response.");
      }
    } finally {
      setLoading((prev) => ({ ...prev, [type]: false }));
    }
  };

  const runFusion = () => {
    try {
      addLog("fusion", "request", "Fusion run clicked");

      // Backend returns JSON, and depending on the runtime it may come back as
      // either number or numeric-string. Normalize to a real number here.
      const availableResults = [results.image, results.video, results.audio]
        .map((r) => {
          if (!r || !r.label) return null;

          const confidence =
            typeof r.confidence === "number" ? r.confidence : parseFloat(r.confidence);
          if (Number.isNaN(confidence)) return null;

          return { ...r, confidence };
        })
        .filter(Boolean);

      if (availableResults.length === 0) {
        addLog("fusion", "warning", "Run at least one detection first");
        alert("Run at least one detection (image, video, or audio) first.");
        return;
      }

      console.log("Fusion inputs:", availableResults);

      const avgConfidence =
        availableResults.reduce((sum, r) => sum + r.confidence, 0) / availableResults.length;

      const labels = availableResults.map((r) => String(r.label).toUpperCase());

      const fakeVotes = labels.filter((l) => l.includes("FAKE")).length;
      const realVotes = labels.filter((l) => l.includes("REAL")).length;

      const finalLabel =
        fakeVotes === 0 && realVotes === 0
          ? labels[0] || "UNKNOWN"
          : fakeVotes >= realVotes
          ? "FAKE"
          : "REAL";

      const fusionResult = {
        label: finalLabel,
        confidence: avgConfidence,
      };

      console.log("Fusion result:", fusionResult);
      setFusion(fusionResult);
      addLog("fusion", "success", "Fusion completed", fusionResult);
    } catch (err) {
      console.error("Fusion error:", err);
      addLog("fusion", "error", "Fusion failed", err?.message ?? String(err));
      alert("Something went wrong while fusing results. Check console for details.");
    }
  };

  const { combinedSummary, pieData, modelTableRows } = (() => {
    const acc = {
      success: 0,
      error: 0,
      image: 0,
      video: 0,
      audio: 0,
      fusion: 0,
      fake: 0,
      real: 0,
      unknown: 0,
      confidenceSum: 0,
      confidenceCount: 0,
      blurSum: 0,
      blurCount: 0,
    };
    const byModel = { image: [], video: [], audio: [] };

    for (const log of logs) {
      const status = String(log.status || "").toLowerCase();
      const source = String(log.source || "").toLowerCase();
      const label = String(log?.payload?.label ?? "").toUpperCase();
      const p = log?.payload;

      if (status === "success") acc.success += 1;
      else if (status === "error") acc.error += 1;

      if (source === "image") acc.image += 1;
      else if (source === "video") acc.video += 1;
      else if (source === "audio") acc.audio += 1;
      else if (source === "fusion") acc.fusion += 1;

      if (status === "success") {
        if (label.includes("FAKE")) acc.fake += 1;
        else if (label.includes("REAL")) acc.real += 1;
        else if (label) acc.unknown += 1;

        const conf = typeof p?.confidence === "number" ? p.confidence : parseFloat(p?.confidence);
        if (Number.isFinite(conf)) {
          acc.confidenceSum += conf;
          acc.confidenceCount += 1;
        }
        const blur =
          typeof p?.blur_percent === "number" ? p.blur_percent : parseFloat(p?.blur_percent);
        if (Number.isFinite(blur)) {
          acc.blurSum += blur;
          acc.blurCount += 1;
        }

        if (["image", "video", "audio"].includes(source)) {
          byModel[source].push({
            label: p?.label ?? "—",
            confidence: Number.isFinite(conf) ? conf : null,
            blur_percent: Number.isFinite(blur) ? blur : null,
            frames_analyzed: p?.frames_analyzed ?? null,
          });
        }
      }
    }

    const avgConfPct =
      acc.confidenceCount > 0 ? (acc.confidenceSum / acc.confidenceCount) * 100 : 0;
    const avgBlurPct = acc.blurCount > 0 ? acc.blurSum / acc.blurCount : 0;

    const pieData = [
      { label: "Success", value: acc.success, color: "#4bf6a9" },
      { label: "Error", value: acc.error, color: "#ff6f9f" },
      { label: "Image", value: acc.image, color: "#7bd6ff" },
      { label: "Video", value: acc.video, color: "#d38bff" },
      { label: "Audio", value: acc.audio, color: "#85f5be" },
      { label: "Fusion", value: acc.fusion, color: "#ffb679" },
      { label: "FAKE", value: acc.fake, color: "#ff4d7e" },
      { label: "REAL", value: acc.real, color: "#36d39b" },
      { label: "UNKNOWN", value: acc.unknown, color: "#8b95bf" },
      ...(acc.confidenceCount > 0
        ? [
            {
              label: "Confidence (avg)",
              value: 1,
              displayPct: Math.round(avgConfPct * 10) / 10,
              color: "#a8e6ff",
            },
          ]
        : []),
      ...(acc.blurCount > 0
        ? [
            {
              label: "Blur (avg)",
              value: 1,
              displayPct: Math.round(avgBlurPct * 10) / 10,
              color: "#c9a8ff",
            },
          ]
        : []),
    ].filter((d) => d.value > 0);

    const modelTableRows = ["image", "video", "audio"].map((key) => {
      const arr = byModel[key];
      const last = arr[arr.length - 1];
      const confVals = arr
        .map((r) => r.confidence)
        .filter((c) => c != null);
      const blurVals = arr
        .map((r) => r.blur_percent)
        .filter((b) => b != null);
      return {
        model: key.charAt(0).toUpperCase() + key.slice(1),
        runs: arr.length,
        lastLabel: last?.label ?? "—",
        avgConfidencePct:
          confVals.length > 0
            ? ((confVals.reduce((a, b) => a + b, 0) / confVals.length) * 100).toFixed(1) + "%"
            : "—",
        avgBlurPct:
          blurVals.length > 0
            ? (blurVals.reduce((a, b) => a + b, 0) / blurVals.length).toFixed(1) + "%"
            : "—",
        lastFrames: last?.frames_analyzed ?? "—",
      };
    });

    return { combinedSummary: acc, pieData, modelTableRows };
  })();

  return (
    <div className="app-shell">
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-grid" />

      <SideInputPreview
        variant="left"
        imageUrl={imagePreviewUrl}
        videoUrl={videoPreviewUrl}
        audioUrl={audioPreviewUrl}
        hasImage={!!files.image}
        onOpenQr={() => openQrFor("image")}
      />
      <SideInputPreview
        variant="right"
        imageUrl={imagePreviewUrl}
        videoUrl={videoPreviewUrl}
        audioUrl={audioPreviewUrl}
        hasVideo={!!files.video}
        hasAudio={!!files.audio}
        onOpenQr={(t) => openQrFor(t)}
      />

      {qrModalOpen && (
        <QrUploadModal
          targetType={qrTargetType}
          baseUrl={qrBaseUrl}
          setBaseUrl={setQrBaseUrl}
          creating={qrCreating}
          session={qrSession}
          error={qrError}
          onClose={() => setQrModalOpen(false)}
          onCreate={createQrSession}
        />
      )}

      <header className="app-header">
        <p className="badge">AI Security Suite</p>
        <h1>Multimodal Deepfake Detection</h1>
        <p className="subtitle">
          Run individual image, video, and audio checks, then fuse the results for a final verdict.
        </p>
      </header>

      <section className="card-grid">
        <ModelCard
          title={DETECTION_CONFIG.image.title}
          type="image"
          file={files.image}
          loading={loading.image}
          onFileChange={(f) => setSelectedFile("image", f)}
          onDetect={() => runDetection("image")}
          result={results.image}
        />

        <ModelCard
          title={DETECTION_CONFIG.video.title}
          type="video"
          file={files.video}
          loading={loading.video}
          onFileChange={(f) => setSelectedFile("video", f)}
          onDetect={() => runDetection("video")}
          result={results.video}
        />

        <ModelCard
          title={DETECTION_CONFIG.audio.title}
          type="audio"
          file={files.audio}
          loading={loading.audio}
          onFileChange={(f) => setSelectedFile("audio", f)}
          onDetect={() => runDetection("audio")}
          result={results.audio}
        />
      </section>

      <BlurrinessPanel image={results.image} video={results.video} />

      <section className="fusion-panel">
        <div className="fusion-panel-head">
          <h2>Fusion Engine</h2>
          <p>Combine available detections into one decision.</p>
        </div>
        <button className="btn btn-primary" onClick={runFusion}>
          Run Fusion
        </button>

        {fusion && (
          <div className="fusion-result">
            <p className="fusion-label">Final Decision: {fusion.label}</p>
            <p className="fusion-confidence">Average Confidence: {fusion.confidence.toFixed(3)}</p>
          </div>
        )}
      </section>

      <section className="logs-panel">
        <div className="logs-head">
          <h2>Backend Logs</h2>
          <button className="btn btn-ghost" onClick={() => setLogs([])}>
            Clear Logs
          </button>
        </div>
        <div className="logs-box">
          {logs.length === 0 ? (
            <p className="logs-empty">No logs yet.</p>
          ) : (
            logs.map((log) => (
              <div key={log.id} className={`log-item log-${log.status}`}>
                <p className="log-meta">
                  [{log.time}] [{String(log.source).toUpperCase()}] [{log.status.toUpperCase()}]
                </p>
                <p className="log-message">{log.message}</p>
                {log.payload !== null && (
                  <pre className="log-payload">{JSON.stringify(log.payload, null, 2)}</pre>
                )}
              </div>
            ))
          )}
        </div>
        <div className="summary-panel">
          <div className="summary-head">
            <h2>Prediction Summary</h2>
            <p>Single chart using all available backend log parameters.</p>
          </div>
          <PieCard title="Combined Backend Log Summary" data={pieData} />
          <SummaryTable rows={modelTableRows} />
        </div>
      </section>
    </div>
  );
}

function SideInputPreview({
  variant,
  imageUrl,
  videoUrl,
  audioUrl,
  hasImage,
  hasVideo,
  hasAudio,
  onOpenQr,
}) {
  const title =
    variant === "left"
      ? "Input Preview (Image)"
      : "Input Preview (Video/Audio)";

  return (
    <aside className={`side-input side-input-${variant}`} aria-label={title}>
      <div className="side-input-head">
        <h3>{title}</h3>
        {variant === "left" ? (
          <button className="btn btn-ghost btn-qr" onClick={() => onOpenQr?.("image")}>
            QR Upload
          </button>
        ) : (
          <div className="side-qr-group">
            <button className="btn btn-ghost btn-qr" onClick={() => onOpenQr?.("video")}>
              QR Video
            </button>
            <button className="btn btn-ghost btn-qr" onClick={() => onOpenQr?.("audio")}>
              QR Audio
            </button>
          </div>
        )}
      </div>

      {variant === "left" ? (
        <>
          {!hasImage ? (
            <div className="side-input-empty">Select an image file.</div>
          ) : (
            <img className="side-input-image" src={imageUrl} alt="Selected input" />
          )}
          <div className="side-input-subtle">
            Video preview is shown on the right panel.
          </div>
        </>
      ) : (
        <>
          {!hasVideo ? (
            <div className="side-input-empty">Select a video file.</div>
          ) : (
            <video
              className="side-input-video"
              src={videoUrl}
              controls
              muted
              playsInline
            />
          )}

          {!hasAudio ? (
            <div className="side-input-empty" style={{ marginTop: 10 }}>
              Select an audio file (optional).
            </div>
          ) : (
            <div className="side-audio">
              <audio src={audioUrl} controls />
            </div>
          )}
        </>
      )}
    </aside>
  );
}

function QrUploadModal({ targetType, baseUrl, setBaseUrl, creating, session, error, onClose, onCreate }) {
  const [qrDataUrl, setQrDataUrl] = useState(null);
  const [suggestedIps, setSuggestedIps] = useState([]);
  const baseUrlLooksLocal =
    !baseUrl ||
    String(baseUrl).includes("127.0.0.1") ||
    String(baseUrl).includes("localhost") ||
    String(baseUrl).includes("0.0.0.0");

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const res = await axios.get(`${API_BASE}/network/ips`);
        if (!alive) return;
        setSuggestedIps(res?.data?.ipv4 || []);
      } catch {
        if (alive) setSuggestedIps([]);
      }
    };
    load();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    setQrDataUrl(null);
    const make = async () => {
      if (!session?.upload_page_url) return;
      try {
        const url = await QRCode.toDataURL(session.upload_page_url, {
          errorCorrectionLevel: "M",
          margin: 2,
          scale: 8,
        });
        if (alive) setQrDataUrl(url);
      } catch {
        if (alive) setQrDataUrl(null);
      }
    };
    make();
    return () => {
      alive = false;
    };
  }, [session?.upload_page_url]);

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true">
      <div className="modal-card">
        <div className="modal-head">
          <h3>QR Upload ({String(targetType).toUpperCase()})</h3>
          <button className="btn btn-ghost" onClick={onClose}>
            Close
          </button>
        </div>
        <p className="modal-sub">
          Set the backend URL that your phone can reach on Wi-Fi (usually your PC IP), then generate a QR code.
        </p>
        <label className="modal-label">
          Backend base URL
          <input
            className="modal-input"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="http://192.168.1.10:8000"
          />
        </label>
        {baseUrlLooksLocal && (
          <p className="modal-warn">
            Using <code>localhost</code>/<code>127.0.0.1</code> will not work on your phone. Use your PC Wi‑Fi IP
            (example: <code>http://192.168.x.x:8000</code>).
          </p>
        )}
        {suggestedIps.length > 0 && (
          <div className="ip-pills">
            {suggestedIps.map((ip) => (
              <button
                key={ip}
                className="btn btn-ghost ip-pill"
                onClick={() => setBaseUrl(`http://${ip}:8000`)}
                type="button"
              >
                {ip}
              </button>
            ))}
          </div>
        )}
        <div className="modal-actions">
          <button className="btn btn-primary" onClick={onCreate} disabled={creating}>
            {creating ? "Generating..." : "Generate QR"}
          </button>
        </div>

        {error && <p className="modal-error">{error}</p>}

        {session && (
          <div className="qr-box">
            {qrDataUrl ? (
              <img className="qr-img" src={qrDataUrl} alt="Upload QR code" />
            ) : (
              <div className="qr-img qr-placeholder" aria-label="Generating QR code">
                Generating QR…
              </div>
            )}
            <div className="qr-meta">
              <p className="muted">Scan with your phone, then upload a file.</p>
              <a className="qr-link" href={session.upload_page_url} target="_blank" rel="noreferrer">
                Open upload page
              </a>
              <p className="muted">Waiting for upload…</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ModelCard({ title, type, file, loading, onFileChange, onDetect, result }) {
  return (
    <div className={`detector-card detector-${type}`}>
      <h3>{title}</h3>
      <p className="detector-hint">{file ? file.name : "Choose a file to begin detection"}</p>
      <label className="file-input-wrap">
        <span className="file-input-label">
          {file ? file.name : "Choose file"}
        </span>
        <input
          type="file"
          onChange={(e) => onFileChange(e.target.files[0])}
          className="file-input"
        />
      </label>
      <button
        className="btn btn-card"
        onClick={onDetect}
        disabled={!file || loading}
      >
        {loading ? "Detecting..." : "Detect"}
      </button>
      {result && (
        <div className="result-box">
          <p>
            <span>Label</span>
            {result.label}
          </p>
          <p>
            <span>Confidence</span>
            {typeof result.confidence === "number" ? result.confidence.toFixed(4) : result.confidence}
          </p>
        </div>
      )}
    </div>
  );
}

function PieCard({ title, data }) {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  const normalized = total > 0 ? data : [{ label: "No Data", value: 1, color: "#5f6b8f" }];
  const normalizedTotal = normalized.reduce((sum, item) => sum + item.value, 0);

  let cumulative = 0;
  const segments = normalized.map((item) => {
    const start = cumulative / normalizedTotal;
    cumulative += item.value;
    const end = cumulative / normalizedTotal;
    return { ...item, start, end };
  });

  const gradient = segments
    .map((s) => `${s.color} ${(s.start * 100).toFixed(2)}% ${(s.end * 100).toFixed(2)}%`)
    .join(", ");

  return (
    <div className="pie-card">
      <h3>{title}</h3>
      <div className="pie-card-inner">
        <div
          className="pie-graph"
          style={{
            background: `conic-gradient(${gradient})`,
          }}
          aria-hidden="false"
          role="img"
          aria-label="Pie chart of backend log summary"
        />
        <div className="pie-legend">
        {data.map((item) => {
          const pct =
            item.displayPct != null
              ? item.displayPct
              : total > 0
              ? (item.value / total) * 100
              : 0;
          return (
            <p key={item.label}>
              <span className="dot" style={{ background: item.color }} />
              {item.label}: {pct.toFixed(1)}%
            </p>
          );
        })}
        {total === 0 && <p className="pie-empty">No matching backend log entries yet.</p>}
        </div>
      </div>
    </div>
  );
}

function SummaryTable({ rows }) {
  return (
    <div className="summary-table-wrap">
      <h3>Model Summary</h3>
      <table className="summary-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Runs</th>
            <th>Last Label</th>
            <th>Avg Confidence</th>
            <th>Avg Blur %</th>
            <th>Frames (Video)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.model}>
              <td>{r.model}</td>
              <td>{r.runs}</td>
              <td>{r.lastLabel}</td>
              <td>{r.avgConfidencePct}</td>
              <td>{r.avgBlurPct}</td>
              <td>{r.lastFrames}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BlurrinessPanel({ image, video }) {
  const getBlur = (obj) => {
    const v = obj?.blur_percent;
    if (typeof v === "number") return v;
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
  };

  const imageBlur = getBlur(image);
  const videoBlur = getBlur(video);

  const entries = [
    { key: "image", title: "Image", blur: imageBlur, meta: image?.label ?? null },
    { key: "video", title: "Video", blur: videoBlur, meta: video?.label ?? null },
  ];

  const hasAny = entries.some((e) => e.blur != null);

  const blurColorClass = (blur) => {
    if (blur == null) return "blur-neutral";
    if (blur < 35) return "blur-low";
    if (blur < 70) return "blur-mid";
    return "blur-high";
  };

  const blurMeaning = (blur) => {
    if (blur == null) return "No blur data";
    if (blur < 35) return "Good focus (mostly sharp)";
    if (blur < 70) return "Moderate blur (some detail loss)";
    return "High blur (low focus quality)";
  };

  return (
    <div className="blur-panel">
      <div className="blur-head">
        <h3>Sharpness / Blur Analysis</h3>
        <p>
          `blur_percent` comes from the backend on a 0-100 scale.
          Lower values are sharper (better focus), higher values are blurrier.
        </p>
      </div>

      {!hasAny ? (
        <p className="blur-empty">Run image or video detection to view sharpness and blur scores.</p>
      ) : (
        <div className="blur-rows">
          {entries.map((e) => (
            <div key={e.key} className="blur-row">
              <div className="blur-meta">
                <span className="blur-title">{e.title}</span>
                <span className="blur-value">
                  {e.blur == null ? "N/A" : `${e.blur.toFixed(1)}%`}
                </span>
              </div>
              <div className="blur-bar-track">
                <div
                  className={`blur-bar-fill ${blurColorClass(e.blur)}`}
                  style={{ width: `${Math.max(0, Math.min(100, e.blur ?? 0))}%` }}
                />
              </div>
              <div className="blur-scale">
                <span>Sharp</span>
                <span>Blurry</span>
              </div>
              <p className="blur-empty" style={{ marginTop: 8 }}>
                {blurMeaning(e.blur)}
              </p>
              {e.meta && (
                <p className="blur-empty" style={{ marginTop: 4 }}>
                  Detection result: {String(e.meta).toUpperCase()}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}