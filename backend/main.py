import asyncio
import logging
import torch
import librosa
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, FileResponse
from transformers import pipeline, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from PIL import Image
import shutil
import os
import uuid
import time
import io
import secrets
import socket
from typing import Tuple

logger = logging.getLogger(__name__)

audio_loaded = False
image_loaded = False
video_loaded = False
audio_load_lock = asyncio.Lock()
image_load_lock = asyncio.Lock()
video_load_lock = asyncio.Lock()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# NETWORK HELPERS (LAN IP)
# =========================
def _get_local_ipv4_addresses() -> list:
    """
    Best-effort list of local IPv4 addresses (excluding loopback).
    Used to help the frontend build a phone-reachable base URL.
    """
    ips = set()
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None):
            if info and info[0] == socket.AF_INET:
                ip = info[4][0]
                if ip and not ip.startswith("127."):
                    ips.add(ip)
    except Exception:
        pass

    # Fallback: "what IP would we use to reach the Internet"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        if ip and not ip.startswith("127."):
            ips.add(ip)
        s.close()
    except Exception:
        pass

    return sorted(ips)


@app.get("/network/ips")
async def network_ips():
    return {"ipv4": _get_local_ipv4_addresses()}

# =========================
# QR UPLOAD SESSIONS
# =========================
QR_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads", "qr"))
os.makedirs(QR_UPLOAD_DIR, exist_ok=True)

# In-memory session store (OK for single-instance dev; replace with Redis/DB for prod)
_qr_sessions = {}  # session_id -> dict


def _now() -> float:
    return time.time()


def _session_get(session_id: str):
    s = _qr_sessions.get(session_id)
    if not s:
        return None
    if s["expires_at"] <= _now():
        s["state"] = "expired"
    return s


def _safe_filename(name: str) -> str:
    name = (name or "upload").strip()
    name = name.replace("\\", "_").replace("/", "_").replace("..", "_")
    return name[:200] if len(name) > 200 else name


def _all_allowed_extensions():
    exts = set()
    for v in ALLOWED_EXTENSIONS.values():
        exts |= set(v)
    return exts


def _validate_any_upload_format(file: UploadFile):
    filename = (file.filename or "").strip()
    ext = os.path.splitext(filename)[1].lower()
    allowed = _all_allowed_extensions()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed))}.",
        )


def _make_qr_png_bytes(data: str) -> bytes:
    try:
        import qrcode  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for QR generation. Install `qrcode[pil]` (and ensure Pillow is installed)."
        ) from e
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=8,
        border=2,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/qr-upload/sessions")
async def create_qr_upload_session(request: Request):
    """
    Creates a one-time upload session and returns an upload URL + QR image URL.

    Notes:
    - The phone must reach your backend over the LAN, so prefer calling this with a base URL
      that is accessible from your phone (e.g. http://192.168.1.10:8000).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    base_url = (body.get("base_url") or "").strip()
    ttl_seconds = body.get("ttl_seconds")
    try:
        ttl_seconds = int(ttl_seconds) if ttl_seconds is not None else 10 * 60
    except Exception:
        ttl_seconds = 10 * 60
    ttl_seconds = max(30, min(ttl_seconds, 60 * 60))

    if not base_url:
        # Will often be http://127.0.0.1:8000 when called locally; caller can override via base_url.
        base_url = str(request.base_url).rstrip("/")
    else:
        base_url = base_url.rstrip("/")

    session_id = secrets.token_urlsafe(16)
    upload_page_url = f"{base_url}/qr-upload/sessions/{session_id}"
    qr_url = f"{base_url}/qr-upload/sessions/{session_id}/qr.png"

    _qr_sessions[session_id] = {
        "id": session_id,
        "created_at": _now(),
        "expires_at": _now() + ttl_seconds,
        "state": "pending",  # pending | uploaded | expired
        "filename": None,
        "saved_path": None,
        "size_bytes": None,
    }

    return {
        "session_id": session_id,
        "upload_page_url": upload_page_url,
        "qr_png_url": qr_url,
        "expires_in_seconds": ttl_seconds,
    }


@app.get("/qr-upload/sessions/{session_id}/qr.png")
async def get_qr_png(session_id: str, request: Request):
    s = _session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    if s["state"] == "expired":
        raise HTTPException(status_code=410, detail="Session expired.")

    base_url = str(request.base_url).rstrip("/")
    upload_page_url = f"{base_url}/qr-upload/sessions/{session_id}"
    png = _make_qr_png_bytes(upload_page_url)
    return Response(content=png, media_type="image/png")


@app.get("/qr-upload/sessions/{session_id}", response_class=HTMLResponse)
async def qr_upload_page(session_id: str):
    s = _session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")

    if s["state"] == "expired":
        return HTMLResponse(
            content="<h2>Upload session expired</h2><p>Please generate a new QR code.</p>",
            status_code=410,
        )

    allowed = ", ".join(sorted(_all_allowed_extensions()))
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Upload file</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; padding: 18px; max-width: 720px; margin: 0 auto; }}
      .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; }}
      .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
      button {{ padding: 10px 14px; border-radius: 10px; border: 1px solid #111; background: #111; color: #fff; font-size: 16px; }}
      button:disabled {{ opacity: 0.5; }}
      .muted {{ color: #666; font-size: 14px; }}
      .ok {{ color: #0a7; }}
      .err {{ color: #b00; }}
      progress {{ width: 100%; height: 18px; }}
      code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <h2>Upload a file</h2>
    <div class="card">
      <p class="muted">Allowed: <code>{allowed}</code></p>
      <div class="row">
        <input id="file" type="file" />
        <button id="btn" onclick="upload()">Upload</button>
      </div>
      <div style="margin-top: 12px;">
        <progress id="prog" value="0" max="100" style="display:none;"></progress>
        <div id="status" class="muted" style="margin-top: 8px;"></div>
      </div>
    </div>
    <script>
      const statusEl = document.getElementById('status');
      const prog = document.getElementById('prog');
      const btn = document.getElementById('btn');
      function setStatus(msg, cls) {{
        statusEl.className = cls || 'muted';
        statusEl.textContent = msg;
      }}
      async function upload() {{
        const f = document.getElementById('file').files[0];
        if (!f) {{
          setStatus('Please choose a file first.', 'err');
          return;
        }}
        btn.disabled = true;
        prog.style.display = 'block';
        prog.value = 0;
        setStatus('Uploading…', 'muted');

        const form = new FormData();
        form.append('file', f);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', `/qr-upload/sessions/{session_id}/file`);
        xhr.upload.onprogress = (e) => {{
          if (e.lengthComputable) {{
            prog.value = Math.round((e.loaded / e.total) * 100);
          }}
        }};
        xhr.onload = () => {{
          btn.disabled = false;
          if (xhr.status >= 200 && xhr.status < 300) {{
            setStatus('Uploaded successfully. You can go back to your desktop app.', 'ok');
          }} else {{
            try {{
              const j = JSON.parse(xhr.responseText);
              setStatus(j.detail || 'Upload failed.', 'err');
            }} catch {{
              setStatus('Upload failed.', 'err');
            }}
          }}
        }};
        xhr.onerror = () => {{
          btn.disabled = false;
          setStatus('Network error during upload.', 'err');
        }};
        xhr.send(form);
      }}
      setStatus('Ready.', 'muted');
    </script>
  </body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/qr-upload/sessions/{session_id}/status")
async def qr_upload_status(session_id: str):
    s = _session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "state": s["state"],
        "filename": s["filename"],
        "size_bytes": s["size_bytes"],
        "expires_at": s["expires_at"],
        "saved_path": s["saved_path"],
    }


@app.post("/qr-upload/sessions/{session_id}/file")
async def qr_upload_file(session_id: str, file: UploadFile = File(...)):
    s = _session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    if s["state"] == "expired":
        raise HTTPException(status_code=410, detail="Session expired.")

    _validate_any_upload_format(file)

    safe = _safe_filename(file.filename or "upload")
    out_name = f"{session_id}_{uuid.uuid4().hex}_{safe}"
    out_path = os.path.join(QR_UPLOAD_DIR, out_name)

    size = 0
    try:
        with open(out_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
                size += len(chunk)
    except Exception as e:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e
    finally:
        try:
            await file.close()
        except Exception:
            pass

    s["state"] = "uploaded"
    s["filename"] = safe
    s["saved_path"] = out_path
    s["size_bytes"] = size

    return {"ok": True, "saved_path": out_path, "size_bytes": size}


@app.get("/qr-upload/sessions/{session_id}/download")
async def qr_upload_download(session_id: str):
    s = _session_get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found.")
    if s["state"] == "expired":
        raise HTTPException(status_code=410, detail="Session expired.")
    if s["state"] != "uploaded" or not s.get("saved_path"):
        raise HTTPException(status_code=409, detail="No file uploaded yet.")
    path = s["saved_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=410, detail="Uploaded file no longer exists on server.")
    filename = s.get("filename") or os.path.basename(path)
    return FileResponse(path, filename=filename)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# AUDIO MODEL
# =========================
audio_model_name = "garystafford/wav2vec2-deepfake-voice-detector"

feature_extractor = None
audio_model = None


def _blur_percent_from_rgb_array(rgb_array):
    """
    Return blur percentage (0-100), higher means blurrier.
    Uses Laplacian variance as a sharpness signal and maps it to a percentage.
    """
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Map sharpness -> blur percent with a smooth bounded transform.
    # Tune divisor if you want stricter/looser blur estimation.
    blur_percent = 100.0 / (1.0 + (lap_var / 100.0))
    return float(max(0.0, min(100.0, blur_percent)))

def predict_audio(path):
    global feature_extractor, audio_model

    if feature_extractor is None or audio_model is None:
        raise RuntimeError("Audio model not loaded yet.")

    # Avoid extremely long inputs that can blow up memory/time for Wav2Vec2.
    # You can increase/decrease this (seconds) if you want.
    max_seconds = 30
    try:
        audio, _ = librosa.load(
            path,
            sr=16000,
            mono=True,
            duration=max_seconds,  # seconds
        )
    except Exception as e:
        raise RuntimeError(f"Audio decoding failed: {e}") from e

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("Audio decoding produced an empty signal.")

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_seconds * 16000),
    )

    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        # Some Wav2Vec2 variants can use attention_mask; pass it when available.
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            logits = audio_model(input_values, attention_mask=attention_mask).logits
        else:
            logits = audio_model(input_values).logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()

    return {
        "label": audio_model.config.id2label[pred],
        "confidence": float(probs[0][pred]),
        "blur_percent": None
    }

# =========================
# IMAGE MODEL
# =========================
image_classifier = None
video_classifier = None
video_frame_classifier = None
video_pipeline_available = False
# Image endpoint: balanced real/fake ViT (FaceForensics-style faces).
image_model_name = "dima806/deepfake_vs_real_image_detection"
# Video: ViT + face crops. Primary: OpenForensics ViT (~96.5% reported on model card).
# https://huggingface.co/hamzenium/ViT-Deepfake-Classifier
# Fallback if download/load fails: https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model
VIDEO_FRAME_MODEL_CANDIDATES = [
    "hamzenium/ViT-Deepfake-Classifier",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
]
video_frame_model_name = VIDEO_FRAME_MODEL_CANDIDATES[0]
# Legacy HF video-classification models are often weak / inconsistent; we prefer frame+face path.
USE_HF_VIDEO_PIPELINE = False
video_model_candidates = [
    "Hemgg/deepfake-vs-real-video-detection",
    "Hemgg/deepfake-video-detection",
]
video_model_name = None
ALLOWED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".m4a", ".flac", ".ogg"},
}


def _validate_upload_format(file: UploadFile, media_type: str):
    filename = (file.filename or "").strip()
    ext = os.path.splitext(filename)[1].lower()
    allowed = ALLOWED_EXTENSIONS[media_type]

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {media_type} format '{ext or 'unknown'}'. "
                f"Allowed formats: {', '.join(sorted(allowed))}."
            ),
        )

    content_type = (file.content_type or "").lower()
    if content_type and content_type != "application/octet-stream":
        if media_type == "image" and not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image MIME type '{content_type}'. Expected an image file.",
            )
        if media_type == "video" and not content_type.startswith("video/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video MIME type '{content_type}'. Expected a video file.",
            )
        if media_type == "audio" and not (
            content_type.startswith("audio/") or content_type.startswith("video/")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio MIME type '{content_type}'. Expected an audio file.",
            )


def predict_image(path):
    global image_classifier

    image = Image.open(path).convert("RGB")
    blur_percent = _blur_percent_from_rgb_array(np.array(image))
    result = image_classifier(image)[0]

    return {
        "label": result["label"],
        "confidence": float(result["score"]),
        "blur_percent": round(blur_percent, 2)
    }

# =========================
# VIDEO MODEL
# =========================
# Balanced decision on blended P(fake) after face crops + ViT (tune 0.48–0.55 if needed).
VIDEO_FAKE_THRESHOLD = 0.52


def _crop_face_or_center(rgb: np.ndarray) -> np.ndarray:
    """
    Deepfake detectors are trained on faces; full-body / background frames hurt accuracy.
    Uses OpenCV Haar faces; falls back to center square crop if no face.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return rgb
    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            pad_x = int(0.12 * fw)
            pad_y = int(0.12 * fh)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + fw + pad_x)
            y1 = min(h, y + fh + pad_y)
            return rgb[y0:y1, x0:x1]
    except Exception as e:
        logger.debug("Face crop skipped: %s", e)
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return rgb[y0 : y0 + side, x0 : x0 + side]


def _frame_probs_from_preds(preds: list) -> Tuple[float, float]:
    """Map pipeline top_k list to P(fake), P(real) using label names."""
    pf = pr = 0.0
    for p in preds or []:
        lab = str(p.get("label", "")).lower()
        sc = float(p.get("score", 0.0))
        if "fake" in lab or "deepfake" in lab:
            pf = max(pf, sc)
        elif "real" in lab or "authentic" in lab:
            pr = max(pr, sc)
    if pf > 0 and pr > 0:
        s = pf + pr
        return pf / s, pr / s
    if pf > 0:
        return pf, 1.0 - pf
    if pr > 0:
        return 1.0 - pr, pr
    return 0.5, 0.5


def extract_frames(video_path, max_frames=32):
    """Extract frames evenly sampled from video. Uses absolute path for Windows compatibility."""
    abs_path = os.path.abspath(video_path)
    cap = cv2.VideoCapture(abs_path)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video file '{video_path}'. "
            "Ensure the file is a valid video (MP4/H.264 recommended) and not corrupted. "
            "Try re-encoding with a standard codec."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: read until we can't
        total_frames = 0
        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    # Sample evenly across the video for better coverage of short clips
    indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=np.int64)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def predict_video(path):
    global video_classifier, video_frame_classifier

    abs_path = os.path.abspath(path)
    frames = extract_frames(abs_path)
    has_frames = len(frames) > 0

    if not has_frames:
        raise RuntimeError(
            "No frames extracted from this video. "
            "Ensure the file is a valid video (MP4/H.264 recommended). "
            "Try re-encoding or a different video file."
        )

    use_frame_fallback = False

    if video_classifier is not None and USE_HF_VIDEO_PIPELINE:
        try:
            predictions = video_classifier(abs_path, top_k=5)
            pf, pr = _frame_probs_from_preds(predictions)
            avg_fake = float(pf)
            avg_real = float(pr)
        except Exception as e:
            logger.warning("Video pipeline inference failed, falling back to face crops: %s", e)
            use_frame_fallback = True
    else:
        use_frame_fallback = True

    if use_frame_fallback:
        if video_frame_classifier is None:
            _load_video_frame_classifier_sync()
        per_fake = []
        per_real = []
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            cropped = _crop_face_or_center(arr)
            pil = Image.fromarray(cropped)
            preds = video_frame_classifier(pil, top_k=2)
            pf, pr = _frame_probs_from_preds(preds)
            per_fake.append(pf)
            per_real.append(pr)
        if per_fake:
            m_f = float(np.mean(per_fake))
            m_r = float(np.mean(per_real))
            med_f = float(np.median(per_fake))
            med_r = float(np.median(per_real))
            avg_fake = 0.4 * m_f + 0.6 * med_f
            avg_real = 0.4 * m_r + 0.6 * med_r
            s = avg_fake + avg_real
            if s > 0:
                avg_fake /= s
                avg_real /= s
        else:
            avg_fake = avg_real = 0.5

    frame_blur_percents = [_blur_percent_from_rgb_array(np.array(f)) for f in frames] if has_frames else []

    if avg_fake >= VIDEO_FAKE_THRESHOLD:
        final_label = "FAKE"
    else:
        final_label = "REAL"
    confidence = float(max(avg_fake, avg_real))

    return {
        "label": final_label,
        "confidence": confidence,
        "frames_analyzed": len(frames),
        "blur_percent": round(float(np.mean(frame_blur_percents)), 2) if frame_blur_percents else None,
        "model": video_model_name or video_frame_model_name,
    }


def _load_audio_sync():
    global feature_extractor, audio_model
    feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name).to(device)
    audio_model.eval()


def _load_image_sync():
    global image_classifier
    image_classifier = pipeline(
        "image-classification",
        model=image_model_name,
        device=0 if torch.cuda.is_available() else -1
    )


def _load_video_frame_classifier_sync():
    """Load strongest available ViT for face crops (deepfake vs real)."""
    global video_frame_classifier, video_frame_model_name
    if video_frame_classifier is not None:
        return
    last_err = None
    for model_id in VIDEO_FRAME_MODEL_CANDIDATES:
        try:
            video_frame_classifier = pipeline(
                "image-classification",
                model=model_id,
                device=0 if torch.cuda.is_available() else -1,
            )
            video_frame_model_name = model_id
            logger.info("Video frame classifier loaded: %s", model_id)
            return
        except Exception as e:
            last_err = e
            logger.warning("Could not load video frame model %s: %s", model_id, e)
    raise RuntimeError(
        f"Failed to load any video frame model. Last error: {last_err}"
    ) from last_err


def _load_video_sync():
    global video_classifier, video_model_name, video_pipeline_available
    video_classifier = None
    video_pipeline_available = False

    if USE_HF_VIDEO_PIPELINE:
        last_error = None
        for candidate in video_model_candidates:
            try:
                video_classifier = pipeline(
                    "video-classification",
                    model=candidate,
                    device=0 if torch.cuda.is_available() else -1,
                )
                video_model_name = candidate
                video_pipeline_available = True
                logger.info("Loaded HF video pipeline: %s", candidate)
                break
            except Exception as e:
                last_error = e
        if not video_pipeline_available:
            logger.warning(
                "HF video pipeline not available (%s). Using face-crop + ViT only.",
                last_error,
            )

    if not USE_HF_VIDEO_PIPELINE or not video_pipeline_available:
        video_model_name = f"face_crop+{video_frame_model_name}"

    _load_video_frame_classifier_sync()


async def ensure_audio_loaded():
    global audio_loaded
    if audio_loaded:
        return
    async with audio_load_lock:
        if audio_loaded:
            return
        try:
            await asyncio.to_thread(_load_audio_sync)
            audio_loaded = True
        except Exception as e:
            raise RuntimeError(f"Audio model loading failed: {e}") from e


async def ensure_image_loaded():
    global image_loaded
    if image_loaded:
        return
    async with image_load_lock:
        if image_loaded:
            return
        try:
            await asyncio.to_thread(_load_image_sync)
            image_loaded = True
        except Exception as e:
            raise RuntimeError(f"Image model loading failed: {e}") from e


async def ensure_video_loaded():
    global video_loaded
    if video_loaded:
        return
    async with video_load_lock:
        if video_loaded:
            return
        try:
            await asyncio.to_thread(_load_video_sync)
            video_loaded = True
        except Exception as e:
            raise RuntimeError(f"Video model loading failed: {e}") from e

# =========================
# ROUTES
# =========================

@app.post("/detect-deepfake-audio")
async def detect_audio(file: UploadFile = File(...)):
    path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        _validate_upload_format(file, "audio")
        await ensure_audio_loaded()
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_audio(path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/detect-deepfake-image")
async def detect_image_api(file: UploadFile = File(...)):
    path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    try:
        _validate_upload_format(file, "image")
        await ensure_image_loaded()
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_image(path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(path):
            os.remove(path)


@app.post("/detect-deepfake-video")
async def detect_video_api(file: UploadFile = File(...)):
    filename = file.filename or "video.mp4"
    path = f"temp_{uuid.uuid4().hex}_{filename}"
    abs_path = os.path.abspath(path)
    try:
        _validate_upload_format(file, "video")
        await ensure_video_loaded()
        with open(abs_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = predict_video(abs_path)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video detection failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if os.path.exists(abs_path):
            try:
                os.remove(abs_path)
            except OSError:
                pass


# =========================
# RUN
# =========================
# uvicorn main:app --reload