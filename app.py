# app.py
import io, os, math, requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
import librosa
import pyloudnorm as pyln

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024  # 60MB upload cap
# Wide-open CORS for a frontend-only client
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# ---------- helpers ----------

def load_audio_from_bytes(raw: bytes, sr_target: int = 48000):
    """Load audio bytes -> mono float32 at sr_target."""
    y, sr = sf.read(io.BytesIO(raw), always_2d=True)  # shape (N, C)
    y = y.astype(np.float32)
    y_mono = np.mean(y, axis=1)
    if sr != sr_target:
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=sr_target).astype(np.float32)
        sr = sr_target
    return y_mono, sr

def analyze_audio(y: np.ndarray, sr: int):
    """Compute metrics quickly & robustly."""
    # shorten very long files to speed up (first 90s is enough for guidance)
    max_seconds = 90
    if y.shape[0] > sr * max_seconds:
        y = y[: sr * max_seconds]

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Loudness (LUFS)
    try:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = None

    # Peak / RMS / Crest
    peak = float(np.max(np.abs(y)) + 1e-12)
    peak_db = 20.0 * math.log10(peak)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    rms_db = 20.0 * math.log10(rms)
    crest_db = peak_db - rms_db

    # Spectral balance
    stft = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    mag = np.mean(np.abs(stft), axis=1) + 1e-12
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_db(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0:
            return -999.0
        return float(20.0 * math.log10(np.mean(mag[idx])))

    low_db  = band_db(20, 200)
    mid_db  = band_db(200, 2000)
    high_db = band_db(2000, 12000)

    # Stereo width proxy (we’re mono in this analyzer → 0)
    width_pct = 0

    issues, notes = [], []
    if lufs is not None:
        if lufs > -10: issues.append("Very loud; little headroom (risk of clipping).")
        if lufs < -20: issues.append("Quiet track; consider raising integrated loudness.")
    if crest_db < 6: issues.append("Low crest factor; likely over-compressed.")
    if (low_db - mid_db) > 3: issues.append("Excess low-end (20–200 Hz).")
    if (high_db - mid_db) > 3: issues.append("Harsh highs (2–12 kHz).")
    if abs(low_db - mid_db) < 1 and abs(high_db - mid_db) < 1:
        notes.append("Spectral balance looks fairly even.")

    return {
        "lufs": round(lufs, 2) if lufs is not None else None,
        "true_peak_db": round(peak_db, 2),
        "crest_factor_db": round(crest_db, 2),
        "low_mid_high_db": {
            "low": round(low_db, 2),
            "mid": round(mid_db, 2),
            "high": round(high_db, 2),
        },
        "stereo_width_pct": int(width_pct),
        "issues": issues,
        "notes": " ".join(notes)[:400]
    }

# ---------- routes ----------

@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return _preflight_ok()
    return jsonify({"ok": True})

def _preflight_ok():
    # Proper preflight response (some browsers need echoed headers)
    resp = app.make_response("")
    resp.status_code = 204
    req_headers = request.headers.get("Access-Control-Request-Headers", "")
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = req_headers or "Content-Type, Authorization"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"]  = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp

@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return _preflight_ok()

    try:
        # accept several field names
        file_obj = None
        for key in ["file", "audio", "upload", "data"]:
            if key in request.files:
                file_obj = request.files[key]
                break

        if file_obj:
            raw = file_obj.read()
        else:
            data = request.get_json(silent=True) or {}
            url = (data.get("url") or "").strip()
            if not url:
                return jsonify({"error": "No file or URL provided."}), 400
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                return jsonify({"error": f"Could not fetch URL (status {r.status_code})."}), 400
            raw = r.content

        y, sr = load_audio_from_bytes(raw, sr_target=48000)
        if y.size < sr * 2:
            return jsonify({"error": "Audio too short for analysis (need >= 2 seconds)."}), 400

        result = analyze_audio(y, sr)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
