# app.py
import io, os, tempfile, json, math, uuid, requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import soundfile as sf
import librosa
import pyloudnorm as pyln

app = Flask(__name__)
CORS(app)  # allow calls from your Lovable domain

# ---- Helpers ----

def load_audio_from_bytes(raw: bytes, sr_target: int = 48000):
    """Load audio bytes, convert to mono float32 at sr_target."""
    y, sr = sf.read(io.BytesIO(raw), always_2d=True)  # (N, C)
    y = y.astype(np.float32)
    # mix to mono
    y_mono = np.mean(y, axis=1)
    # resample if needed (librosa uses float64 internally; cast back to float32)
    if sr != sr_target:
        y_rs = librosa.resample(y_mono.astype(np.float32), orig_sr=sr, target_sr=sr_target)
        sr = sr_target
        y_mono = y_rs.astype(np.float32)
    return y_mono, sr

def analyze_audio(y: np.ndarray, sr: int):
    """Compute metrics required by the UI."""
    # safety: trim NaNs/Infs
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Loudness (LUFS) ---
    meter = pyln.Meter(sr)  # EBU R128
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = float('nan')

    # --- Peaks and crest factor ---
    peak = float(np.max(np.abs(y)) + 1e-12)  # avoid log(0)
    peak_db = 20.0 * math.log10(peak)
    rms = float(np.sqrt(np.mean(y**2)) + 1e-12)
    rms_db = 20.0 * math.log10(rms)
    crest_db = peak_db - rms_db

    # --- Spectral balance (very rough bands) ---
    S, freqs = librosa.magphase(np.abs(librosa.stft(y, n_fft=4096)))[0], librosa.fft_frequencies(sr=sr, n_fft=4096)
    mag = np.mean(S, axis=1) + 1e-12  # average magnitude across time

    def band_db(f_lo, f_hi):
        idx = np.where((freqs >= f_lo) & (freqs < f_hi))[0]
        if idx.size == 0: return -999.0
        val = np.mean(mag[idx])
        return float(20.0 * math.log10(val))

    low_db  = band_db(20, 200)
    mid_db  = band_db(200, 2000)
    high_db = band_db(2000, 12000)

    # --- Stereo width proxy (correlation of L/R)
    # If we only have mono (our default), width is 0. If stereo bytes provided, estimate before mono mix.
    width_pct = 0
    # (Optional) If you want real stereo width, you can accept stereo and compute:
    # corr = np.corrcoef(y_stereo[:,0], y_stereo[:,1])[0,1]; width_pct = int(max(0, 100*(1-corr)))

    # --- Heuristics / issues ---
    issues = []
    notes = []

    if not math.isnan(lufs):
        if lufs > -10:
            issues.append("Track is very loud; risk of clipping or limited headroom.")
        if lufs < -20:
            issues.append("Track is quiet; consider raising integrated loudness.")
    if crest_db < 6:
        issues.append("Low crest factor; mix may be over-compressed.")
    if (low_db - mid_db) > 3:
        issues.append("Excessive low-end energy (20–200 Hz).")
    if (high_db - mid_db) > 3:
        issues.append("Harsh/bright highs (2–12 kHz).")
    if abs(low_db - mid_db) < 1 and abs(high_db - mid_db) < 1:
        notes.append("Overall spectral balance looks even.")

    response = {
        "lufs": round(lufs, 2) if not math.isnan(lufs) else None,
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
    return response

# ---- Routes ----

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts:
      - multipart/form-data with 'file'
      - application/json: { "url": "https://.../audio.wav" }  (direct audio file URL; NOT YouTube)
    Returns: JSON metrics for Lovable UI.
    """
    try:
        if "file" in request.files:
            raw = request.files["file"].read()
        else:
            data = request.get_json(silent=True) or {}
            url = (data.get("url") or "").strip()
            if not url:
                return jsonify({"error": "No file or URL provided."}), 400
            # Only direct audio URLs here (no YouTube). yt-dlp often blocked on hosts.
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                return jsonify({"error": f"Could not fetch URL (status {r.status_code})."}), 400
            raw = r.content

        # Load + analyze
        y, sr = load_audio_from_bytes(raw, sr_target=48000)
        if y.size < sr * 2:  # require >= 2 sec
            return jsonify({"error": "Audio too short for analysis (need >= 2 seconds)."}), 400

        result = analyze_audio(y, sr)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render/Heroku will set PORT; default to 5000 locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
