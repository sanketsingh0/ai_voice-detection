from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io
import os

# ---------------- APP INIT ----------------
app = FastAPI(
    title="AI Voice Detection API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"status": "API is running"}

# ---------------- API KEY ----------------
API_KEY = os.getenv("API_KEY")

# ---------------- REQUEST MODEL ----------------
class AudioInput(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# ---------------- BASE64 SAFE DECODE ----------------
def safe_b64decode(data: str) -> bytes:
    data = data.strip().replace("\n", "").replace(" ", "")
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)
    return base64.b64decode(data)

# ---------------- MAIN ENDPOINT ----------------
@app.post("/detect")
def detect_audio(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    # API KEY CHECK
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode audio
        audio_bytes = safe_b64decode(data.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        # Load audio
        y, sr = librosa.load(audio_buffer, sr=None)

        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std = float(np.std(mfcc))

        # Simple detection logic
        if mfcc_std < 15:
            prediction = "AI_GENERATED"
            confidence = round(1 - (mfcc_std / 20), 2)
        else:
            prediction = "HUMAN"
            confidence = round(min(mfcc_std / 40, 1.0), 2)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "language": data.language,
            "audio_format": data.audio_format,
            "status": "success"
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Audio processing failed"
        )
