from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io
import os

app = FastAPI(
    title="AI Voice Detection API",
    docs_url="/docs",
    redoc_url="/redoc"
)
@app.get("/")
def root():
    return {"status": "API is running"}
    @app.get("/routes")
def routes():
    return [{"path": r.path, "methods": list(r.methods)} for r in app.router.routes]


# ---- Read API KEY from Environment Variable ----
API_KEY = os.getenv("API_KEY")


# ---- Request Model (MATCHES GUVI PORTAL) ----
class AudioInput(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


def safe_b64decode(data: str) -> bytes:
    data = data.strip().replace("\n", "").replace(" ", "")
    padding = len(data) % 4
    if padding:
        data += "=" * (4 - padding)
    return base64.b64decode(data)


@app.post("/detect")
def detect_audio(
    data: AudioInput,
    x_api_key: str = Header(None)
):
    # ---- API KEY CHECK ----
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # ---- Decode Audio ----
        audio_bytes = safe_b64decode(data.audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        # ---- Load Audio ----
        y, sr = librosa.load(audio_buffer, sr=None)

        # ---- Feature Extraction ----
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std = float(np.std(mfcc))

        # ---- Simple Detection Logic ----
        if mfcc_std < 15:
            prediction = "AI_GENERATED"
            confidence = round(1 - (mfcc_std / 20), 2)
        else:
            prediction = "HUMAN"
            confidence = round(min(mfcc_std / 40, 1.0), 2)

        # ---- REQUIRED RESPONSE ----
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


