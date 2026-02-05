from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import librosa
import numpy as np
import io

app = FastAPI()

API_KEY = "GUVI1234"

class AudioInput(BaseModel):
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
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_bytes = safe_b64decode(data.audio_base64)

        audio_buffer = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_buffer, sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_std = float(np.std(mfcc))

        if mfcc_std < 15:
            result = "AI_GENERATED"
            confidence = round(1 - (mfcc_std / 20), 2)
        else:
            result = "HUMAN"
            confidence = round(min(mfcc_std / 40, 1.0), 2)

        return {
            "result": result,
            "confidence": confidence
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Audio processing failed"
        )
