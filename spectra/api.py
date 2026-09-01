import os
import time
import asyncio
import tempfile
import uuid
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow_hub as hub
from tf_keras.models import load_model
from google.cloud import storage
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Query
from dotenv import load_dotenv
import tensorflow as tf

# Cargar variables del archivo .env
load_dotenv()
from spectra.models.models import load_models
from spectra.main import process_waveform, process_file
from spectra.processing.audio import (
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio
)
from spectra.processing.classifier import predict_sound

app = FastAPI(title="Spectra Audio Classifier API")
import tensorflow_hub as hub

# --------------------------------------------------
# Prediction history
# --------------------------------------------------

prediction_history = deque(maxlen=100)

# --------------------------------------------------
# Probability smoother
# --------------------------------------------------
class RollingPredictionBuffer:
    def __init__(self, maxlen: int = 50):
        self._buffer = deque(maxlen=maxlen)
        self._lock = asyncio.Lock()

    async def push(self, predictions: List[Dict[str, Any]], source: str = "file") -> Dict[str, Any]:
        entry = {
            "timestamp": time.time(),
            "source": source,
            "predictions": predictions,
        }
        async with self._lock:
            self._buffer.append(entry)
        return entry

    async def latest(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._buffer[-1] if self._buffer else None

    async def recent(self, n: int = 10) -> List[Dict[str, Any]]:
        async with self._lock:
            items = list(self._buffer)
        return items[-n:]

    async def clear(self):
        async with self._lock:
            self._buffer.clear()

file_prediction_buffer = RollingPredictionBuffer(maxlen=100)

class ProbabilitySmoother:

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.smoothed = None

    def update(self, probabilities):

        if self.smoothed is None:
            self.smoothed = probabilities.copy()

        else:
            self.smoothed = (
                self.alpha * probabilities
                + (1 - self.alpha) * self.smoothed
            )

        return self.smoothed


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_prediction(
    predictions,
    source
):

    entry = {
        "timestamp": time.time(),
        "source": source,
        "predictions": predictions
    }

    prediction_history.append(entry)

    return entry


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok"
    }

# --------------------------------------------------
# Audio file prediction
# --------------------------------------------------
@app.get("/recent")
async def get_recent_file_predictions(n: int = Query(10, ge=1, le=100)):
    return {"predictions": await file_prediction_buffer.recent(n)}

@app.post("/predict")
async def predict_file(
    file: UploadFile = File(...)
):

    suffix = (
        os.path.splitext(file.filename)[1]
        or ".wav"
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as tmp:

        contents = await file.read()

        tmp.write(contents)

        tmp_path = tmp.name

    try:

        results = process_file(
            tmp_path,
            window_seconds=1.0,
            overlap=0.5,
            max_classes=3,
            confidence_threshold=0.20
        )

        entry = await file_prediction_buffer.push(results, source="file")

        return {
            "filename": file.filename,
            "predictions": results,
            "timestamp": entry["timestamp"]
        }

    finally:

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
