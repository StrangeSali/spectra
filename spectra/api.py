import os
import time
import asyncio
import tempfile
from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow_hub as hub
from tf_keras.models import load_model
from fastapi import FastAPI, UploadFile, File, Query
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()
from spectra.main import process_file

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


    async def push(
        self,
        predictions,
        source="file",
    ):

        entry = {
            "timestamp": time.time(),
            "source": source,
            "predictions": predictions,
        }

        async with self._lock:

            self._buffer.append(
                entry
            )

        return entry


    async def latest(self):

        async with self._lock:

            if not self._buffer:
                return None

            return self._buffer[-1]


    async def recent(
        self,
        n=10,
    ):

        async with self._lock:

            items = list(
                self._buffer
            )

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
