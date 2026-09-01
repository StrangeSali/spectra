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

# Let TF use all available CPU cores for a single inference call (intra-op),
# and allow a couple of ops to run in parallel (inter-op). On CPU-only
# machines this is often the single biggest lever for per-call latency -
# by default TF can leave cores idle during a forward pass.
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() or 4)
tf.config.threading.set_inter_op_parallelism_threads(2)

from spectra.processing.audio import (
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio
)
from spectra.processing.yamnet_utils import extract_features
from spectra.processing.classifier import predict_probabilities, predict_sound

app = FastAPI(title="Spectra Audio Classifier API")

# --------------------------------------------------
# Carga de modelos
# --------------------------------------------------
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
LOCAL_MODEL_PATH = "dense_50.keras"

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading Classifier...")
classifier_model = None

if MODEL_BUCKET_NAME and MODEL_NAME:
    try:
        print(f"Intentando descargar desde GCS: {MODEL_BUCKET_NAME}/{MODEL_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_NAME)
        temp_model_path = "/tmp/dense_50.keras"
        blob.download_to_filename(temp_model_path)
        classifier_model = load_model(temp_model_path)
        print("Modelo descargado exitosamente desde GCS.")
    except Exception as e:
        print(f"No se pudo cargar desde GCS ({e}). Se intentará modelo local...")

if classifier_model is None and os.path.exists(LOCAL_MODEL_PATH):
    print(f"Cargando modelo local desde: {LOCAL_MODEL_PATH}")
    classifier_model = load_model(LOCAL_MODEL_PATH)

if classifier_model is None:
    print("ADVERTENCIA: No se encontró ningún modelo (ni en GCS ni local).")

# Minimum top-class confidence (after smoothing) required to report a
# prediction as "confident" over the mic stream. Below this, ambient
# noise / silence tends to produce a shifting cloud of low-confidence,
# unrelated guesses - better to say "no clear sound" than to display one.
MIN_CONFIDENT_TOP_SCORE = 0.30


# --------------------------------------------------
# Rolling memory buffer
# --------------------------------------------------
class RollingPredictionBuffer:
    """
    Thread/async-safe rolling buffer that keeps only the most recent N
    prediction results. Backed by a deque(maxlen=...), so once it's full,
    pushing a new entry automatically evicts the oldest one - memory stays
    bounded no matter how long a stream runs.
    """

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


class LatestValueSlot:
    """
    A single-slot 'mailbox' for audio chunks, not a FIFO queue.

    Why: if inference (YAMNet + classifier) takes longer than the rate at
    which mic chunks arrive, a normal queue keeps growing and predictions
    fall further and further behind real time ("lag"). This slot always
    holds only the *newest* unprocessed chunk - a fresh chunk silently
    overwrites whatever hasn't been picked up yet, so the consumer is
    guaranteed to always be working on the freshest audio available.
    """

    def __init__(self):
        self._item = None
        self._event = asyncio.Event()

    def put(self, item):
        self._item = item
        self._event.set()

    async def get(self):
        await self._event.wait()
        item = self._item
        self._item = None
        self._event.clear()
        return item


class RollingAudioBuffer:
    """
    Keeps a fixed-length sliding window of the most recent audio samples.
    Small chunks arrive frequently (e.g. every 0.25s) and get folded into
    this window, so every inference still sees a full window of context
    (e.g. 1s) even though updates happen several times more often than
    that - this is what shrinks perceived latency without sacrificing the
    amount of audio the model sees per prediction.
    """

    def __init__(self, window_size: int = 16000):
        self.window_size = window_size
        self._buf = np.zeros(window_size, dtype=np.float32)

    def push_chunk(self, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if n >= self.window_size:
            self._buf = chunk[-self.window_size:].copy()
        else:
            self._buf = np.concatenate([self._buf[n:], chunk])
        return self._buf.copy()


class ProbabilitySmoother:
    """
    Exponential moving average over raw class probabilities, applied
    *before* they're turned into top-3 labels.

    Why probabilities and not labels: smoothing after the fact (e.g.
    averaging or voting on the last few displayed labels) treats a
    confident 0.93 result and a barely-above-threshold 0.11 result as
    equally "real" votes. Smoothing the underlying probability vector
    lets confident, consistent signals build up across consecutive
    windows while brief noisy spikes get damped proportionally to how
    weak they were - so the displayed result stops flickering without
    adding a fixed delay.

    alpha controls responsiveness: higher = reacts faster to real
    changes but smooths less; lower = steadier but slower to update.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._smoothed: Optional[np.ndarray] = None

    def update(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities)
        if self._smoothed is None:
            self._smoothed = probabilities.copy()
        else:
            self._smoothed = self.alpha * probabilities + (1 - self.alpha) * self._smoothed
        return self._smoothed


# Global rolling buffer for file-based (/predict) results
file_prediction_buffer = RollingPredictionBuffer(maxlen=100)

# Per-connection state for mic sessions, keyed by session_id
# Each entry: {"buffer": RollingPredictionBuffer, "slot": LatestValueSlot, "task": asyncio.Task}
mic_sessions: Dict[str, Dict[str, Any]] = {}


async def run_inference(waveform: np.ndarray) -> np.ndarray:
    """Runs the (blocking, CPU-heavy) YAMNet + classifier inference in a
    worker thread so it never blocks the asyncio event loop - this matters
    for keeping other concurrent connections responsive.

    Returns raw class probabilities (not yet top-k'd), so callers can
    optionally smooth them (see ProbabilitySmoother) before turning them
    into displayed labels via predict_sound."""

    def _infer():
        t0 = time.perf_counter()
        processed, _ = preprocess_audio(waveform)
        t1 = time.perf_counter()
        _, embedding = extract_features(yamnet_model, processed)
        t2 = time.perf_counter()
        probabilities = predict_probabilities(embedding, classifier_model)
        t3 = time.perf_counter()

        print(
            f"[timing] preprocess={1000*(t1-t0):.1f}ms  "
            f"yamnet={1000*(t2-t1):.1f}ms  "
            f"classifier={1000*(t3-t2):.1f}ms  "
            f"total={1000*(t3-t0):.1f}ms"
        )
        return probabilities

    return await asyncio.to_thread(_infer)


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Spectra AI API is running"}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if classifier_model is None:
        raise HTTPException(status_code=500, detail="Model file not loaded on server.")

    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        waveform = load_audio_file(tmp_path)

        # Runs window-splitting + inference over the whole file in a worker
        # thread. This keeps the event loop free while a file is processing,
        # so it doesn't stall other requests or active mic websocket sessions
        # (which also run on this same event loop).
        def _infer_file():
            windows = split_audio_into_windows(
                waveform,
                sample_rate=16000,
                window_seconds=1.0,
                overlap=0.5
            )

            all_probabilities = []
            for window in windows:
                processed_window, _ = preprocess_audio(window)
                _, embedding = extract_features(yamnet_model, processed_window)
                probabilities = predict_probabilities(embedding, classifier_model)
                all_probabilities.append(probabilities)

            mean_probabilities = np.array(all_probabilities).mean(axis=0)
            return predict_sound(
                mean_probabilities,
                max_classes=3,
                confidence_threshold=0.20
            )

        results = await asyncio.to_thread(_infer_file)

        # Push into the rolling buffer so /predict/latest and /predict/recent
        # can serve this result instantly without recomputation.
        entry = await file_prediction_buffer.push(results, source="file")

        return {
            "filename": file.filename,
            "predictions": results,
            "timestamp": entry["timestamp"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/predict/latest")
async def get_latest_file_prediction():
    """Returns the most recent /predict result immediately - no processing,
    just a buffer read, so it's always fast regardless of how heavy
    inference is."""
    entry = await file_prediction_buffer.latest()
    if entry is None:
        raise HTTPException(status_code=404, detail="No predictions yet.")
    return entry


@app.get("/predict/recent")
async def get_recent_file_predictions(n: int = Query(10, ge=1, le=100)):
    return {"predictions": await file_prediction_buffer.recent(n)}


# --------------------------------------------------
# WebSocket: real-time microphone predictions
# --------------------------------------------------
@app.websocket("/predict-mic")
async def websocket_predict_mic(websocket: WebSocket):
    """
    Accepts a live stream of raw Float32 audio bytes from a client microphone.

    Design: the receive loop and the inference loop are decoupled.
    - The receive loop just reads bytes off the socket and drops them into a
      single-slot LatestValueSlot - this is O(1) and never blocks on
      inference, so the socket read buffer never backs up.
    - A separate background task continuously pulls the newest chunk from
      the slot, runs inference, stores the result in this session's rolling
      buffer, and pushes it to the client. If inference is momentarily
      slower than the mic's chunk rate, older chunks are simply skipped
      instead of queueing up - predictions stay close to real time instead
      of drifting further behind ("no lag").
    """
    if classifier_model is None:
        await websocket.close(code=1011, reason="Model file not loaded on server.")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    buffer = RollingPredictionBuffer(maxlen=200)
    slot = LatestValueSlot()
    audio_window = RollingAudioBuffer(window_size=16000)  # keep last ~1s of context
    smoother = ProbabilitySmoother(alpha=0.4)  # damps window-to-window flicker

    async def process_loop():
        while True:
            waveform = await slot.get()
            try:
                probabilities = await run_inference(waveform)
                smoothed = smoother.update(probabilities)
                results = predict_sound(smoothed, max_classes=3, confidence_threshold=0.20)

                # If nothing clears a stricter floor, report "no confident
                # sound" instead of forcing out a noisy top-3 guess where
                # every candidate is barely above the base threshold - this
                # is a real "no clear signal" state, not something
                # smoothing can or should paper over.
                top_confidence = results[0]["confidence"] if results else 0.0
                is_confident = top_confidence >= MIN_CONFIDENT_TOP_SCORE

                entry = await buffer.push(results, source="mic")
                await websocket.send_json({
                    "status": "processing",
                    "session_id": session_id,
                    "confident": is_confident,
                    "predictions": results if is_confident else [],
                    "timestamp": entry["timestamp"],
                })
            except Exception as e:
                try:
                    await websocket.send_json({"status": "error", "detail": str(e)})
                except Exception:
                    break

    worker_task = asyncio.create_task(process_loop())
    mic_sessions[session_id] = {"buffer": buffer, "slot": slot, "task": worker_task}

    print(f"Client connected to real-time audio stream via WebSocket (session={session_id}).")

    try:
        # Let the client know its session_id so it can poll /predict-mic/{session_id}/latest
        await websocket.send_json({"status": "connected", "session_id": session_id})

        while True:
            data = await websocket.receive_bytes()
            if not data:
                continue

            waveform = np.frombuffer(data, dtype=np.float32)
            if len(waveform) == 0:
                continue

            # Fold the new (small, frequent) chunk into the rolling ~1s
            # window, so inference always runs on a full window of audio
            # even though updates arrive faster than that.
            windowed_waveform = audio_window.push_chunk(waveform)

            # Never blocks: just overwrites the pending chunk if the
            # consumer hasn't caught up yet.
            slot.put(windowed_waveform)

    except WebSocketDisconnect:
        print(f"Microphone stream client disconnected (session={session_id}).")

    except Exception as e:
        print(f"Error handling WebSocket stream (session={session_id}): {str(e)}")
        try:
            await websocket.send_json({"status": "error", "detail": str(e)})
        except Exception:
            pass  # Connection might already be broken

    finally:
        worker_task.cancel()
        mic_sessions.pop(session_id, None)


@app.get("/predict-mic/{session_id}/latest")
async def get_latest_mic_prediction(session_id: str):
    """Low-latency polling alternative to the WebSocket push: always returns
    whatever is currently sitting in that session's rolling buffer, instantly."""
    session = mic_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or closed session_id.")
    entry = await session["buffer"].latest()
    if entry is None:
        raise HTTPException(status_code=404, detail="No predictions yet for this session.")
    return entry


@app.get("/predict-mic/{session_id}/recent")
async def get_recent_mic_predictions(session_id: str, n: int = Query(10, ge=1, le=200)):
    session = mic_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or closed session_id.")
    return {"predictions": await session["buffer"].recent(n)}
