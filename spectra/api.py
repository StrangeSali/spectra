import os
import time
import asyncio
import tempfile
import uuid

from collections import deque
from typing import Optional, List, Dict, Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Query,
)
from google.cloud import storage
from tf_keras.models import load_model

from spectra.processing.audio import (
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio,
)

from spectra.processing.yamnet_utils import (
    extract_features,
    load_yamnet_class_names,
    detect_speech,
)

from spectra.processing.classifier import (
    predict_probabilities,
    predict_sound,
)


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()


# ==================================================
# TENSORFLOW CONFIGURATION
# ==================================================

tf.config.threading.set_intra_op_parallelism_threads(
    os.cpu_count() or 4
)

tf.config.threading.set_inter_op_parallelism_threads(
    2
)


# ==================================================
# FASTAPI
# ==================================================

app = FastAPI(
    title="Spectra Audio Classifier API"
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_BUCKET_NAME = os.getenv(
    "MODEL_BUCKET_NAME"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME"
)

LOCAL_MODEL_PATH = (
    "dense_50.keras"
)


# ==================================================
# AUDIO CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000

AUDIO_WINDOW_SIZE = 16000


# ==================================================
# HYBRID MODEL CONFIGURATION
# ==================================================

YAMNET_SPEECH_THRESHOLD = float(
    os.getenv(
        "YAMNET_SPEECH_THRESHOLD",
        "0.30",
    )
)

MIN_CONFIDENT_TOP_SCORE = float(
    os.getenv(
        "MIN_CONFIDENT_TOP_SCORE",
        "0.30",
    )
)


# ==================================================
# ADAPTIVE QUIET DETECTION
# ==================================================

CALIBRATION_SAMPLES = 5

ABSOLUTE_MIN_RMS = 0.0025

NOISE_MULTIPLIER = 1.35

NOISE_MARGIN = 0.0005


# ==================================================
# LOAD YAMNET
# ==================================================

print(
    "Loading YAMNet..."
)

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

yamnet_class_names = (
    load_yamnet_class_names(
        yamnet_model
    )
)

print(
    f"YAMNet loaded "
    f"({len(yamnet_class_names)} classes)."
)


# ==================================================
# LOAD ESC-50 CLASSIFIER
# ==================================================

print(
    "Loading classifier..."
)

classifier_model = None


if MODEL_BUCKET_NAME and MODEL_NAME:

    try:

        print(
            f"Downloading model from GCS: "
            f"{MODEL_BUCKET_NAME}/{MODEL_NAME}"
        )

        storage_client = (
            storage.Client()
        )

        bucket = storage_client.bucket(
            MODEL_BUCKET_NAME
        )

        blob = bucket.blob(
            MODEL_NAME
        )

        temp_model_path = (
            "/tmp/dense_50.keras"
        )

        blob.download_to_filename(
            temp_model_path
        )

        classifier_model = load_model(
            temp_model_path
        )

        print(
            "Classifier loaded from GCS."
        )

    except Exception as error:

        print(
            "Could not load classifier "
            f"from GCS: {error}"
        )


if (
    classifier_model is None
    and os.path.exists(
        LOCAL_MODEL_PATH
    )
):

    print(
        f"Loading local classifier: "
        f"{LOCAL_MODEL_PATH}"
    )

    classifier_model = load_model(
        LOCAL_MODEL_PATH
    )


if classifier_model is None:

    print(
        "WARNING: classifier model "
        "could not be loaded."
    )


# ==================================================
# ROLLING PREDICTION BUFFER
# ==================================================

class RollingPredictionBuffer:

    def __init__(
        self,
        maxlen=50,
    ):

        self._buffer = deque(
            maxlen=maxlen
        )

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


# ==================================================
# LATEST VALUE SLOT
# ==================================================

class LatestValueSlot:

    def __init__(self):

        self._item = None

        self._event = (
            asyncio.Event()
        )


    def put(
        self,
        item,
    ):

        self._item = item

        self._event.set()


    async def get(self):

        await self._event.wait()

        item = self._item

        self._item = None

        self._event.clear()

        return item


# ==================================================
# ROLLING AUDIO BUFFER
# ==================================================

class RollingAudioBuffer:

    def __init__(
        self,
        window_size=AUDIO_WINDOW_SIZE,
    ):

        self.window_size = (
            window_size
        )

        self._buf = np.zeros(
            window_size,
            dtype=np.float32,
        )


    def push_chunk(
        self,
        chunk,
    ):

        chunk = np.asarray(
            chunk,
            dtype=np.float32,
        ).reshape(-1)

        n = len(
            chunk
        )

        if n >= self.window_size:

            self._buf = (
                chunk[
                    -self.window_size:
                ].copy()
            )

        else:

            self._buf = np.concatenate(
                [
                    self._buf[n:],
                    chunk,
                ]
            )

        return self._buf.copy()


# ==================================================
# PROBABILITY SMOOTHER
# ==================================================

class ProbabilitySmoother:

    def __init__(
        self,
        alpha=0.4,
    ):

        self.alpha = alpha

        self._smoothed = None


    def update(
        self,
        probabilities,
    ):

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if self._smoothed is None:

            self._smoothed = (
                probabilities.copy()
            )

        else:

            self._smoothed = (
                self.alpha
                * probabilities
                + (
                    1
                    - self.alpha
                )
                * self._smoothed
            )

        return self._smoothed


# ==================================================
# ADAPTIVE NOISE GATE
# ==================================================

class AdaptiveNoiseGate:

    def __init__(self):

        self.calibration_values = []

        self.noise_floor = None

        self.threshold = (
            ABSOLUTE_MIN_RMS
        )


    @property
    def calibrated(self):

        return (
            self.noise_floor
            is not None
        )


    def update(
        self,
        rms,
    ):

        rms = float(
            rms
        )

        # ------------------------------------------
        # INITIAL CALIBRATION
        # ------------------------------------------

        if not self.calibrated:

            self.calibration_values.append(
                rms
            )

            if (
                len(
                    self.calibration_values
                )
                >= CALIBRATION_SAMPLES
            ):

                self.noise_floor = float(
                    np.median(
                        self.calibration_values
                    )
                )

                self._update_threshold()

            return {
                "calibrating":
                    not self.calibrated,

                "quiet":
                    True,

                "threshold":
                    self.threshold,
            }


        # ------------------------------------------
        # NORMAL OPERATION
        # ------------------------------------------

        quiet = (
            rms
            <= self.threshold
        )


        if quiet:

            self.noise_floor = (
                0.95
                * self.noise_floor
                + 0.05
                * rms
            )

            self._update_threshold()


        return {
            "calibrating":
                False,

            "quiet":
                quiet,

            "threshold":
                self.threshold,
        }


    def _update_threshold(self):

        self.threshold = max(
            ABSOLUTE_MIN_RMS,

            self.noise_floor
            * NOISE_MULTIPLIER
            + NOISE_MARGIN,
        )


# ==================================================
# GLOBAL BUFFERS
# ==================================================

file_prediction_buffer = (
    RollingPredictionBuffer(
        maxlen=100
    )
)

mic_sessions: Dict[
    str,
    Dict[str, Any],
] = {}


# ==================================================
# HYBRID MODEL
# ==================================================

def hybrid_predict(
    esc50_probabilities,
    yamnet_scores,
    max_classes=3,
):

    (
        is_speech,
        speech_confidence,
        yamnet_speech_class,
    ) = detect_speech(
        yamnet_scores,
        yamnet_class_names,
        threshold=(
            YAMNET_SPEECH_THRESHOLD
        ),
    )


    # ----------------------------------------------
    # YAMNET SPEECH SPECIALIST
    # ----------------------------------------------

    if is_speech:

        return [
            {
                "class_name":
                    "speech",

                "raw_class_name":
                    yamnet_speech_class,

                "display_label":
                    "People talking",

                "category":
                    "Human",

                "confidence":
                    round(
                        float(
                            speech_confidence
                        ),
                        2,
                    ),

                "source":
                    "yamnet",
            }
        ]


    # ----------------------------------------------
    # ESC-50 ENVIRONMENTAL CLASSIFIER
    # ----------------------------------------------

    results = predict_sound(
        esc50_probabilities,
        max_classes=max_classes,
        confidence_threshold=0.0,
    )


    for result in results:

        result["source"] = (
            "esc50"
        )


    return results


# ==================================================
# INFERENCE
# ==================================================

async def run_inference(
    waveform,
):

    def _infer():

        t0 = time.perf_counter()


        processed, rms = (
            preprocess_audio(
                waveform
            )
        )

        t1 = time.perf_counter()


        (
            yamnet_scores,
            embedding,
        ) = extract_features(
            yamnet_model,
            processed,
        )

        t2 = time.perf_counter()


        probabilities = (
            predict_probabilities(
                embedding,
                classifier_model,
            )
        )

        t3 = time.perf_counter()


        print(
            f"[timing] "
            f"preprocess="
            f"{1000*(t1-t0):.1f}ms  "
            f"yamnet="
            f"{1000*(t2-t1):.1f}ms  "
            f"classifier="
            f"{1000*(t3-t2):.1f}ms  "
            f"total="
            f"{1000*(t3-t0):.1f}ms"
        )


        return (
            probabilities,
            yamnet_scores,
            float(rms),
        )


    return await asyncio.to_thread(
        _infer
    )


# ==================================================
# ROOT
# ==================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message":
            "Spectra AI API is running",
        "mode":
            "Hybrid YAMNet + ESC-50",
    }


# ==================================================
# FILE PREDICTION
# ==================================================

@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...),
):

    if classifier_model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model file not loaded "
                "on server."
            ),
        )


    tmp_path = None


    try:

        suffix = (
            os.path.splitext(
                file.filename
            )[1]
            or ".wav"
        )


        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as tmp:

            contents = await file.read()

            tmp.write(
                contents
            )

            tmp_path = tmp.name


        waveform = load_audio_file(
            tmp_path
        )


        def _infer_file():

            windows = (
                split_audio_into_windows(
                    waveform,
                    sample_rate=16000,
                    window_seconds=1.0,
                    overlap=0.5,
                )
            )


            all_probabilities = []

            all_yamnet_scores = []


            for window in windows:

                (
                    processed_window,
                    _,
                ) = preprocess_audio(
                    window
                )


                (
                    yamnet_scores,
                    embedding,
                ) = extract_features(
                    yamnet_model,
                    processed_window,
                )


                probabilities = (
                    predict_probabilities(
                        embedding,
                        classifier_model,
                    )
                )


                all_probabilities.append(
                    probabilities
                )


                mean_scores = (
                    np.asarray(
                        yamnet_scores
                    ).mean(
                        axis=0
                    )
                )


                all_yamnet_scores.append(
                    mean_scores
                )


            if not all_probabilities:

                return []


            mean_probabilities = (
                np.asarray(
                    all_probabilities
                ).mean(
                    axis=0
                )
            )


            mean_yamnet_scores = (
                np.asarray(
                    all_yamnet_scores
                ).mean(
                    axis=0
                )
            )


            return hybrid_predict(
                mean_probabilities,
                mean_yamnet_scores,
                max_classes=3,
            )


        results = await asyncio.to_thread(
            _infer_file
        )


        entry = (
            await file_prediction_buffer.push(
                results,
                source="file",
            )
        )


        return {
            "filename":
                file.filename,

            "predictions":
                results,

            "timestamp":
                entry["timestamp"],
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        )


    finally:

        if (
            tmp_path
            and os.path.exists(
                tmp_path
            )
        ):

            os.remove(
                tmp_path
            )


# ==================================================
# FILE HISTORY
# ==================================================

@app.get("/predict/latest")
async def get_latest_file_prediction():

    entry = (
        await file_prediction_buffer.latest()
    )

    if entry is None:

        raise HTTPException(
            status_code=404,
            detail="No predictions yet.",
        )

    return entry


@app.get("/predict/recent")
async def get_recent_file_predictions(
    n: int = Query(
        10,
        ge=1,
        le=100,
    ),
):

    return {
        "predictions":
            await file_prediction_buffer.recent(
                n
            )
    }


# ==================================================
# WEBSOCKET MICROPHONE
# ==================================================

@app.websocket("/predict-mic")
async def websocket_predict_mic(
    websocket: WebSocket,
):

    if classifier_model is None:

        await websocket.close(
            code=1011,
            reason=(
                "Model file not loaded "
                "on server."
            ),
        )

        return


    await websocket.accept()


    session_id = str(
        uuid.uuid4()
    )


    buffer = RollingPredictionBuffer(
        maxlen=200
    )


    slot = LatestValueSlot()


    audio_window = RollingAudioBuffer(
        window_size=AUDIO_WINDOW_SIZE
    )


    smoother = ProbabilitySmoother(
        alpha=0.4
    )


    noise_gate = AdaptiveNoiseGate()


    # ==================================================
    # INFERENCE LOOP
    # ==================================================

    async def process_loop():

        while True:

            waveform = await slot.get()


            try:

                (
                    probabilities,
                    yamnet_scores,
                    rms,
                ) = await run_inference(
                    waveform
                )


                gate = noise_gate.update(
                    rms
                )


                # --------------------------------------
                # INITIAL CALIBRATION
                # --------------------------------------

                if gate["calibrating"]:

                    await websocket.send_json(
                        {
                            "status":
                                "calibrating",

                            "session_id":
                                session_id,

                            "rms":
                                rms,

                            "predictions":
                                [],
                        }
                    )

                    continue


                # --------------------------------------
                # QUIET
                # --------------------------------------

                if gate["quiet"]:

                    results = []

                    entry = await buffer.push(
                        results,
                        source="mic",
                    )


                    await websocket.send_json(
                        {
                            "status":
                                "processing",

                            "session_id":
                                session_id,

                            "confident":
                                False,

                            "quiet":
                                True,

                            "rms":
                                rms,

                            "quiet_threshold":
                                gate["threshold"],

                            "predictions":
                                [],

                            "timestamp":
                                entry["timestamp"],
                        }
                    )

                    continue


                # --------------------------------------
                # ENVIRONMENTAL PROBABILITY SMOOTHING
                # --------------------------------------

                smoothed = smoother.update(
                    probabilities
                )


                # --------------------------------------
                # HYBRID CLASSIFICATION
                # --------------------------------------

                results = hybrid_predict(
                    smoothed,
                    yamnet_scores,
                    max_classes=3,
                )


                top_confidence = (
                    float(
                        results[0][
                            "confidence"
                        ]
                    )
                    if results
                    else 0.0
                )


                is_confident = (
                    top_confidence
                    >= MIN_CONFIDENT_TOP_SCORE
                )


                displayed_results = (
                    results
                    if is_confident
                    else []
                )


                entry = await buffer.push(
                    displayed_results,
                    source="mic",
                )


                await websocket.send_json(
                    {
                        "status":
                            "processing",

                        "session_id":
                            session_id,

                        "confident":
                            is_confident,

                        "quiet":
                            False,

                        "rms":
                            rms,

                        "quiet_threshold":
                            gate["threshold"],

                        "predictions":
                            displayed_results,

                        "timestamp":
                            entry["timestamp"],
                    }
                )


            except Exception as error:

                try:

                    await websocket.send_json(
                        {
                            "status":
                                "error",

                            "detail":
                                str(error),
                        }
                    )

                except Exception:

                    break


    worker_task = asyncio.create_task(
        process_loop()
    )


    mic_sessions[
        session_id
    ] = {
        "buffer":
            buffer,

        "slot":
            slot,

        "task":
            worker_task,
    }


    print(
        "Microphone client connected "
        f"(session={session_id})."
    )


    try:

        await websocket.send_json(
            {
                "status":
                    "connected",

                "session_id":
                    session_id,
            }
        )


        while True:

            data = (
                await websocket.receive_bytes()
            )


            if not data:
                continue


            waveform = np.frombuffer(
                data,
                dtype=np.float32,
            )


            if len(waveform) == 0:
                continue


            # Browser audio must already have been
            # resampled to 16 kHz by Streamlit.

            windowed_waveform = (
                audio_window.push_chunk(
                    waveform
                )
            )


            slot.put(
                windowed_waveform
            )


    except WebSocketDisconnect:

        print(
            "Microphone client disconnected "
            f"(session={session_id})."
        )


    except Exception as error:

        print(
            "WebSocket error "
            f"(session={session_id}): "
            f"{error}"
        )


    finally:

        worker_task.cancel()

        mic_sessions.pop(
            session_id,
            None,
        )


# ==================================================
# MICROPHONE HISTORY
# ==================================================

@app.get(
    "/predict-mic/{session_id}/latest"
)
async def get_latest_mic_prediction(
    session_id: str,
):

    session = mic_sessions.get(
        session_id
    )


    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown or closed "
                "session_id."
            ),
        )


    entry = await session[
        "buffer"
    ].latest()


    if entry is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "No predictions yet "
                "for this session."
            ),
        )


    return entry


@app.get(
    "/predict-mic/{session_id}/recent"
)
async def get_recent_mic_predictions(
    session_id: str,

    n: int = Query(
        10,
        ge=1,
        le=200,
    ),
):

    session = mic_sessions.get(
        session_id
    )


    if session is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown or closed "
                "session_id."
            ),
        )


    return {
        "predictions":
            await session[
                "buffer"
            ].recent(
                n
            )
    }
