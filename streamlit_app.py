import os
import queue
import time

import av
import numpy as np
import requests
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from spectra.graphics.renderer import render_frame

from spectra.processing.categories import (
    SOUNDS_DICT,
    DEFAULT_CATEGORY,
)


# ==================================================
# CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000

TARGET_FPS = 12
FRAME_DURATION = 1 / TARGET_FPS

# How often we ask the API for its latest prediction.
POLL_INTERVAL = 0.25


# ==================================================
# API CONFIGURATION
# ==================================================

API_BASE_URL = os.getenv(
    "SPECTRA_API_URL",
    "https://spectra-1087886990522.europe-west1.run.app",
)


# --------------------------------------------------
# LIVE AUDIO ENDPOINT
#
# IMPORTANT:
# This endpoint still needs to be implemented
# and deployed by the API team.
# --------------------------------------------------

CHUNK_URL = (
    f"{API_BASE_URL}/predict"
)


# --------------------------------------------------
# LATEST PREDICTION ENDPOINT
#
# This endpoint already exists.
# --------------------------------------------------

RECENT_URL = (
    f"{API_BASE_URL}/recent?n=1"
)


# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="Spectra AI",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ==================================================
# MOBILE CSS
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 500px;
        padding-top: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    h1 {
        text-align: center;
    }

    video {
        display: none !important;
    }

    audio {
        width: 100% !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.title("Spectra AI")

st.caption("Real-Time Environmental Audio Visualizer")


# ==================================================
# STREAMLIT SESSION STATE
# ==================================================

if "latest_predictions" not in st.session_state:

    st.session_state[
        "latest_predictions"
    ] = []


if "latest_rms" not in st.session_state:

    st.session_state[
        "latest_rms"
    ] = 0.0


if "latest_timestamp" not in st.session_state:

    st.session_state[
        "latest_timestamp"
    ] = None


if "last_poll_time" not in st.session_state:

    st.session_state[
        "last_poll_time"
    ] = 0.0


# ==================================================
# AUDIO QUEUE
#
# WebRTC receives microphone frames in its own
# processing thread.
#
# The audio processor puts those frames here.
#
# The main Streamlit controller then sends them
# to the API.
# ==================================================

audio_queue = queue.Queue(
    maxsize=20
)


# ==================================================
# API -> GRAPHICS ADAPTER
# ==================================================

def adapt_predictions(
    api_predictions,
):
    """
    Convert API predictions into the structure
    expected by renderer.py.

    Example API result:

        {
            "class_name": "frog",
            "confidence": 0.42
        }

    Becomes:

        {
            "category": "Animal",
            "display_label": "Animal",
            "confidence": 0.42
        }

    Only the strongest prediction for each broad
    Spectra visual category is kept.
    """

    if not api_predictions:

        return []


    best_by_category = {}


    for prediction in api_predictions:

        class_name = prediction.get(
            "class_name",
            "",
        )


        if not class_name:

            continue


        # --------------------------------------------------
        # CLASS -> VISUAL CATEGORY
        # --------------------------------------------------

        category = prediction.get(
            "category"
        )


        if not category:

            category = SOUNDS_DICT.get(
                class_name,
                DEFAULT_CATEGORY,
            )


        if category == DEFAULT_CATEGORY:

            continue


        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        try:

            confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0


        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )


        # --------------------------------------------------
        # DISPLAY LABEL
        # --------------------------------------------------

        display_label = prediction.get(
            "display_label",
            category,
        )


        # --------------------------------------------------
        # ONE ICON PER BROAD CATEGORY
        # --------------------------------------------------

        current = best_by_category.get(
            category
        )


        if (
            current is None
            or confidence
            > current[
                "confidence"
            ]
        ):

            best_by_category[
                category
            ] = {

                "category":
                    category,

                "display_label":
                    display_label,

                "confidence":
                    confidence,
            }


    adapted = list(
        best_by_category.values()
    )


    # Highest-confidence category becomes hero.
    adapted.sort(
        key=lambda prediction:
            prediction[
                "confidence"
            ],
        reverse=True,
    )


    # Renderer supports maximum 3 icons.
    return adapted[:3]


# ==================================================
# WEBRTC AUDIO PROCESSOR
#
# Browser microphone
#       ↓
# WebRTC
#       ↓
# SpectraAudioProcessor.recv()
#       ↓
# resample to 16 kHz mono Float32
#       ↓
# calculate RMS
#       ↓
# audio_queue
#
# IMPORTANT:
# No HTTP request is made inside recv().
# ==================================================

class SpectraAudioProcessor(
    AudioProcessorBase
):

    def __init__(
        self
    ):

        self.latest_rms = 0.0


        # Browser microphones commonly provide
        # audio around 48 kHz.
        #
        # Spectra expects 16 kHz mono.

        self.resampler = (
            av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=SAMPLE_RATE,
            )
        )


    def recv(
        self,
        frame: av.AudioFrame,
    ) -> av.AudioFrame:

        try:

            # ==================================================
            # 1. RESAMPLE
            # ==================================================

            resampled_frames = (
                self.resampler.resample(
                    frame
                )
            )


            if not resampled_frames:

                return frame


            chunks = []


            for resampled_frame in (
                resampled_frames
            ):

                audio = (
                    resampled_frame
                    .to_ndarray()
                )


                audio = np.asarray(
                    audio,
                    dtype=np.float32,
                ).reshape(-1)


                if len(audio) > 0:

                    chunks.append(
                        audio
                    )


            if not chunks:

                return frame


            audio = np.concatenate(
                chunks
            )


            # ==================================================
            # 2. SAFETY NORMALIZATION
            # ==================================================

            peak = float(
                np.max(
                    np.abs(
                        audio
                    )
                )
            )


            if peak > 1.0:

                audio = (
                    audio / peak
                )


            # ==================================================
            # 3. LOCAL RMS
            #
            # RMS is used only for the visualization.
            # ==================================================

            self.latest_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            audio
                        )
                    )
                )
            )


            # ==================================================
            # 4. QUEUE AUDIO
            #
            # Do not block the WebRTC processing thread.
            # ==================================================

            try:

                audio_queue.put_nowait(
                    audio
                )


            except queue.Full:

                # If we're falling behind, discard
                # the oldest audio so the visualization
                # remains close to real time.

                try:

                    audio_queue.get_nowait()

                except queue.Empty:

                    pass


                try:

                    audio_queue.put_nowait(
                        audio
                    )

                except queue.Full:

                    pass


        except Exception as error:

            print(
                "Audio processor error:",
                error,
            )


        # SENDONLY still expects us to return the
        # original incoming frame.
        return frame


# ==================================================
# WEBRTC MICROPHONE
# ==================================================

webrtc_ctx = webrtc_streamer(

    key="spectra-microphone",

    mode=WebRtcMode.SENDONLY,

    media_stream_constraints={

        "video":
            False,

        "audio": {

            "echoCancellation":
                True,

            "noiseSuppression":
                False,

            "autoGainControl":
                False,
        },
    },

    audio_processor_factory=(
        SpectraAudioProcessor
    ),

    async_processing=True,
)


# ==================================================
# SEND AUDIO CHUNK TO API
# ==================================================

def send_chunk(
    audio,
):
    """
    Send one 16 kHz mono Float32 audio chunk
    to the Spectra API.

    This corresponds to the teacher's:

        send(chunk)

    The backend still needs:

        POST /predict-chunk
    """

    if audio is None:

        return False


    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)


    if len(audio) == 0:

        return False


    try:
        #post(url,
        # files={"file": (filename, chunk, "audio/wav")},
        #timeout=timeout,
        #)
        requests.post(
            CHUNK_URL,
            data=audio.tobytes(),
            files={"file": ("chunk.wav", chunk, "audio/wav")},
            timeout=10,
        )
        return True


    except requests.RequestException as error:

        st.write(
            "Audio chunk send error:",
            error,
        )


        return False


# ==================================================
# ASK API FOR LATEST PREDICTION
# ==================================================

def ask_api():
    """
    Retrieve the newest prediction already
    calculated by the API.

    Current endpoint:

        GET /recent?n=1
    """

    try:

        response = requests.get(
            RECENT_URL,
            timeout=10,
        )


        response.raise_for_status()


        data = response.json()


        # Current API response:
        #
        # {
        #     "predictions": [
        #         {
        #             "timestamp": ...,
        #             "source": ...,
        #             "predictions": [...]
        #         }
        #     ]
        # }

        history = data.get(
            "predictions",
            [],
        )


        if not history:

            return None


        return history[-1]


    except requests.RequestException as error:

        print(
            "Prediction polling error:",
            error,
        )


        return None


# ==================================================
# STREAMLIT PLACEHOLDERS
# ==================================================

status_placeholder = (
    st.empty()
)

frame_placeholder = (
    st.empty()
)


# ==================================================
# LIVE CONTROLLER LOOP
#
# Teacher's architecture:
#
# 1. WebRTC receives microphone audio
#
# 2. send_chunk()
#       ↓
#    POST /predict-chunk
#
# 3. ask_api()
#       ↓
#    GET /recent
#
# 4. render_frame()
#
# 5. Display with Streamlit
#
# 6. Repeat at steady FPS
# ==================================================

while True:

    frame_start = (
        time.perf_counter()
    )


    # ==================================================
    # 1. SEND AVAILABLE MICROPHONE AUDIO
    # ==================================================

    if webrtc_ctx.state.playing:

        chunks_sent = 0


        # Don't spend an unlimited amount of time
        # emptying an audio backlog during one frame.

        while True:

            try:

                chunk = (
                    audio_queue
                    .get_nowait()
                )

            except queue.Empty:

                break


            send_chunk(
                chunk
            )


            chunks_sent += 1


    # ==================================================
    # 2. GET CURRENT RMS
    # ==================================================

    rms = (
        st.session_state[
            "latest_rms"
        ]
    )


    if (
        webrtc_ctx.audio_processor
        is not None
    ):

        rms = float(
            webrtc_ctx
            .audio_processor
            .latest_rms
        )


        st.session_state[
            "latest_rms"
        ] = rms


    # ==================================================
    # 3. POLL API
    # ==================================================

    now = (
        time.perf_counter()
    )


    if (
        now
        - st.session_state[
            "last_poll_time"
        ]
        >= POLL_INTERVAL
    ):

        result = ask_api()


        if result is not None:

            result_timestamp = (
                result.get(
                    "timestamp"
                )
            )


            # Only process a genuinely new
            # prediction from the API.

            if (
                result_timestamp
                != st.session_state[
                    "latest_timestamp"
                ]
            ):

                raw_predictions = (
                    result.get(
                        "predictions",
                        [],
                    )
                )


                predictions = (
                    adapt_predictions(
                        raw_predictions
                    )
                )


                st.session_state[
                    "latest_predictions"
                ] = predictions


                st.session_state[
                    "latest_timestamp"
                ] = (
                    result_timestamp
                )


        st.session_state[
            "last_poll_time"
        ] = now


    # ==================================================
    # 4. CURRENT DISPLAY PREDICTIONS
    # ==================================================

    predictions = (
        st.session_state[
            "latest_predictions"
        ]
    )


    # ==================================================
    # 5. STATUS
    # ==================================================

    if not webrtc_ctx.state.playing:

        status_placeholder.info(
            "Start the microphone to begin listening."
        )


    elif predictions:

        primary = (
            predictions[0]
        )


        status_placeholder.success(
            f"Detected: "
            f"{primary['display_label']} "
            f"— "
            f"{primary['confidence'] * 100:.0f}%"
        )


    else:

        status_placeholder.caption(
            "Listening…"
        )


    # ==================================================
    # 6. RENDER
    # ==================================================

    frame = render_frame(
        predictions,
        rms,
    )


    # Streamlit 2026 syntax:
    #
    # OLD:
    # use_container_width=True
    #
    # NEW:
    # width="stretch"

    frame_placeholder.image(
        frame,
        channels="RGB",
        width="stretch",
    )


    # ==================================================
    # 7. KEEP FRAME RATE STEADY
    # ==================================================

    elapsed = (
        time.perf_counter()
        - frame_start
    )


    sleep_time = (
        FRAME_DURATION
        - elapsed
    )


    if sleep_time > 0:

        time.sleep(
            sleep_time
        )
