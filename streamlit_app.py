# ==================================================
# IMPORTS
# ==================================================

import json
import os
import threading
import time

import av
import numpy as np
import requests
import streamlit as st
import websocket

from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from spectra.graphics.renderer import render_frame


# ==================================================
# CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000

TARGET_FPS = 12
FRAME_DURATION = 1 / TARGET_FPS

# How often we ask the API for its latest prediction.
POLL_INTERVAL = 0.25


# ==================================================
# API URLS
# ==================================================

# Deployed API
API_BASE_URL = os.getenv(
    "SPECTRA_API_URL",
    "https://spectra-1087886990522.europe-west1.run.app",
)

# WebSocket used to SEND live microphone audio.
SPECTRA_WS_URL = os.getenv(
    "SPECTRA_WS_URL",
    API_BASE_URL.replace("https://", "wss://")
    .replace("http://", "ws://")
    + "/predict-mic",
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

st.caption("Real-time sound analysis")


# ==================================================
# API -> GRAPHICS ADAPTER
# ==================================================

def adapt_predictions(api_predictions):
    """
    Convert API predictions into the simple structure
    expected by renderer.py.

    We keep only the strongest prediction for each
    visual category.
    """

    if not api_predictions:
        return []


    best_by_category = {}


    for prediction in api_predictions:

        category = prediction.get(
            "category",
            "Background",
        )


        if category == "Background":
            continue


        try:

            confidence = float(
                prediction.get(
                    "confidence",
                    0.0,
                )
            )

        except (TypeError, ValueError):

            confidence = 0.0


        display_label = prediction.get(
            "display_label",
            prediction.get(
                "class_name",
                category,
            ),
        )


        current = best_by_category.get(
            category
        )


        if (
            current is None
            or confidence > current["confidence"]
        ):

            best_by_category[category] = {

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


    adapted.sort(
        key=lambda prediction:
            prediction["confidence"],
        reverse=True,
    )


    if not adapted:
        return []


    # --------------------------------------------------
    # PRIMARY SOUND
    # --------------------------------------------------

    displayed = [
        adapted[0]
    ]


    # --------------------------------------------------
    # SECONDARY SOUNDS
    #
    # Only show secondary sounds if confidence >= 40%.
    # --------------------------------------------------

    for prediction in adapted[1:]:

        if (
            prediction["confidence"]
            >= 0.40
        ):

            displayed.append(
                prediction
            )


        if len(displayed) >= 3:
            break


    return displayed


# ==================================================
# WEBSOCKET CLIENT
#
# RESPONSIBILITY:
#
# Browser audio
#      ↓
# WebSocket
#      ↓
# /predict-mic
#
# Predictions themselves will be POLLED separately.
# ==================================================

class SpectraWebSocketClient:

    def __init__(self, url):

        self.url = url

        self.ws = None
        self.thread = None

        self.session_id = None
        self.connected = False
        self.status = "connecting"

        self.lock = threading.Lock()


    # ==================================================
    # CONNECT
    # ==================================================

    def connect(self):

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            return


        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )


        self.thread = threading.Thread(
            target=self.ws.run_forever,
            daemon=True,
        )


        self.thread.start()


    # ==================================================
    # CALLBACKS
    # ==================================================

    def _on_open(
        self,
        ws,
    ):

        with self.lock:

            self.connected = True
            self.status = "connected"


        print(
            "Connected to Spectra live API"
        )


    def _on_message(
        self,
        ws,
        message,
    ):
        """
        The API sends WebSocket messages too.

        For this architecture we mainly use them to
        obtain the session_id.

        Predictions are deliberately fetched separately
        using GET /predict-mic/{session_id}/latest.
        """

        try:

            data = json.loads(
                message
            )

        except Exception as error:

            print(
                "Invalid WebSocket message:",
                error,
            )

            return


        status = data.get(
            "status"
        )


        # --------------------------------------------------
        # INITIAL CONNECTION
        # --------------------------------------------------

        if status == "connected":

            with self.lock:

                self.session_id = data.get(
                    "session_id"
                )

                self.status = "connected"


            print(
                "Spectra session:",
                self.session_id,
            )


        # --------------------------------------------------
        # API ERROR
        # --------------------------------------------------

        elif status == "error":

            with self.lock:

                self.status = "error"


            print(
                "Spectra API error:",
                data.get(
                    "detail"
                ),
            )


        # --------------------------------------------------
        # PROCESSING
        #
        # We deliberately DON'T store predictions here.
        #
        # They will be retrieved using HTTP GET.
        # --------------------------------------------------

        elif status == "processing":

            with self.lock:

                self.status = "processing"


    # ==================================================
    # CLOSE / ERROR
    # ==================================================

    def _on_close(
        self,
        ws,
        close_status_code,
        close_msg,
    ):

        with self.lock:

            self.connected = False
            self.status = "disconnected"


        print(
            "Disconnected from Spectra API"
        )


    def _on_error(
        self,
        ws,
        error,
    ):

        with self.lock:

            self.status = "error"


        print(
            "Spectra WebSocket error:",
            error,
        )


    # ==================================================
    # SEND AUDIO
    # ==================================================

    def send_audio(
        self,
        audio,
    ):

        if not self.connected:
            return


        if self.ws is None:
            return


        audio = np.asarray(
            audio,
            dtype=np.float32,
        ).reshape(-1)


        if len(audio) == 0:
            return


        try:

            self.ws.send(
                audio.tobytes(),

                opcode=(
                    websocket
                    .ABNF
                    .OPCODE_BINARY
                ),
            )


        except Exception as error:

            print(
                "Audio send error:",
                error,
            )


    # ==================================================
    # STATE
    # ==================================================

    def get_state(self):

        with self.lock:

            return {

                "session_id":
                    self.session_id,

                "status":
                    self.status,

                "connected":
                    self.connected,
            }


    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        if self.ws is not None:

            try:
                self.ws.close()

            except Exception:
                pass


        with self.lock:

            self.connected = False


# ==================================================
# ONE WEBSOCKET CLIENT PER STREAMLIT SESSION
# ==================================================

if (
    "spectra_ws_client"
    not in st.session_state
):

    st.session_state[
        "spectra_ws_client"
    ] = SpectraWebSocketClient(
        SPECTRA_WS_URL
    )


ws_client = st.session_state[
    "spectra_ws_client"
]


ws_client.connect()


# ==================================================
# STREAMLIT STATE
# ==================================================

if "latest_predictions" not in st.session_state:

    st.session_state[
        "latest_predictions"
    ] = []


if "latest_rms" not in st.session_state:

    st.session_state[
        "latest_rms"
    ] = 0.0


if "last_poll_time" not in st.session_state:

    st.session_state[
        "last_poll_time"
    ] = 0.0


if "latest_prediction_timestamp" not in st.session_state:

    st.session_state[
        "latest_prediction_timestamp"
    ] = None


# ==================================================
# AUDIO PROCESSOR
#
# Browser microphone
#      ↓
# usually ~48 kHz
#      ↓
# PyAV resampling
#      ↓
# 16 kHz Float32
#      ↓
# WebSocket
# ==================================================

class SpectraAudioProcessor(
    AudioProcessorBase
):

    def __init__(self):

        self.latest_rms = 0.0


        self.resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=SAMPLE_RATE,
        )


    def recv(
        self,
        frame: av.AudioFrame,
    ) -> av.AudioFrame:

        try:

            # ==================================================
            # 1. RESAMPLE TO 16 kHz
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
                    audio
                    / peak
                )


            # ==================================================
            # 3. LOCAL RMS
            #
            # RMS is visual information.
            #
            # The current API WebSocket does not include
            # RMS in the latest-buffer endpoint, so we can
            # calculate it locally for renderer.py.
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
            # 4. SEND AUDIO TO API
            # ==================================================

            ws_client.send_audio(
                audio
            )


        except Exception as error:

            print(
                "Audio processor error:",
                error,
            )


        return frame


# ==================================================
# MICROPHONE
# ==================================================

webrtc_ctx = webrtc_streamer(

    key="spectra-microphone",

    mode=WebRtcMode.SENDONLY,

    media_stream_constraints={

        "video": False,

        "audio": {

            "echoCancellation": True,

            "noiseSuppression": False,

            "autoGainControl": False,
        },
    },

    audio_processor_factory=(
        SpectraAudioProcessor
    ),

    async_processing=True,
)


# ==================================================
# API POLLING
# ==================================================

def ask_api(
    session_id,
):
    """
    Get the latest prediction already calculated
    by the API for this microphone session.

    NO inference happens here.

    GET:
    /predict-mic/{session_id}/latest
    """

    if not session_id:
        return None


    url = (
        f"{API_BASE_URL}"
        f"/predict-mic/"
        f"{session_id}"
        f"/latest"
    )


    try:

        response = requests.get(
            url,
            timeout=2,
        )


        # --------------------------------------------------
        # Session exists but has no prediction yet.
        # --------------------------------------------------

        if response.status_code == 404:

            return None


        response.raise_for_status()


        return response.json()


    except requests.RequestException as error:

        print(
            "Prediction polling error:",
            error,
        )

        return None


# ==================================================
# STATUS + GRAPHICS PLACEHOLDERS
# ==================================================

status_placeholder = st.empty()

frame_placeholder = st.empty()


# ==================================================
# LIVE CONTROLLER LOOP
#
# This is essentially your teacher's pseudocode:
#
# 1. WebRTC gets audio
# 2. AudioProcessor sends it
# 3. We poll latest predictions
# 4. renderer draws them
# 5. repeat at steady FPS
# ==================================================

while True:

    frame_start = (
        time.perf_counter()
    )


    # ==================================================
    # CURRENT WEBSOCKET STATE
    # ==================================================

    ws_state = (
        ws_client.get_state()
    )


    session_id = (
        ws_state["session_id"]
    )


    # ==================================================
    # GET LOCAL RMS FROM AUDIO PROCESSOR
    # ==================================================

    rms = st.session_state[
        "latest_rms"
    ]


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
    # POLL API
    # ==================================================

    now = time.perf_counter()


    if (
        session_id is not None
        and (
            now
            - st.session_state[
                "last_poll_time"
            ]
            >= POLL_INTERVAL
        )
    ):

        result = ask_api(
            session_id
        )


        if result is not None:

            result_timestamp = (
                result.get(
                    "timestamp"
                )
            )


            # Only adapt/store if this is a new
            # prediction from the API.
            if (
                result_timestamp
                != st.session_state[
                    "latest_prediction_timestamp"
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
                    "latest_prediction_timestamp"
                ] = result_timestamp


        st.session_state[
            "last_poll_time"
        ] = now


    # ==================================================
    # CURRENT DISPLAY PREDICTIONS
    # ==================================================

    predictions = (
        st.session_state[
            "latest_predictions"
        ]
    )


    # ==================================================
    # STATUS MESSAGE
    # ==================================================

    if not webrtc_ctx.state.playing:

        status_placeholder.info(
            "Start the microphone to begin listening."
        )


    elif not ws_state["connected"]:

        status_placeholder.warning(
            "Connecting to Spectra…"
        )


    elif session_id is None:

        status_placeholder.info(
            "Starting Spectra session…"
        )


    elif ws_state["status"] == "error":

        status_placeholder.error(
            "Could not communicate with Spectra API."
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
    # RENDER
    # ==================================================

    frame = render_frame(
        predictions,
        rms,
    )


    frame_placeholder.image(
        frame,
        channels="RGB",
        use_container_width=True,
    )


    # ==================================================
    # KEEP FRAME RATE STEADY
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