# ==================================================
# IMPORTS
# ==================================================

import json
import os
import threading
import time

import av
import numpy as np
import streamlit as st
import websocket

from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from spectra.graphics.renderer import (
    render_frame
)


# ==================================================
# CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000

TARGET_FPS = 12

FRAME_DURATION = (
    1 / TARGET_FPS
)


# Local development:
#
# ws://127.0.0.1:8001/predict-mic
#
# Later, when API is deployed, set:
#
# SPECTRA_WS_URL=wss://your-api.../predict-mic

SPECTRA_WS_URL = os.getenv(
    "SPECTRA_WS_URL",
    "ws://127.0.0.1:8001/predict-mic",
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


st.title(
    "Spectra AI"
)


st.caption(
    "Real-time sound analysis"
)


# ==================================================
# API -> GRAPHICS ADAPTER
# ==================================================

def adapt_predictions(
    api_predictions,
):

    if not api_predictions:

        return []


    best_by_category = {}


    for prediction in api_predictions:

        category = prediction.get(
            "category",
            "Background",
        )


        confidence = float(
            prediction.get(
                "confidence",
                0.0,
            )
        )


        if category == "Background":
            continue


        # ------------------------------------------
        # KEEP SPECIFIC HYBRID LABEL
        # ------------------------------------------

        display_label = prediction.get(
            "display_label",
            prediction.get(
                "class_name",
                category,
            ),
        )


        current = (
            best_by_category.get(
                category
            )
        )


        if (
            current is None
            or confidence
            > current["confidence"]
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

                "source":
                    prediction.get(
                        "source"
                    ),
            }


    adapted = list(
        best_by_category.values()
    )


    adapted.sort(
        key=lambda prediction:
            prediction[
                "confidence"
            ],
        reverse=True,
    )


    # ----------------------------------------------
    # PRIMARY SOUND ALWAYS ALLOWED
    # ----------------------------------------------

    if not adapted:

        return []


    displayed = [
        adapted[0]
    ]


    # ----------------------------------------------
    # SECONDARY SOUNDS MUST BE STRONG
    # ----------------------------------------------

    for prediction in adapted[1:]:

        if (
            prediction[
                "confidence"
            ]
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
# ==================================================

class SpectraWebSocketClient:

    def __init__(
        self,
        url,
    ):

        self.url = url

        self.ws = None

        self.thread = None

        self.latest_predictions = []

        self.latest_rms = 0.0

        self.status = (
            "connecting"
        )

        self.session_id = None

        self.connected = False

        self.lock = (
            threading.Lock()
        )


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

            self.status = (
                "connected"
            )


        print(
            "Connected to Spectra API"
        )


    def _on_message(
        self,
        ws,
        message,
    ):

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


        # ------------------------------------------
        # CONNECTED
        # ------------------------------------------

        if status == "connected":

            with self.lock:

                self.session_id = (
                    data.get(
                        "session_id"
                    )
                )

                self.status = (
                    "connected"
                )


            print(
                "Spectra session:",
                self.session_id,
            )


        # ------------------------------------------
        # CALIBRATING
        # ------------------------------------------

        elif status == "calibrating":

            with self.lock:

                self.status = (
                    "calibrating"
                )

                self.latest_predictions = []

                self.latest_rms = float(
                    data.get(
                        "rms",
                        0.0,
                    )
                )


        # ------------------------------------------
        # PREDICTIONS
        # ------------------------------------------

        elif status == "processing":

            predictions = data.get(
                "predictions",
                [],
            )


            with self.lock:

                self.status = (
                    "listening"
                    if not predictions
                    else "sound"
                )

                self.latest_predictions = (
                    predictions
                )

                self.latest_rms = float(
                    data.get(
                        "rms",
                        self.latest_rms,
                    )
                )


        # ------------------------------------------
        # ERROR
        # ------------------------------------------

        elif status == "error":

            with self.lock:

                self.status = "error"


            print(
                "Spectra API error:",
                data.get(
                    "detail"
                ),
            )


    def _on_close(
        self,
        ws,
        close_status_code,
        close_msg,
    ):

        with self.lock:

            self.connected = False

            self.status = (
                "disconnected"
            )


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
                "predictions":
                    list(
                        self.latest_predictions
                    ),

                "rms":
                    float(
                        self.latest_rms
                    ),

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
# ONE CLIENT PER STREAMLIT SESSION
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
# AUDIO PROCESSOR
# ==================================================

class SpectraAudioProcessor(
    AudioProcessorBase
):

    def __init__(self):

        self.latest_rms = 0.0


        # ------------------------------------------
        # IMPORTANT
        #
        # Browser microphones commonly produce
        # 48 kHz audio.
        #
        # Spectra/YAMNet expects 16 kHz.
        #
        # PyAV handles the conversion here BEFORE
        # audio is sent to the API.
        # ------------------------------------------

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


            # --------------------------------------
            # SAFETY NORMALIZATION
            # --------------------------------------

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


            # --------------------------------------
            # LOCAL RMS
            # --------------------------------------

            self.latest_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            audio
                        )
                    )
                )
            )


            # --------------------------------------
            # SEND 16 kHz FLOAT32 AUDIO
            # --------------------------------------

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
# STATUS
# ==================================================

status_placeholder = st.empty()


# ==================================================
# GRAPHICS CONTAINER
# ==================================================

frame_placeholder = st.empty()


# ==================================================
# LIVE GRAPHICS
# ==================================================

while True:

    frame_start = (
        time.perf_counter()
    )


    state = ws_client.get_state()


    raw_predictions = (
        state["predictions"]
    )


    predictions = (
        adapt_predictions(
            raw_predictions
        )
    )


    # ----------------------------------------------
    # RMS
    #
    # Prefer API RMS because it corresponds to the
    # exact 16 kHz waveform used for inference.
    # ----------------------------------------------

    rms = float(
        state["rms"]
    )


    # ----------------------------------------------
    # STATUS MESSAGE
    # ----------------------------------------------

    if state["status"] == "calibrating":

        status_placeholder.info(
            "Calibrating background sound… "
            "keep reasonably quiet for a few seconds."
        )


    elif state["status"] == "error":

        status_placeholder.error(
            "Could not communicate with Spectra API."
        )


    elif not state["connected"]:

        status_placeholder.warning(
            "Connecting to Spectra…"
        )


    elif predictions:

        primary = (
            predictions[0]
        )

        status_placeholder.success(
            f"Detected: "
            f"{primary['display_label']}"
        )


    else:

        status_placeholder.caption(
            "Listening…"
        )


    # ----------------------------------------------
    # RENDER
    # ----------------------------------------------

    frame = render_frame(
        predictions,
        rms,
    )


    frame_placeholder.image(
        frame,
        channels="RGB",
        use_container_width=True,
    )


    # ----------------------------------------------
    # FRAME RATE
    # ----------------------------------------------

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
