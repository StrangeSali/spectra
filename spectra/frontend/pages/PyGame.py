import os
import time
import streamlit.components.v1 as components
import base64
import av
import numpy as np
import requests
import streamlit as st
from live_audio import AudioDownsampler, RollingAudioBuffer,LatestChunkUploader, rms_of_pcm16, pcm16_to_wav_bytes,RecentPredictionsPoller
from streamlit_webrtc import (
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from graphics.renderer import render_frame, reset_animation_state, render_frame_jpeg

from support.classes import DEFAULT_CATEGORY, SOUNDS_DICT

JPEG_QUALITY = 75


def frame_data_uri(predictions, rms):
    jpeg = render_frame_jpeg(
        predictions,
        rms,
        quality=JPEG_QUALITY,
    )

    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")

# ==================================================
# CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2
TARGET_FPS = 12
FRAME_DURATION = 1 / TARGET_FPS
HOP_SECONDS=1
# How often we ask the API for its latest prediction.
POLL_INTERVAL = 0.25
UPLOAD_TIMEOUT=10
MAX_PREDICTIONS=3
STALE_AFTER=6.0
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



def show_prediction_list(slots, predictions):

    message, rows = slots

    if predictions:
        message.empty()
    else:
        message.info("No sound detected yet.")

    for index, (label, bar) in enumerate(rows):

        if index < len(predictions):

            pred = predictions[index]

            category = pred.get("category", "Unknown")
            confidence = float(pred.get("confidence", 0.0))

            label.write(f"**{category}** — {confidence * 100:.1f}%")
            bar.progress(min(max(confidence, 0.0), 1.0))

        else:

            label.empty()
            bar.empty()

def make_prediction_slots():

    message = st.empty()
    rows = [(st.empty(), st.empty()) for _ in range(MAX_PREDICTIONS)]

    return message, rows

class SpectraAudioProcessor(AudioProcessorBase):

    def __init__(self):
        self.latest_rms = 0.0
        self.last_error = None

        self._downsampler = AudioDownsampler(SAMPLE_RATE)

        self._buffer = RollingAudioBuffer(
            window_seconds=WINDOW_SECONDS,
            hop_seconds=HOP_SECONDS,
            sample_rate=SAMPLE_RATE,
        )

        self.uploader = LatestChunkUploader(CHUNK_URL, timeout=UPLOAD_TIMEOUT)
        self.uploader.start()

    async def recv_queued(self, frames):
        # Any exception here would kill the WebRTC track, so keep it contained.
        try:
            pcm = self._downsampler.process(frames)

            if pcm.size:
                self.latest_rms = rms_of_pcm16(pcm)

                window = self._buffer.push(pcm)

                if window is not None:
                    self.uploader.submit(pcm16_to_wav_bytes(window, SAMPLE_RATE))

        except Exception as error:  # surfaced in the diagnostics panel
            self.last_error = f"{type(error).__name__}: {error}"

        # SENDONLY: nothing is played back, frames are simply discarded.
        return frames

    def on_ended(self):
        self.uploader.stop()

# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="Spectra AI",
    page_icon="🎧",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #121218;
        --surface: #1B1B24;
        --border: #2A2A36;
        --text: #F2F0EA;
        --text-dim: #D8D6D0;
        --cyan: #00E5C7;
        --violet: #8C6DFF;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
    }

    .stApp {
        background: var(--bg);
    }



#MainMenu, footer { visibility: hidden; }

header {
    background: transparent !important;
}

    /* ---- Hero ---- */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, var(--cyan), var(--violet));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-dim);
        font-size: 1.02rem;
        max-width: 34rem;
        margin-bottom: 1.6rem;
    }

    /* ---- Waveform accent (single motion moment) ---- */
    .waveform {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 34px;
        margin-bottom: 1.8rem;
    }
    .waveform span {
        display: block;
        width: 4px;
        border-radius: 2px;
        background: linear-gradient(180deg, var(--cyan), var(--violet));
        animation: pulse 1.2s ease-in-out infinite;
    }
    .waveform span:nth-child(1) { height: 10px; animation-delay: 0s; }
    .waveform span:nth-child(2) { height: 24px; animation-delay: 0.1s; }
    .waveform span:nth-child(3) { height: 14px; animation-delay: 0.2s; }
    .waveform span:nth-child(4) { height: 30px; animation-delay: 0.3s; }
    .waveform span:nth-child(5) { height: 18px; animation-delay: 0.4s; }
    .waveform span:nth-child(6) { height: 26px; animation-delay: 0.5s; }
    .waveform span:nth-child(7) { height: 12px; animation-delay: 0.6s; }
    .waveform span:nth-child(8) { height: 20px; animation-delay: 0.7s; }

    @keyframes pulse {
        0%, 100% { transform: scaleY(0.6); opacity: 0.7; }
        50% { transform: scaleY(1); opacity: 1; }
    }

    /* ---- Cards (st.container(border=True)) ---- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 14px;
        padding: 0.4rem;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(90deg, var(--cyan), var(--violet));
        color: #0D0D12;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.88;
        color: #0D0D12;
    }

    /* ---- Progress bars (confidence) ---- */
    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, var(--cyan), var(--violet));
    }

    /* ---- Section labels ---- */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.05rem;
        margin: 1.4rem 0 0.6rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------
st.markdown(
    """
    <div class="waveform">
        <span></span><span></span><span></span><span></span>
        <span></span><span></span><span></span><span></span>
    </div>
    <div class="hero-title">Spectra AI</div>
    <div class="hero-subtitle">
        Record a sound and see it classified and visualized in real time —
        built for hearing accessibility.
    </div>
    """,
    unsafe_allow_html=True,
)

def inject_webrtc_styles():
    components.html(
        """
        <script>
        function styleWebrtcIframe() {
            try {
                const parentDoc = window.parent.document;
                const iframe = parentDoc.querySelector(
                    'iframe[title="streamlit_webrtc.webrtc_streamer"]'
                );
                if (!iframe || !iframe.contentDocument) return;

                const doc = iframe.contentDocument;
                let style = doc.getElementById('spectra-injected-style');
                if (!style) {
                    style = doc.createElement('style');
                    style.id = 'spectra-injected-style';
                    doc.head.appendChild(style);
                }

                style.textContent = `
                    html, body, #root {
                        background: #1B1B24 !important;
                        color: #F2F0EA !important;
                        margin: 0 !important;
                    }
                    select {
                        display: none !important;
                    }
                    label {
                        display: none !important;
                    }
                    button {
                        background: linear-gradient(90deg, #00E5C7, #8C6DFF) !important;
                        color: #0D0D12 !important;
                        font-family: 'Space Grotesk', sans-serif !important;
                        font-weight: 700 !important;
                        border: none !important;
                        border-radius: 8px !important;
                        padding: 0.7rem 2rem !important;
                        text-transform: uppercase !important;
                        letter-spacing: 0.05em !important;
                        cursor: pointer !important;
                    }
                    video {
                        display: none !important;
                    }
                `;
            } catch (e) {
                // cross-origin or iframe not ready yet — will retry
            }
        }

        // Iframe gets recreated on Streamlit reruns, so keep reapplying.
        setInterval(styleWebrtcIframe, 300);
        </script>
        """,
        height=0,
        width=0,
    )

# # ==================================================
# # MOBILE CSS
# # ==================================================

# st.markdown(
#     """
#     <style>

#     .block-container {
#         max-width: 500px;
#         padding-top: 1rem;
#         padding-left: 0.5rem;
#         padding-right: 0.5rem;
#     }

#     h1 {
#         text-align: center;
#     }

#     video {
#         display: none !important;
#     }

#     audio {
#         width: 100% !important;
#     }

#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# st.title("Spectra AI")

# st.caption("Real-Time Environmental Audio Visualizer")

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
# WEBRTC MICROPHONE
# ==================================================

with st.container(border=True):

    st.markdown(
        '<div class="section-label">Live microphone</div>',
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
    key="spectra_streamer",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=SpectraAudioProcessor,
    media_stream_constraints={"video": False, "audio": True},
)
    inject_webrtc_styles()

    status_placeholder = st.empty()


# ==================================================
# ASK API FOR LATEST PREDICTION
# ==================================================

def get_latest_predictions(n):
    try:

        response = requests.get(
            RECENT_URL,
            timeout=n,
        )
        response.raise_for_status()
        data = response.json()
        return data


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


with st.container(border=True):

    col_img, col_pred = st.columns([1, 1])

    with col_img:

        st.markdown(
            '<div class="section-label">Visualization</div>',
            unsafe_allow_html=True,
        )

        frame_placeholder = st.empty()

    with col_pred:

        st.markdown(
            '<div class="section-label">Predictions</div>',
            unsafe_allow_html=True,
        )

        prediction_slots = make_prediction_slots()

if not webrtc_ctx.state.playing:

    #level, message = choose_status(False, None, None, [])
    #STATUS_WRITERS[level](status_placeholder, message)

    frame_placeholder.image(
        frame_data_uri([], 0.0),
        width="stretch",
    )

    show_prediction_list(prediction_slots, [])

else:

    reset_animation_state()

    poller = RecentPredictionsPoller(
        fetch=get_latest_predictions,
        interval=POLL_INTERVAL,
        n=1,
    )
    poller.start()

    # Server timestamp of OUR first classified chunk. Older /recent entries
    # (from earlier runs or other clients) are not shown as live results.
    first_upload_timestamp = None

    shown_predictions = None

    try:

        while webrtc_ctx.state.playing:

            frame_start = time.perf_counter()

            processor = webrtc_ctx.audio_processor

            rms = 0.0
            upload_stats = None
            processor_error = None

            if processor is not None:
                rms = float(processor.latest_rms)
                upload_stats = processor.uploader.snapshot()
                processor_error = processor.last_error

                if first_upload_timestamp is None and upload_stats["uploads"] > 0:
                    first_upload_timestamp = upload_stats["last_timestamp"]

            recent = poller.latest()


            is_live = (
                recent["timestamp"] is not None
                and first_upload_timestamp is not None
                and recent["timestamp"] >= first_upload_timestamp
            )

            is_fresh = (
                recent["updated_at"] is not None
                and time.monotonic() - recent["updated_at"] <= STALE_AFTER
            )

            predictions = (
                adapt_predictions(recent["predictions"]) if is_live and is_fresh else []
            )


            frame_placeholder.image(
                frame_data_uri(predictions, rms),
                width="stretch",
            )

            if predictions != shown_predictions:
                show_prediction_list(prediction_slots, predictions)
                shown_predictions = predictions

            sleep_time = FRAME_DURATION - (time.perf_counter() - frame_start)

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:

        poller.stop()
