import streamlit as st
import requests

from spectra.graphics.renderer import render_frame
from spectra.processing.categories import SOUNDS_DICT, DEFAULT_CATEGORY


# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

URL = "https://spectra-1087886990522.europe-west1.run.app/predict"
#URL = "http://0.0.0.0:8000/predict"

st.set_page_config(
    page_title="Spectra AI",
    page_icon="🎧",
    layout="centered",
)


# -----------------------------------------------------------------------
# STYLE — ORIGINAL DESIGN
# -----------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #121218;
        --surface: #1B1B24;
        --border: #2A2A36;
        --text: #F2F0EA;
        --text-dim: #9C9AA8;
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

    #MainMenu, footer, header {
        visibility: hidden;
    }

    /* ---- Hero ---- */

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.1;
        margin-bottom: 0.3rem;
        background: linear-gradient(
            90deg,
            var(--cyan),
            var(--violet)
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: var(--text-dim);
        font-size: 1.02rem;
        max-width: 34rem;
        margin-bottom: 1.6rem;
    }

    /* ---- Waveform accent ---- */

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
        background: linear-gradient(
            180deg,
            var(--cyan),
            var(--violet)
        );
        animation: pulse 1.2s ease-in-out infinite;
    }

    .waveform span:nth-child(1) {
        height: 10px;
        animation-delay: 0s;
    }

    .waveform span:nth-child(2) {
        height: 24px;
        animation-delay: 0.1s;
    }

    .waveform span:nth-child(3) {
        height: 14px;
        animation-delay: 0.2s;
    }

    .waveform span:nth-child(4) {
        height: 30px;
        animation-delay: 0.3s;
    }

    .waveform span:nth-child(5) {
        height: 18px;
        animation-delay: 0.4s;
    }

    .waveform span:nth-child(6) {
        height: 26px;
        animation-delay: 0.5s;
    }

    .waveform span:nth-child(7) {
        height: 12px;
        animation-delay: 0.6s;
    }

    .waveform span:nth-child(8) {
        height: 20px;
        animation-delay: 0.7s;
    }

    @keyframes pulse {
        0%, 100% {
            transform: scaleY(0.6);
            opacity: 0.7;
        }

        50% {
            transform: scaleY(1);
            opacity: 1;
        }
    }

    /* ---- Cards ---- */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border: 1px solid var(--border) !important;
        border-radius: 14px;
        padding: 0.4rem;
    }

    /* ---- Buttons ---- */

    .stButton > button {
        background: linear-gradient(
            90deg,
            var(--cyan),
            var(--violet)
        );
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

    /* ---- Progress bars ---- */

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(
            90deg,
            var(--cyan),
            var(--violet)
        );
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
# ADAPT MODEL PREDICTIONS FOR YOUR MOBILE GRAPHICS
# -----------------------------------------------------------------------

def adapt_predictions(api_predictions):
    """
    Convert detailed model predictions into the broad
    categories used by the Spectra renderer.

    Examples:
        sneezing    -> Human
        mouse_click -> Human
        dog         -> Animal
        rain        -> Nature

    Only the strongest prediction for each broad category
    is kept.
    """

    best_by_category = {}


    for prediction in api_predictions:

        class_name = prediction.get(
            "class_name",
            "",
        )

        confidence = float(
            prediction.get(
                "confidence",
                0.0,
            )
        )


        # ---------------------------------------------------------------
        # If API already returns category, use it.
        # Otherwise use categories.py.
        # ---------------------------------------------------------------

        category = prediction.get(
            "category"
        )


        if not category:

            category = SOUNDS_DICT.get(
                class_name,
                DEFAULT_CATEGORY,
            )


        # ---------------------------------------------------------------
        # Don't show Background as an active sound.
        # ---------------------------------------------------------------

        if category == DEFAULT_CATEGORY:
            continue


        # ---------------------------------------------------------------
        # Keep only strongest prediction per broad category.
        # ---------------------------------------------------------------

        current = best_by_category.get(
            category
        )


        if (
            current is None
            or confidence > current["confidence"]
        ):

            best_by_category[category] = {
                "category": category,
                "display_label": category,
                "confidence": confidence,
            }


    # ---------------------------------------------------------------
    # Dictionary -> list
    # ---------------------------------------------------------------

    adapted = list(
        best_by_category.values()
    )


    # ---------------------------------------------------------------
    # Highest confidence first.
    # This becomes the large hero icon in renderer.py.
    # ---------------------------------------------------------------

    adapted.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True,
    )


    # Mobile renderer supports up to 3 active sounds.
    return adapted[:3]


# -----------------------------------------------------------------------
# HERO — ORIGINAL
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


# -----------------------------------------------------------------------
# RECORDING CARD — ORIGINAL
# -----------------------------------------------------------------------

with st.container(border=True):

    st.markdown(
        '<div class="section-label">Record</div>',
        unsafe_allow_html=True,
    )


    audio_value = st.audio_input(
        "Record a voice message"
    )


    audio_bytes = None


    if audio_value is not None:

        audio_bytes = audio_value.getvalue()


        if len(audio_bytes) == 0:

            st.error(
                "I'm sorry, but I don't have access to your microphone. "
                "Please check your browser permissions and try again."
            )

            audio_bytes = None


        else:

            st.audio(
                audio_bytes,
                format="audio/wav",
            )


    analyze_clicked = st.button(
        "Analyze Audio & Visualize",
        type="primary",
        disabled=audio_bytes is None,
        use_container_width=True,
    )


# -----------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------

if analyze_clicked and audio_bytes:

    with st.spinner(
        "Processing audio signal and generating visualization..."
    ):

        files = {
            "file": (
                "audio.wav",
                audio_bytes,
                "audio/wav",
            )
        }


        try:

            # -----------------------------------------------------------
            # COLLEAGUE'S EXISTING DEPLOYED API
            # -----------------------------------------------------------

            response = requests.post(
                URL,
                files=files,
                timeout=60,
            )


            if response.status_code == 200:

                data = response.json()

                st.success(
                    "Analysis complete!"
                )


                # -------------------------------------------------------
                # RAW API PREDICTIONS
                # -------------------------------------------------------

                raw_predictions = data.get(
                    "predictions",
                    [],
                )


                # -------------------------------------------------------
                # PREPARE YOUR GRAPHICS PREDICTIONS
                # -------------------------------------------------------

                graphics_predictions = (
                    adapt_predictions(
                        raw_predictions
                    )
                )


                # -------------------------------------------------------
                # CREATE YOUR MOBILE SPECTRA FRAME
                #
                # Current recorded-audio endpoint doesn't provide RMS,
                # so use a small static value for now.
                # -------------------------------------------------------

                rms = 0.05


                frame = render_frame(
                    graphics_predictions,
                    rms,
                )


                # =======================================================
                # RESULTS
                #
                # SAME TWO-COLUMN IDEA AS COLLEAGUE'S ORIGINAL:
                #
                # LEFT  -> YOUR MOBILE APP
                # RIGHT -> PREDICTIONS
                # =======================================================

                with st.container(
                    border=True
                ):

                    col_img, col_pred = st.columns(
                        [1, 1]
                    )


                    # ===================================================
                    # LEFT — YOUR MOBILE SPECTRA UI
                    # ===================================================

                    with col_img:

                        st.markdown(
                            '<div class="section-label">'
                            'Visualization'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        st.image(
                            frame,
                            channels="RGB",
                            use_container_width=True,
                        )


                    # ===================================================
                    # RIGHT — BROAD PREDICTIONS
                    # ===================================================

                    with col_pred:

                        st.markdown(
                            '<div class="section-label">'
                            'Predictions'
                            '</div>',
                            unsafe_allow_html=True,
                        )


                        if graphics_predictions:

                            for pred in graphics_predictions:

                                category = pred.get(
                                    "category",
                                    "Unknown",
                                )

                                confidence = float(
                                    pred.get(
                                        "confidence",
                                        0.0,
                                    )
                                )


                                st.write(
                                    f"**{category}** — "
                                    f"{confidence * 100:.1f}%"
                                )


                                st.progress(
                                    min(
                                        max(
                                            confidence,
                                            0.0,
                                        ),
                                        1.0,
                                    )
                                )


                        else:

                            st.info(
                                "No predictions returned "
                                "by the backend yet."
                            )


                # -------------------------------------------------------
                # DEBUG — ORIGINAL IDEA PRESERVED
                #
                # This is especially useful now:
                #
                # USER sees:
                #     Human 52%
                #
                # Team can inspect:
                #     sneezing 52%
                # -------------------------------------------------------

                with st.expander(
                    "Show raw API response (debug)"
                ):

                    st.json(
                        data
                    )


            else:

                st.error(
                    f"Server error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )


        except requests.exceptions.RequestException as e:

            st.error(
                f"Failed to connect to the API: {e}"
            )


# -----------------------------------------------------------------------
# FOOTER — ORIGINAL
# -----------------------------------------------------------------------

st.markdown(
    '<p style="color:#9C9AA8; font-size:0.82rem; margin-top:2rem;">'
    "Spectra AI — client interface. "
    "Inference and rendering run on a separate backend service."
    "</p>",
    unsafe_allow_html=True,
)
