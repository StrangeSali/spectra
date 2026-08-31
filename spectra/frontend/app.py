
import streamlit as st
import requests
import base64

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
URL = "https://spectra-1087886990522.europe-west1.run.app/predict"

st.set_page_config(
    page_title="Spectra AI",
    page_icon="🎧",
    layout="centered",
)

# -----------------------------------------------------------------------
# STYLE — dark studio background, cyan/violet spectrum accent,
# Space Grotesk for display type, Inter for body text.
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

    #MainMenu, footer, header { visibility: hidden; }

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

# -----------------------------------------------------------------------
# RECORDING CARD
# -----------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="section-label">Record</div>', unsafe_allow_html=True)
    audio_value = st.audio_input("Record a voice message")

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
            st.audio(audio_bytes, format="audio/wav")

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
    with st.spinner("Processing audio signal and generating visualization..."):
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}

        try:
            response = requests.post(URL, files=files)

            if response.status_code == 200:
                data = response.json()
                st.success("Analysis complete!")

                # -----------------------------------------------------
                # RESULTS
                # -----------------------------------------------------
                with st.container(border=True):
                    col_img, col_pred = st.columns([1, 1])

                    with col_img:
                        st.markdown(
                            '<div class="section-label">Visualization</div>',
                            unsafe_allow_html=True,
                        )
                        if "image_base64" in data:
                            image_bytes = base64.b64decode(data["image_base64"])
                            st.image(
                                image_bytes,
                                caption="Audio Spectrum Visualization",
                                use_container_width=True,
                            )
                        elif "image_url" in data:
                            st.image(
                                data["image_url"],
                                caption="Audio Spectrum Visualization",
                                use_container_width=True,
                            )
                        else:
                            st.info("No visualization returned by the backend yet.")

                    with col_pred:
                        st.markdown(
                            '<div class="section-label">Predictions</div>',
                            unsafe_allow_html=True,
                        )
                        predictions = data.get("predictions", [])
                        if predictions:
                            for pred in predictions:
                                class_name = pred.get("class_name", "Unknown").capitalize()
                                confidence = float(pred.get("confidence", 0.0))

                                st.write(f"**{class_name}** — {confidence * 100:.1f}%")
                                st.progress(min(max(confidence, 0.0), 1.0))
                        else:
                            st.info("No predictions returned by the backend yet.")

                # -----------------------------------------------------
                # DEBUG — raw response, to check the backend's actual
                # field names without touching the backend code.
                # -----------------------------------------------------
                with st.expander("Show raw API response (debug)"):
                    st.json(data)

            else:
                st.error(f"Server error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API: {e}")

st.markdown(
    '<p style="color:#9C9AA8; font-size:0.82rem; margin-top:2rem;">'
    "Spectra AI — client interface. Inference and rendering run on a separate backend service."
    "</p>",
    unsafe_allow_html=True,
)
