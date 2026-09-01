import os
import requests
import base64
import streamlit as st
from spectra.processing.categories import SOUNDS_DICT, DEFAULT_CATEGORY

# -----------------------------------------------------------------------
# CONFIG & PATHS
# -----------------------------------------------------------------------
URL = "https://spectra-1087886990522.europe-west1.run.app/predict"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

st.set_page_config(
    page_title="Spectra AI",
    page_icon="🎧",
    layout="centered",
)

# -----------------------------------------------------------------------
# CLAYMORPHISM THEME — Papel kraft / arcilla cálida
# -----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800&family=Inter:wght@400;500&display=swap');

/* ── Fondo global: papel kraft con grano ─────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #C8A876 !important;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E");
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Ancho máximo centrado ───────────────────────────────────────── */
.block-container {
    max-width: 680px !important;
    padding: 2rem 1.5rem !important;
}

/* ── Logo / waveform icon ────────────────────────────────────────── */
.clay-icon {
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 28px;
    margin-bottom: 0.5rem;
}
.clay-icon span {
    display: block;
    width: 4px;
    border-radius: 3px;
    background: #4A3520;
    opacity: 0.55;
    animation: claywave 1.4s ease-in-out infinite;
}
.clay-icon span:nth-child(1) { height: 8px;  animation-delay: 0s; }
.clay-icon span:nth-child(2) { height: 20px; animation-delay: 0.12s; }
.clay-icon span:nth-child(3) { height: 12px; animation-delay: 0.24s; }
.clay-icon span:nth-child(4) { height: 24px; animation-delay: 0.36s; }
.clay-icon span:nth-child(5) { height: 16px; animation-delay: 0.48s; }
.clay-icon span:nth-child(6) { height: 22px; animation-delay: 0.60s; }
.clay-icon span:nth-child(7) { height: 10px; animation-delay: 0.72s; }
@keyframes claywave {
    0%, 100% { transform: scaleY(0.55); opacity: 0.4; }
    50%       { transform: scaleY(1);   opacity: 0.7; }
}

/* ── Título y subtítulo ──────────────────────────────────────────── */
.clay-title {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 2.6rem;
    color: #2C1A0E;
    letter-spacing: -0.5px;
    margin: 0 0 0.3rem;
}
.clay-sub {
    font-size: 1rem;
    color: #5A3E28;
    max-width: 30rem;
    line-height: 1.5;
    margin-bottom: 1.6rem;
}

/* ── Panel de arcilla (el "bloque" de plastilina) ────────────────── */
.clay-panel {
    background: #8B5E3C;
    border-radius: 28px 32px 26px 30px;
    padding: 1.4rem 1.4rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow:
        inset 0 3px 6px rgba(255,255,255,0.18),
        inset 0 -4px 8px rgba(0,0,0,0.30),
        0 8px 20px rgba(80,40,10,0.30),
        0 2px 4px rgba(80,40,10,0.15);
    position: relative;
}
.clay-panel-dark {
    background: #5C3A1E;
    border-radius: 26px 30px 28px 24px;
    padding: 1.4rem;
    box-shadow:
        inset 0 3px 6px rgba(255,255,255,0.10),
        inset 0 -4px 8px rgba(0,0,0,0.40),
        0 8px 20px rgba(40,20,5,0.35);
}
.clay-label {
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #F5E6D0;
    margin-bottom: 0.8rem;
}
.clay-hint {
    font-size: 0.88rem;
    color: #D4B89A;
    margin-bottom: 0.6rem;
}

/* ── Botón GRANDE de micrófono ───────────────────────────────────── */
.clay-mic-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    width: 100%;
    background: #B85C2A;
    border: none;
    border-radius: 18px 22px 18px 20px;
    padding: 1rem 1.4rem;
    cursor: pointer;
    box-shadow:
        inset 0 3px 5px rgba(255,255,255,0.20),
        inset 0 -4px 8px rgba(0,0,0,0.35),
        0 6px 14px rgba(100,30,10,0.35);
    transition: transform 0.1s ease, box-shadow 0.1s ease;
    margin-bottom: 0.8rem;
}
.clay-mic-btn:hover { transform: translateY(-2px); }
.clay-mic-btn:active {
    transform: translateY(2px);
    box-shadow:
        inset 0 2px 4px rgba(255,255,255,0.15),
        inset 0 -2px 4px rgba(0,0,0,0.30),
        0 2px 6px rgba(100,30,10,0.25);
}
.clay-mic-icon { font-size: 1.6rem; }
.clay-mic-text {
    font-family: 'Nunito', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: #FFF0E0;
}

/* ── Botón "Analyze" (cta verde) ──────────────────────────────────── */
.stButton > button[data-testid="baseButton-primary"] {
    background: #3D7A5E !important;
    color: #E8F5EE !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 16px 20px 16px 18px !important;
    padding: 0.75rem 1.4rem !important;
    box-shadow:
        inset 0 3px 5px rgba(255,255,255,0.20),
        inset 0 -4px 8px rgba(0,0,0,0.30),
        0 6px 14px rgba(20,60,40,0.30) !important;
    transition: transform 0.1s ease !important;
    width: 100% !important;
}
.stButton > button[data-testid="baseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    color: #E8F5EE !important;
}
.stButton > button[data-testid="baseButton-secondary"] {
    background: #8B5E3C !important;
    color: #F5E6D0 !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow:
        inset 0 2px 4px rgba(255,255,255,0.15),
        inset 0 -3px 6px rgba(0,0,0,0.25),
        0 4px 10px rgba(80,40,10,0.25) !important;
}

/* ── Quitar borde de los st.container ────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* ── Barras de progreso ──────────────────────────────────────────── */
[data-testid="stProgress"] > div {
    background: rgba(0,0,0,0.20) !important;
    border-radius: 12px !important;
    height: 14px !important;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #A83030, #C04040) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 2px 3px rgba(255,255,255,0.25), inset 0 -2px 3px rgba(0,0,0,0.25) !important;
}

/* ── Texto de predicciones ───────────────────────────────────────── */
.pred-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    color: #F0DFC8;
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 0.2rem;
}

/* ── Success / error ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 16px !important;
}

/* ── Debug expander ──────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(0,0,0,0.15) !important;
    border-radius: 12px !important;
    border: none !important;
    color: #D4B89A !important;
}

/* ── Audio widget ────────────────────────────────────────────────── */
[data-testid="stAudio"] audio {
    border-radius: 12px;
    width: 100%;
}

/* ── Waveform grabado (st.audio_input) ───────────────────────────── */
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    border: none !important;
}

/* ── Footer ──────────────────────────────────────────────────────── */
.clay-footer {
    font-size: 0.80rem;
    color: #7A5535;
    margin-top: 2rem;
    text-align: left;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------
st.markdown("""
<div class="clay-icon">
    <span></span><span></span><span></span>
    <span></span><span></span><span></span><span></span>
</div>
<div class="clay-title">Spectra AI</div>
<div class="clay-sub">
    Record a sound and see it classified and visualized in real time —
    built for hearing accessibility.
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# RECORDING CARD
# -----------------------------------------------------------------------
st.markdown('<div class="clay-panel">', unsafe_allow_html=True)
st.markdown('<div class="clay-label">Record</div>', unsafe_allow_html=True)

audio_value = st.audio_input("Record a new sound", label_visibility="collapsed")

audio_bytes = None
if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    if len(audio_bytes) == 0:
        st.error("Microphone access failed. Check browser permissions.")
        audio_bytes = None
    else:
        st.audio(audio_bytes, format="audio/wav")

analyze_clicked = st.button(
    "🔍  Analyze Audio & Visualize",
    type="primary",
    disabled=audio_bytes is None,
    use_container_width=True,
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------
# ANALYSIS & RESULTS
# -----------------------------------------------------------------------
if analyze_clicked and audio_bytes:
    with st.spinner("Processing audio... please wait."):
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        try:
            response = requests.post(URL, files=files)

            if response.status_code == 200:
                data = response.json()

                st.markdown(
                    '<div style="background:#3D7A5E;border-radius:14px;'
                    'padding:0.75rem 1.2rem;color:#E8F5EE;font-weight:600;'
                    'font-family:Nunito,sans-serif;margin-bottom:0.8rem;">'
                    'Analysis complete!</div>',
                    unsafe_allow_html=True,
                )

                # Panel oscuro de resultados
                st.markdown('<div class="clay-panel-dark">', unsafe_allow_html=True)

                predictions = data.get("predictions", [])
                top_class = "unknown"
                if predictions:
                    top_class = predictions[0].get("class_name", "unknown").lower()

                detected_category = SOUNDS_DICT.get(top_class, DEFAULT_CATEGORY).lower()
                img_file = f"{detected_category}.png"
                img_path = os.path.join(ASSETS_DIR, img_file)

                col_img, col_pred = st.columns([1, 1])

                with col_img:
                    st.markdown('<div class="clay-label">Visualization</div>', unsafe_allow_html=True)
                    if os.path.exists(img_path):
                        st.image(img_path, caption=f"Category: {detected_category.capitalize()}",
                                 use_container_width=True)
                    elif "image_base64" in data:
                        image_bytes = base64.b64decode(data["image_base64"])
                        st.image(image_bytes, caption="Audio Spectrum", use_container_width=True)
                    else:
                        st.info("No visualization asset available.")

                with col_pred:
                    st.markdown('<div class="clay-label">Predictions</div>', unsafe_allow_html=True)
                    if predictions:
                        for pred in predictions:
                            class_name = pred.get("class_name", "Unknown").capitalize()
                            confidence = float(pred.get("confidence", 0.0))
                            pct = f"{confidence * 100:.1f}%"
                            st.markdown(
                                f'<div class="pred-row">'
                                f'<span>{class_name}</span>'
                                f'<span style="color:#D4B89A;font-size:0.85rem">{pct}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            st.progress(min(max(confidence, 0.0), 1.0))
                    else:
                        st.info("No predictions returned.")

                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("› Show raw API response (debug)"):
                    st.json(data)

            else:
                st.error(f"Server error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the API: {e}")

# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.markdown(
    '<p class="clay-footer">Spectra AI — client interface. '
    'Inference and rendering run on a separate backend service.</p>',
    unsafe_allow_html=True,
)
