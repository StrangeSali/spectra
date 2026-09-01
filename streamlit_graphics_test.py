import time

import streamlit as st

from spectra.graphics.renderer import (
    render_frame,
    reset_animation_state,
)


# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(
    page_title="Spectra AI",
    layout="wide",
)

st.title("Spectra AI - Animated Graphics Test")


# ==================================================
# TEST DATA
#
# Still fake for now.
# Later these values will come from GET /predictions.
# ==================================================

predictions = [
    {
        "category": "Alert",
        "display_label": "Siren",
        "confidence": 0.60,
    },
    {
        "category": "Clapping",
        "display_label": "Clapping",
        "confidence": 0.90,
    },
    {
        "category": "Vehicle",
        "display_label": "Car",
        "confidence": 0.75,
    },
]

rms = 0.08


# ==================================================
# CONTROLS
# ==================================================

col1, col2 = st.columns(2)

with col1:
    start_animation = st.button(
        "Start animation"
    )

with col2:
    stop_animation = st.button(
        "Stop animation"
    )


# ==================================================
# SESSION STATE
#
# Streamlit reruns the script when a button is
# clicked, so we keep the running status here.
# ==================================================

if "animation_running" not in st.session_state:
    st.session_state.animation_running = False


if start_animation:
    st.session_state.animation_running = True


if stop_animation:
    st.session_state.animation_running = False


# ==================================================
# FRAME PLACEHOLDER
#
# This creates ONE location in the webpage.
# Every new frame replaces the previous frame.
# ==================================================

left_column, phone_column, right_column = st.columns(
    [0.5, 2, 0.5]
)

with phone_column:
    frame_placeholder = st.empty()


# ==================================================
# ANIMATION
# ==================================================

TARGET_FPS = 15

FRAME_DURATION = 1 / TARGET_FPS


if st.session_state.animation_running:

    while st.session_state.animation_running:

        frame_start = time.perf_counter()


        # ------------------------------------------
        # Ask Pygame to render ONE new frame
        # ------------------------------------------

        frame = render_frame(
            predictions,
            rms,
        )


        # ------------------------------------------
        # Replace previous image in Streamlit
        # ------------------------------------------

        frame_placeholder.image(
            frame,
            channels="RGB",
            width=430,
        )


        # ------------------------------------------
        # FPS control
        # ------------------------------------------

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


else:

    # Show one frame even when animation
    # isn't running.

    frame = render_frame(
        predictions,
        rms,
    )

    frame_placeholder.image(
        frame,
        channels="RGB",
        width=430,
    )
