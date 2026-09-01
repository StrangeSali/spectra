import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import streamlit as st
import requests
import pandas as pd
import os
from spectra.frontend.support.classes import DEFAULT_CATEGORY, SOUNDS_DICT

st.title("Clean MVP")


URL = "https://spectra-1087886990522.europe-west1.run.app/predict"

with st.container(border=True):

    st.markdown(
        '<div class="section-label">Audio Recorder</div>',
        unsafe_allow_html=True,
    )

    audio_value = st.audio_input(
    "Click on the microphone icon to start recording",
    key="my_audio_input")

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
col1, col2  = st.columns(2)
with col2:
    if st.button("Record again", type="secondary", use_container_width=True):
        if "my_audio_input" in st.session_state:
            del st.session_state["my_audio_input"]
        st.rerun()

with col1:
    analyze_clicked = st.button(
        "Analyze Audio & Visualize",
        type="primary",
        disabled=audio_bytes is None,
        use_container_width=True,
    )

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

        response = requests.post(
                    URL,
                    files=files,
                    timeout=60,
                )

        predictions = response.json()["predictions"]
        df = pd.DataFrame(predictions)
        #st.dataframe(df)

        classes = []
        for group in df.class_name:
            if SOUNDS_DICT[group] not in classes:
                classes.append(SOUNDS_DICT[group])

        best_class = SOUNDS_DICT[df.sort_values(by="confidence", ascending=False).iloc[0].class_name]
        confidences = df.sort_values(by="confidence", ascending=False).confidence

        #All 3 classes belong to the same group:
        if len(classes) == 1 or confidences[1] < 0.20:

            st.subheader(best_class.title())
            st.progress(confidences[0], text="Confidence %")

            best_class_image_path = os.path.join("spectra/frontend/images", f"{best_class}.png")
            try:
                st.image(best_class_image_path)
            except:
                st.text("Background")

        #Classes from different groups - 2 images
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(best_class.title())
                st.progress(confidences[0], text="Confidence %")

                best_class_image_path = os.path.join("spectra/frontend/images", f"{best_class}.png")
                try:
                    st.image(best_class_image_path)
                except:
                    st.text("Background")

            with col2:
                st.subheader(classes[1].title())
                st.progress(confidences[1], text="Confidence %")

                class_image_path = os.path.join("spectra/frontend/images", f"{classes[1]}.png")
                try:
                    st.image(class_image_path)
                except:
                    st.text("Background")
