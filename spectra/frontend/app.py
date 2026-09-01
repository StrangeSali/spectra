import streamlit as st
import requests
import pandas as pd
import os
st.title("Clean MVP")


URL = "https://spectra-1087886990522.europe-west1.run.app/predict"

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
        st.dataframe(df)
        best_class = df.sort_values(by="confidence", ascending=False).iloc[0].class_name
        st.write(best_class)
        best_class_image_path = os.path.join("spectra/frontend/images", f"{best_class}.jpg")
        try:
            st.image(best_class_image_path)
        except:
            st.text("We dont have an image yet for the class")
