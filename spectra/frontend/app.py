import streamlit as  st
import requests
url = "https://spectra-1087886990522.europe-west1.run.app/predict"

st.title("Audio predtion")

audio_value = st.audio_input("Record a voice message")

if audio_value:

    #requests.post(audio_value)

    #prediction =
