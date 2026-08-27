import json
import os
import pyaudio

import tensorflow_hub as hub
from tf_keras.models import load_model
from google.cloud import storage

from processing.audio import capture_audio_chunk, preprocess_audio
from processing.yamnet_utils import extract_features
from processing.classifier import predict_sound


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")


# --------------------------------------------------
# Load YAMNet
# --------------------------------------------------

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)


# --------------------------------------------------
# Load classifier from GCS
# --------------------------------------------------

storage_client = storage.Client()

bucket = storage_client.bucket(
    MODEL_BUCKET_NAME
)

blob = bucket.blob(
    MODEL_NAME
)

local_model_path = "/tmp/dense_50.keras"

blob.download_to_filename(
    local_model_path
)

classifier_model = load_model(
    local_model_path
)


# --------------------------------------------------
# Process audio
# --------------------------------------------------

def process_audio(mic, chunk_size=1024):

    waveform = capture_audio_chunk(
        mic,
        chunk_size
    )

    waveform, rms = preprocess_audio(
        waveform
    )

    scores, embedding = extract_features(
        yamnet_model,
        waveform
    )

    result = predict_sound(
        embedding,
        classifier_model
    )

    return result


# --------------------------------------------------
# Run continuously
# --------------------------------------------------

def run(mic):

    while True:

        result = process_audio(mic)

        print(json.dumps(result))

import pyaudio


if __name__ == "__main__":

    audio = pyaudio.PyAudio()

    mic = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    try:
        run(mic)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        mic.stop_stream()
        mic.close()
        audio.terminate()
