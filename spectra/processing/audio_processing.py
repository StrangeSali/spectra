import sys
import pandas as pd
import numpy as np
import pyaudio
import threading
from models.classifier import YAMNET_MODEL
from spectra.processing.category_mapping import SOUNDS_DICT, DEFAULT_CATEGORY

# --- YAMNET COMPATIBLE CONFIGURATION ---


def sound_to_sample(model):
    """
    Worker Thread 1: Handles the microphone hardware stream.
    Captures chunks, scales them, and drops them into the queue.
    """
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 1024
    CHANNELS = 1
    FORMAT = pyaudio.paFloat32

    pa = pyaudio.PyAudio()

    mic = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE)

    print("Listening...Press CTRL+C to kill it")

    while True:

        # 1. Grab raw audio bytes from hardware
        data = mic.read(CHUNK_SIZE, exception_on_overflow=False)

        # 2. Convert to perfectly scaled float32 NumPy array (-1.0 to 1.0)

        sample = np.frombuffer(data, dtype=np.float32)

        normalized = sample / 32768.0

        intensity = np.sqrt(np.mean(normalized ** 2))

        # Pass the processed data directly into the classification function
        generate_classification(model, sample)


def generate_classification(model, sample):
    """
    Generates the classification output using the processed data.
    """
    scores, embeddings, spectrogram = model(sample)

    mean_scores = scores.numpy().mean(axis=0)

    class_names = pd.read_csv(model.class_map_path().numpy().decode('utf-8'))['display_name'].tolist()

    top_5_classes = np.argpartition(mean_scores, -3)[-3:]

    top_3_dict = {}
    for idx in top_5_classes:
        category = SOUNDS_DICT.get(class_names[idx].strip(), DEFAULT_CATEGORY)
        confidence = float(mean_scores[idx])

        # Keep the highest confidence if a category repeats
        if category not in top_3_dict or confidence > top_3_dict[category]:
            top_3_dict[category] = confidence

    # 3. Print the resulting dictionary
    print(top_3_dict)

import threading

if __name__ == "__main__":
    # Your instantiated model goes here
    model = YAMNET_MODEL

    # 1. Start the classification worker in the background
    classification_thread = threading.Thread(
        target=generate_classification,
        args=(model,),
        daemon=True
    )
    classification_thread.start()

    # 2. Start the microphone loop in the main thread
    sound_to_sample()
