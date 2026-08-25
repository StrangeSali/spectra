import sys
import pandas as pd
import numpy as np
import pyaudio
from spectra.processing.category_mapping import SOUNDS_DICT, DEFAULT_CATEGORY

# --- YAMNET COMPATIBLE CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 10
CHANNELS = 1
FORMAT = pyaudio.paFloat32

def sound_to_sample(model):
    """
    Worker Thread 1: Handles the microphone hardware stream.
    Captures chunks, scales them, and drops them into the queue.
    """

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

        scores, embeddings, spectrogram = model(sample)

        mean_scores = scores.numpy().mean(axis=0)

        top_5_classes = np.argpartition(mean_scores, -3)[-3:]

        class_names = pd.read_csv(model.class_map_path().numpy().decode('utf-8'))['display_name'].tolist()

        yamnet_classes = [SOUNDS_DICT.get(class_names[i].strip(),DEFAULT_CATEGORY) for i in top_5_classes]

        print(yamnet_classes)
