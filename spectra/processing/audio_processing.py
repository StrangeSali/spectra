import sys
import numpy as np
import pyaudio
from sklearn.dummy import DummyClassifier

# --- YAMNET COMPATIBLE CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 10
CHANNELS = 1
FORMAT = pyaudio.paFloat32


def create_model():

    classy = DummyClassifier(strategy="most_frequent")
    X_train = np.zeros((10, CHUNK_SIZE)) # 10 dummy samples
    y_train = np.array([0, 1, 2, 1, 0, 2, 0, 1, 2, 1]) # Fixed array format
    classy.fit(X_train, y_train)

    return classy


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

        y_pred = model.predict(sample.reshape(1,-1))

        print(y_pred, intensity)

if __name__ == "__main__":
    model = create_model()
    sound_to_sample(model)
