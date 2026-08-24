import sys
import queue
import threading
import numpy as np
import pyaudio

# --- YAMNET COMPATIBLE CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 10
CHANNELS = 1
FORMAT = pyaudio.paFloat32

# The thread-safe Bridge Queue for asynchronous streaming
audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def sound_to_sample():
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

    while True:

        # 1. Grab raw audio bytes from hardware
        data = mic.read(CHUNK_SIZE, exception_on_overflow=False)

        # 2. Convert to perfectly scaled float32 NumPy array (-1.0 to 1.0)
        samples = np.frombuffer(data, dtype=np.float32)

        # 3. Asynchronously push to queue (Drop oldest if YAMNet falls behind)
        try:
            audio_queue.put_nowait(samples)
        except queue.Full:
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(samples)
            except queue.Empty:
                pass

if __name__ == "__main__":
    # Start the microphone collector in the background (Thread 1)
    mic_worker = threading.Thread(target=sound_to_array_thread, daemon=True)
    mic_worker.start()

    # Run the YAMNet consumer loop on the main thread (Thread 2)
    try:
        yamnet_inference_thread()
    except KeyboardInterrupt:
        print("\nProgram stopped by user. Clean exit completed.")
        sys.exit(0)
