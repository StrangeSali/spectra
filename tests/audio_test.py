import sys
import queue
import threading
import time
import numpy as np
import pyaudio

# --- YAMNET COMPATIBLE CONFIGURATION ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 10
CHANNELS = 1
FORMAT = pyaudio.paFloat32

# The Bridge Queue
audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

def mic_stream_thread():
    """Worker 1: Constantly reads from microphone and pushes to queue."""
    pa_instance = pyaudio.PyAudio()

    try:
        mic = pa_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
    except Exception as e:
        print(f"\n[ERROR] Could not open microphone: {e}")
        print("Please check your microphone connection and Mac Privacy Settings.")
        sys.exit(1)

    print(" -> Microphone Stream Thread started successfully (Aubio Bypassed!).")

    while True:
        try:
            # Read chunk
            data = mic.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.float32)

            # Push to queue (asynchronous hand-off)
            try:
                audio_queue.put_nowait(samples)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                    audio_queue.put_nowait(samples)
                except queue.Empty:
                    pass

            # Calculate volume to make sure the mic is alive
            volume = np.sum(samples**2) / len(samples)
            if volume > 0.001:
                print(f"[Mic Live] Audio energy detected (Vol: {volume:.4f})")

        except Exception as e:
            print(f"\n[Mic Error] {e}")
            break

def mock_yamnet_thread():
    """Worker 2: Simulates YAMNet pulling data out of the queue."""
    print(" -> Mock YAMNet Inference Thread started successfully.")
    print("Listening... Make some noise to test. Press Ctrl+C to stop.\n")

    while True:
        # Pull the scaled float32 array out of the queue
        samples = audio_queue.get()

        # Simulate YAMNet processing delay (0.04 seconds)
        time.sleep(0.04)

        current_queue_size = audio_queue.qsize()

        if len(samples) == CHUNK_SIZE:
            print(f"      [YAMNet AI] Safely pulled {len(samples)} scaled floats. Queue backlog: {current_queue_size}/10")

        audio_queue.task_done()

if __name__ == "__main__":
    # Start the mic thread
    mic_worker = threading.Thread(target=mic_stream_thread, daemon=True)
    mic_worker.start()

    # Start the consumer thread
    try:
        mock_yamnet_thread()
    except KeyboardInterrupt:
        print("\nTesting stopped by user. Clean exit.")
