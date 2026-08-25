import threading
import time
import numpy as np
import pyaudio
from spectra.models.classifier import yamnet_model, class_names

# --- YAMNET COMPATIBLE CONFIGURATION ---
SAMPLE_RATE = 16000
BUFFER_SECONDS = 1.0
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_SECONDS)  # 16000 samples = 1.0s window
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paFloat32


class YAMNetInferenceWorker(threading.Thread):
    """
    Background thread that continuously listens to the microphone,
    maintains a 1.0s rolling audio buffer, and runs YAMNet inference
    on it in real time.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.rolling_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.lock = threading.Lock()
        self.latest_predictions = []
        self.running = True

    def run(self):
        pa = pyaudio.PyAudio()
        mic = pa.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print("YAMNetInferenceWorker listening... Press CTRL+C to stop.")

        while self.running:
            # 1. Grab raw audio bytes from hardware
            data = mic.read(CHUNK_SIZE, exception_on_overflow=False)
            new_samples = np.frombuffer(data, dtype=np.float32)

            # 2. Slide the rolling buffer: drop oldest samples, append newest
            with self.lock:
                self.rolling_buffer = np.roll(self.rolling_buffer, -len(new_samples))
                self.rolling_buffer[-len(new_samples):] = new_samples
                current_window = self.rolling_buffer.copy()

            # 3. Run inference on the current 1.0s window
            self._run_inference(current_window)

        mic.stop_stream()
        mic.close()
        pa.terminate()

    def _run_inference(self, waveform):
        scores, embeddings, spectrogram = yamnet_model(waveform)
        mean_scores = np.mean(scores.numpy(), axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]
        results = [(class_names[i], float(mean_scores[i])) for i in top_indices]

        with self.lock:
            self.latest_predictions = results

    def get_latest_predictions(self):
        """Thread-safe read of the most recent predictions."""
        with self.lock:
            return self.latest_predictions.copy()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    import sys
    import soundfile as sf

    # WSL has no real microphone access, so we simulate live audio
    # by feeding a test .wav file into the rolling buffer in a loop.
    # This validates the buffer + inference logic without real hardware.
    SIMULATE_WITH_FILE = True

    if SIMULATE_WITH_FILE:
        print("Running in SIMULATION mode (no real microphone in WSL).")
        test_filepath = "raw_data/test_tone.wav"
        waveform, sr = sf.read(test_filepath, dtype='float32')

        worker = YAMNetInferenceWorker()
        # Manually feed chunks into the buffer instead of using PyAudio
        chunk_size = CHUNK_SIZE
        num_chunks = len(waveform) // chunk_size

        for i in range(num_chunks):
            chunk = waveform[i * chunk_size : (i + 1) * chunk_size]
            with worker.lock:
                worker.rolling_buffer = np.roll(worker.rolling_buffer, -len(chunk))
                worker.rolling_buffer[-len(chunk):] = chunk
                current_window = worker.rolling_buffer.copy()

            worker._run_inference(current_window)
            time.sleep(0.1)  # simulate real-time pacing

            if i % 5 == 0:
                print(f"Chunk {i}/{num_chunks}: {worker.get_latest_predictions()[:2]}")

        print("\nSimulation complete. Final predictions:")
        print(worker.get_latest_predictions())

    else:
        # Real microphone mode (for machines with actual audio hardware)
        worker = YAMNetInferenceWorker()
        worker.start()
        try:
            while True:
                time.sleep(1)
                print(worker.get_latest_predictions())
        except KeyboardInterrupt:
            print("\nStopping worker...")
            worker.stop()
