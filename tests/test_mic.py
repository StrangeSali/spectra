import json
import queue
import sounddevice as sd
import websocket
import os

# Your deployed Cloud Run WebSocket URL
WS_URL = os.getenv("CLOUD_RUN_WS_URL")

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())


def main():
    print("Connecting to API...")

    ws = websocket.create_connection(WS_URL)

    print("Connected!")
    print("🎤 Microphone is ON")
    print("Speak / clap / knock / make sounds...")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
        ):

            while True:
                audio = audio_queue.get()

                # Send raw float32 audio to the API
                ws.send(audio.tobytes(), opcode=websocket.ABNF.OPCODE_BINARY)

                # Check whether API returned a prediction
                ws.settimeout(0.01)

                try:
                    response = ws.recv()

                    if response:
                        print("Prediction:", response)

                except websocket.WebSocketTimeoutException:
                    pass

    except KeyboardInterrupt:
        print("\nStopping microphone...")

    finally:
        ws.close()


if __name__ == "__main__":
    main()
