import asyncio
import websockets
import pyaudio
import sys
import os  # <-- Add this import

# 1. READ FROM ENVIRONMENT VARIABLES (with a safety fallback check)
CLOUD_RUN_WS_URL = os.environ.get("CLOUD_RUN_WS_URL", "")

# Audio parameters matching YAMNet requirements
RATE = 16000
CHUNK = 1024

async def stream_mic():
    if "YOUR_CLOUD_RUN_URL_HERE" in CLOUD_RUN_WS_URL:
        print("❌ Error: Please update CLOUD_RUN_WS_URL with your actual Cloud Run URL.")
        sys.exit(1)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print(f"🎙️ Connected to local microphone.")
    print(f"🌐 Connecting to Cloud Run: {CLOUD_RUN_WS_URL}")

    try:
        async with websockets.connect(CLOUD_RUN_WS_URL) as websocket:
            print("🚀 Connection established! Streaming audio data...")
            print("🛑 Press Ctrl+C to stop streaming.")

            while True:
                # Read raw binary PCM audio data from microphone
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Send binary data down the WebSocket pipe
                await websocket.send(data)

    except KeyboardInterrupt:
        print("\n⏹️ Streaming stopped by user.")
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n❌ Cloud Run disconnected: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    asyncio.run(stream_mic())
