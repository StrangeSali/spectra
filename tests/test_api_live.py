import asyncio
import websockets
import pyaudio
import numpy as np
import json

CHUNK = 16000  # 1 second at 16kHz
RATE = 16000

async def stream_mic():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # Use 127.0.0.1 explicitly to avoid IPv6/IPv4 lookup confusion
    async with websockets.connect("ws://127.0.0.1:8000/predict-mic") as ws:
        print("Streaming mic audio... Ctrl+C to stop")
        try:
            while True:
                # CRITICAL FIX: Run the blocking PyAudio read in a separate thread
                data = await asyncio.to_thread(
                    stream.read, CHUNK, exception_on_overflow=False
                )

                await ws.send(data)
                response = await ws.recv()
                print(json.loads(response))
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

try:
    asyncio.run(stream_mic())
except KeyboardInterrupt:
    pass
