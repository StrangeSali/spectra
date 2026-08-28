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

    async with websockets.connect("ws://localhost:8000/predict-mic") as ws:
        print("Streaming mic audio... Ctrl+C to stop")
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                await ws.send(data)
                response = await ws.recv()
                print(json.loads(response))
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

asyncio.run(stream_mic())
