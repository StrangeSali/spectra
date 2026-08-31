import asyncio
import websockets
import pyaudio
import numpy as np
import json

CHUNK = 4000  # 1 second at 16kHz
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

    async with websockets.connect("ws://127.0.0.1:8000/predict-mic") as ws:
        # Consume the initial "connected" message first
        hello = json.loads(await ws.recv())
        print("Connected:", hello)
        session_id = hello.get("session_id")

        print("Streaming mic audio... Ctrl+C to stop")

        async def send_audio():
            while True:
                data = await asyncio.to_thread(
                    stream.read, CHUNK, exception_on_overflow=False
                )
                await ws.send(data)

        async def receive_predictions():
            async for message in ws:
                print(json.loads(message))

        try:
            # Run sending and receiving concurrently instead of alternating,
            # so mic capture never blocks waiting on a network round trip.
            await asyncio.gather(send_audio(), receive_predictions())
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
