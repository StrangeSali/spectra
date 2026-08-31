import asyncio
import websockets
import pyaudio
import json

CHUNK = 4000
RATE = 16000

WS_URL = "wss://spectra-1087886990522.europe-west1.run.app/predict-mic"

async def stream_mic():
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    async with websockets.connect(WS_URL) as ws:

        hello = json.loads(await ws.recv())
        print("Connected:", hello)

        print("Streaming mic audio... Ctrl+C to stop")

        async def send_audio():
            while True:
                data = await asyncio.to_thread(
                    stream.read,
                    CHUNK,
                    exception_on_overflow=False
                )

                await ws.send(data)

        async def receive_predictions():
            async for message in ws:
                try:
                    print(json.dumps(json.loads(message), indent=2))
                except Exception:
                    print(message)

        try:
            await asyncio.gather(
                send_audio(),
                receive_predictions()
            )

        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

if __name__ == "__main__":
    asyncio.run(stream_mic())
