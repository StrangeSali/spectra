import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets

WS_URL = "wss://spectra-1087886990522.europe-west1.run.app/predict-mic"

RATE = 8000
SECONDS = 1

async def main():

    async with websockets.connect(WS_URL) as ws:

        print(await ws.recv())

        while True:

            audio = sd.rec(
                int(RATE * SECONDS),
                samplerate=RATE,
                channels=1,
                dtype="float32"
            )

            sd.wait()

            await ws.send(audio.flatten().tobytes())

            try:
                response = await asyncio.wait_for(
                    ws.recv(),
                    timeout=2
                )

                print(json.loads(response))

            except asyncio.TimeoutError:
                print("No prediction returned")

asyncio.run(main())
