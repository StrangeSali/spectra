import asyncio
import websockets

WS_URL = "wss://spectra-1087886990522.europe-west1.run.app/predict-mic"

async def main():
    async with websockets.connect(WS_URL) as ws:
        print("Connected!")

        msg = await ws.recv()
        print(msg)

asyncio.run(main())
