import numpy as np


def capture_audio_chunk(mic, chunk_size=1024):

    data = mic.read(
        chunk_size,
        exception_on_overflow=False
    )

    return np.frombuffer(
        data,
        dtype=np.float32
    )


def preprocess_audio(waveform):

    rms = np.sqrt(
        np.mean(waveform ** 2)
    )

    return waveform, rms
