import numpy as np
import librosa


SAMPLE_RATE = 16000


def capture_audio_chunk(mic, chunk_size=1024):

    data = mic.read(
        chunk_size,
        exception_on_overflow=False
    )

    return np.frombuffer(
        data,
        dtype=np.float32
    )


def load_audio_file(filepath):

    waveform, _ = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    return waveform.astype(np.float32)


def split_audio_into_windows(
    waveform,
    sample_rate=SAMPLE_RATE,
    window_seconds=1.0,
    overlap=0.5
):

    window_size = int(
        sample_rate * window_seconds
    )

    hop_size = int(
        window_size * (1 - overlap)
    )

    windows = []

    start = 0

    while start < len(waveform):

        end = start + window_size

        window = waveform[start:end]

        # Ignore very short final chunks
        if len(window) < window_size * 0.5:
            break

        # Pad final chunk if necessary
        if len(window) < window_size:

            window = np.pad(
                window,
                (0, window_size - len(window))
            )

        windows.append(window)

        start += hop_size

    return windows


def preprocess_audio(waveform):

    rms = np.sqrt(
        np.mean(waveform ** 2)
    )

    return waveform, rms
