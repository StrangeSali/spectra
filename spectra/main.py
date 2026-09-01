import json
import os
import pyaudio

import tensorflow_hub as hub
from tf_keras.models import load_model
from google.cloud import storage
from spectra.models.models import load_models

from spectra.processing.audio import (
    capture_audio_chunk,
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio
)

from spectra.processing.yamnet_utils import (
    extract_features
)

from spectra.processing.classifier import (
    predict_probabilities,
    predict_sound
)

MODEL_BUCKET_NAME = os.getenv(
    "MODEL_BUCKET_NAME"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME"
)

AUDIO_SOURCE = os.getenv(
    "AUDIO_SOURCE",
    "mic"
)

AUDIO_FILE = os.getenv(
    "AUDIO_FILE"
)



yamnet_model, classifier_model = load_models()

# --------------------------------------------------
# Process one waveform
# --------------------------------------------------

def process_waveform(waveform):

    waveform, rms = preprocess_audio(
        waveform
    )

    scores, embedding = extract_features(
        yamnet_model,
        waveform
    )

    probabilities = predict_probabilities(
        embedding,
        classifier_model
    )

    return probabilities


# --------------------------------------------------
# Process microphone
# --------------------------------------------------

def process_microphone(
    mic,
    chunk_size=1024
):

    waveform = capture_audio_chunk(
        mic,
        chunk_size
    )

    probabilities = process_waveform(
        waveform
    )

    result = predict_sound(
        probabilities
    )

    return result


# --------------------------------------------------
# Run microphone continuously
# --------------------------------------------------

def run_microphone(mic):

    while True:

        result = process_microphone(
            mic
        )

        print(
            json.dumps(result)
        )


# --------------------------------------------------
# Process audio file
# --------------------------------------------------

def process_file(
    filepath,
    window_seconds=1.0,
    overlap=0.5,
    max_classes=3,
    confidence_threshold=0.20
):

    print(
        f"Loading audio file: {filepath}"
    )

    waveform = load_audio_file(
        filepath
    )

    duration = (
        len(waveform) / 16000
    )

    print(
        f"Audio duration: {duration:.2f}s"
    )

    windows = split_audio_into_windows(
        waveform,
        sample_rate=16000,
        window_seconds=window_seconds,
        overlap=overlap
    )

    print(
        f"Processing {len(windows)} windows..."
    )

    all_probabilities = []

    for i, window in enumerate(windows):

        print(
            f"Processing window "
            f"{i + 1}/{len(windows)}"
        )

        probabilities = process_waveform(
            window
        )

        all_probabilities.append(
            probabilities
        )

    # Convert to numpy array
    all_probabilities = __import__(
        "numpy"
    ).array(
        all_probabilities
    )

    # Average predictions across windows
    mean_probabilities = (
        all_probabilities.mean(
            axis=0
        )
    )

    # Get final top 3
    results = predict_sound(
        mean_probabilities,
        max_classes=max_classes,
        confidence_threshold=confidence_threshold
    )

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    # ==================================================
    # FILE MODE
    # ==================================================

    if AUDIO_SOURCE == "file":

        if not AUDIO_FILE:

            raise ValueError(
                "AUDIO_FILE must be provided "
                "when AUDIO_SOURCE=file"
            )

        result = process_file(
            AUDIO_FILE,
            window_seconds=1.0,
            overlap=0.5,
            max_classes=3,
            confidence_threshold=0.20
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )


    # ==================================================
    # MICROPHONE MODE
    # ==================================================

    elif AUDIO_SOURCE == "mic":

        audio = pyaudio.PyAudio()

        mic = audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

        try:

            run_microphone(
                mic
            )

        except KeyboardInterrupt:

            print(
                "\nStopping..."
            )

        finally:

            mic.stop_stream()
            mic.close()

            audio.terminate()


    else:

        raise ValueError(
            f"Unknown AUDIO_SOURCE: "
            f"{AUDIO_SOURCE}. "
            f"Use 'mic' or 'file'."
        )
