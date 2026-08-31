import json
import os

import numpy as np
import pyaudio
import tensorflow_hub as hub

from google.cloud import storage
from tf_keras.models import load_model

from spectra.processing.audio import (
    capture_audio_chunk,
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio,
)

from spectra.processing.yamnet_utils import (
    extract_features,
    load_yamnet_class_names,
    detect_speech,
    get_top_yamnet_classes,
)

from spectra.processing.classifier import (
    predict_probabilities,
    predict_sound,
)


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_BUCKET_NAME = os.getenv(
    "MODEL_BUCKET_NAME"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME"
)

AUDIO_SOURCE = os.getenv(
    "AUDIO_SOURCE",
    "mic",
)

AUDIO_FILE = os.getenv(
    "AUDIO_FILE"
)


# ==================================================
# AUDIO CONFIGURATION
# ==================================================

SAMPLE_RATE = 16000

MIC_WINDOW_SECONDS = 1.0

MIC_CHUNK_SIZE = int(
    SAMPLE_RATE
    * MIC_WINDOW_SECONDS
)


# ==================================================
# HYBRID CLASSIFICATION CONFIGURATION
# ==================================================

# YAMNet only overrides ESC-50 when speech evidence
# is sufficiently strong.
#
# You can tune this later if necessary.
YAMNET_SPEECH_THRESHOLD = float(
    os.getenv(
        "YAMNET_SPEECH_THRESHOLD",
        "0.30",
    )
)


# ==================================================
# JSON HELPER
# ==================================================

def make_json_serializable(
    value,
):

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()


    if isinstance(
        value,
        np.generic,
    ):

        return value.item()


    if isinstance(
        value,
        dict,
    ):

        return {
            key:
                make_json_serializable(
                    item
                )
            for key, item
            in value.items()
        }


    if isinstance(
        value,
        (list, tuple),
    ):

        return [
            make_json_serializable(
                item
            )
            for item
            in value
        ]


    return value


# ==================================================
# LOAD YAMNET
# ==================================================

print(
    "Loading YAMNet..."
)

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)

print(
    "YAMNet loaded."
)


# ==================================================
# LOAD YAMNET CLASS NAMES
# ==================================================

print(
    "Loading YAMNet class map..."
)

yamnet_class_names = (
    load_yamnet_class_names(
        yamnet_model
    )
)

print(
    f"Loaded "
    f"{len(yamnet_class_names)} "
    f"YAMNet classes."
)


# ==================================================
# LOAD ESC-50 CLASSIFIER
# ==================================================

print(
    "Loading classifier..."
)

storage_client = storage.Client()

bucket = storage_client.bucket(
    MODEL_BUCKET_NAME
)

blob = bucket.blob(
    MODEL_NAME
)

local_model_path = (
    "/tmp/dense_50.keras"
)

blob.download_to_filename(
    local_model_path
)

classifier_model = load_model(
    local_model_path
)

print(
    "Classifier loaded."
)


# ==================================================
# PROCESS ONE WAVEFORM
# ==================================================

def process_waveform(
    waveform,
):

    waveform, rms = preprocess_audio(
        waveform
    )


    yamnet_scores, embedding = (
        extract_features(
            yamnet_model,
            waveform,
        )
    )


    esc50_probabilities = (
        predict_probabilities(
            embedding,
            classifier_model,
        )
    )


    return (
        esc50_probabilities,
        yamnet_scores,
        rms,
    )


# ==================================================
# HYBRID PREDICTION
# ==================================================

def hybrid_predict(
    esc50_probabilities,
    yamnet_scores,
    max_classes=3,
):
    """
    Hybrid strategy:

    1. Check whether YAMNet strongly detects speech.
    2. If yes:
           return Human / People talking.
    3. Otherwise:
           use our trained ESC-50 classifier.

    YAMNet is therefore a specialist detector,
    not the main Spectra classifier.
    """

    (
        is_speech,
        speech_confidence,
        yamnet_speech_class,
    ) = detect_speech(
        yamnet_scores,
        yamnet_class_names,
        threshold=(
            YAMNET_SPEECH_THRESHOLD
        ),
    )


    # ==================================================
    # SPEECH OVERRIDE
    # ==================================================

    if is_speech:

        return [
            {
                "class_name":
                    "speech",

                "raw_class_name":
                    yamnet_speech_class,

                "display_label":
                    "People talking",

                "category":
                    "Human",

                "confidence":
                    round(
                        speech_confidence,
                        2,
                    ),

                "source":
                    "yamnet",
            }
        ]


    # ==================================================
    # ESC-50 CLASSIFICATION
    # ==================================================

    results = predict_sound(
        esc50_probabilities,
        max_classes=max_classes,
        confidence_threshold=0.0,
    )


    # Mark where the prediction came from.
    for result in results:

        result["source"] = (
            "esc50"
        )


    return results


# ==================================================
# PROCESS MICROPHONE
# ==================================================

def process_microphone(
    mic,
    chunk_size=MIC_CHUNK_SIZE,
):

    waveform = capture_audio_chunk(
        mic,
        chunk_size,
    )


    (
        esc50_probabilities,
        yamnet_scores,
        rms,
    ) = process_waveform(
        waveform
    )


    result = hybrid_predict(
        esc50_probabilities,
        yamnet_scores,
        max_classes=3,
    )


    return (
        result,
        rms,
    )


# ==================================================
# RUN MICROPHONE CONTINUOUSLY
# ==================================================

def run_microphone(
    mic,
):

    while True:

        predictions, rms = (
            process_microphone(
                mic
            )
        )


        predictions = (
            make_json_serializable(
                predictions
            )
        )


        rms = float(
            rms
        )


        print(
            "Predictions:",
            predictions,
        )


        print(
            "RMS:",
            rms,
        )


        print(
            json.dumps(
                {
                    "predictions":
                        predictions,

                    "rms":
                        rms,
                },
                indent=2,
            )
        )


# ==================================================
# PROCESS AUDIO FILE
# ==================================================

def process_file(
    filepath,
    window_seconds=1.0,
    overlap=0.5,
    max_classes=3,
    confidence_threshold=0.20,
):

    print(
        f"Loading audio file: "
        f"{filepath}"
    )


    waveform = load_audio_file(
        filepath
    )


    duration = (
        len(waveform)
        / SAMPLE_RATE
    )


    print(
        f"Audio duration: "
        f"{duration:.2f}s"
    )


    windows = split_audio_into_windows(
        waveform,
        sample_rate=SAMPLE_RATE,
        window_seconds=window_seconds,
        overlap=overlap,
    )


    print(
        f"Processing "
        f"{len(windows)} "
        f"windows..."
    )


    all_esc50_probabilities = []

    all_yamnet_scores = []


    for i, window in enumerate(
        windows
    ):

        print(
            f"Processing window "
            f"{i + 1}/"
            f"{len(windows)}"
        )


        (
            esc50_probabilities,
            yamnet_scores,
            rms,
        ) = process_waveform(
            window
        )


        all_esc50_probabilities.append(
            esc50_probabilities
        )


        # Average the YAMNet class scores
        # within each one-second window.
        yamnet_mean_scores = (
            np.asarray(
                yamnet_scores
            ).mean(
                axis=0
            )
        )


        all_yamnet_scores.append(
            yamnet_mean_scores
        )


    if not all_esc50_probabilities:

        return []


    # ==================================================
    # ESC-50 AVERAGE
    # ==================================================

    all_esc50_probabilities = (
        np.asarray(
            all_esc50_probabilities,
            dtype=np.float32,
        )
    )


    mean_esc50_probabilities = (
        all_esc50_probabilities.mean(
            axis=0
        )
    )


    # ==================================================
    # YAMNET AVERAGE
    # ==================================================

    all_yamnet_scores = np.asarray(
        all_yamnet_scores,
        dtype=np.float32,
    )


    mean_yamnet_scores = (
        all_yamnet_scores.mean(
            axis=0
        )
    )


    # ==================================================
    # HYBRID FINAL RESULT
    # ==================================================

    results = hybrid_predict(
        mean_esc50_probabilities,
        mean_yamnet_scores,
        max_classes=max_classes,
    )


    # Apply confidence threshold for file mode.
    results = [
        result
        for result in results
        if (
            float(
                result.get(
                    "confidence",
                    0.0,
                )
            )
            >= confidence_threshold
        )
    ]


    return make_json_serializable(
        results
    )


# ==================================================
# DEBUG YAMNET
# ==================================================

def print_yamnet_debug(
    yamnet_scores,
    top_k=8,
):

    top_classes = (
        get_top_yamnet_classes(
            yamnet_scores,
            yamnet_class_names,
            top_k=top_k,
        )
    )


    print(
        "\n--- YAMNet ---"
    )


    for item in top_classes:

        print(
            f"{item['class_name']}: "
            f"{item['confidence']:.3f}"
        )


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":


    # ==================================================
    # FILE MODE
    # ==================================================

    if AUDIO_SOURCE == "file":


        if not AUDIO_FILE:

            raise ValueError(
                "AUDIO_FILE must be "
                "provided when "
                "AUDIO_SOURCE=file"
            )


        result = process_file(
            AUDIO_FILE,
            window_seconds=1.0,
            overlap=0.5,
            max_classes=3,
            confidence_threshold=0.20,
        )


        print(
            json.dumps(
                result,
                indent=2,
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
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=1024,
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


    # ==================================================
    # INVALID MODE
    # ==================================================

    else:

        raise ValueError(
            f"Unknown AUDIO_SOURCE: "
            f"{AUDIO_SOURCE}. "
            f"Use 'mic' or 'file'."
        )
