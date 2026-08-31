import csv

import numpy as np
import tensorflow as tf


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def extract_features(
    yamnet_model,
    waveform,
):

    scores, embeddings, spectrogram = (
        yamnet_model(
            waveform
        )
    )

    embedding = tf.reduce_mean(
        embeddings,
        axis=0,
    ).numpy()

    return (
        scores.numpy(),
        embedding,
    )


# ==================================================
# LOAD YAMNET CLASS NAMES
# ==================================================

def load_yamnet_class_names(
    yamnet_model,
):
    """
    Load the official YAMNet class names from the
    class-map file bundled with the TensorFlow Hub model.
    """

    class_map_path = (
        yamnet_model
        .class_map_path()
        .numpy()
        .decode("utf-8")
    )

    class_names = []

    with open(
        class_map_path,
        "r",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            class_names.append(
                row["display_name"]
            )

    return class_names


# ==================================================
# MEAN YAMNET SCORES
# ==================================================

def get_mean_yamnet_scores(
    scores,
):
    """
    YAMNet produces scores for several time frames.

    Collapse them into one score per class.
    """

    scores = np.asarray(
        scores
    )

    if scores.ndim == 1:
        return scores

    return scores.mean(
        axis=0
    )


# ==================================================
# TOP YAMNET CLASSES
# ==================================================

def get_top_yamnet_classes(
    scores,
    class_names,
    top_k=5,
):

    mean_scores = (
        get_mean_yamnet_scores(
            scores
        )
    )

    top_indices = np.argsort(
        mean_scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "class_name":
                    class_names[
                        index
                    ],

                "confidence":
                    float(
                        mean_scores[
                            index
                        ]
                    ),
            }
        )

    return results


# ==================================================
# SPEECH DETECTION
# ==================================================

def detect_speech(
    scores,
    class_names,
    threshold=0.30,
):
    """
    Detect whether YAMNet has meaningful evidence
    for human speech.

    ESC-50 does not contain a normal speech class,
    so YAMNet acts as a specialist for this case.

    Returns:

        is_speech
        speech_confidence
        detected_yamnet_class
    """

    mean_scores = (
        get_mean_yamnet_scores(
            scores
        )
    )

    best_score = 0.0
    best_label = None

    speech_terms = (
        "speech",
        "conversation",
        "narration",
        "monologue",
        "talking",
    )

    for index, label in enumerate(
        class_names
    ):

        normalized_label = (
            label.lower()
        )

        if not any(
            term in normalized_label
            for term in speech_terms
        ):
            continue

        score = float(
            mean_scores[
                index
            ]
        )

        if score > best_score:

            best_score = score
            best_label = label

    return (
        best_score >= threshold,
        best_score,
        best_label,
    )
