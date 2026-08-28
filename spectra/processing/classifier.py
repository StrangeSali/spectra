import numpy as np

from processing.categories import (
    ESC50_CLASSES,
    SOUNDS_DICT,
    DEFAULT_CATEGORY
)


def predict_probabilities(
    embedding,
    classifier_model
):

    probabilities = classifier_model.predict(
        embedding.reshape(1, -1),
        verbose=0
    )[0]

    return probabilities


def predict_sound(
    probabilities,
    max_classes=3,
    confidence_threshold=0.20
):

    top_indices = np.argsort(
        probabilities
    )[::-1][:max_classes]

    results = []

    for idx in top_indices:

        confidence = float(
            probabilities[idx]
        )

        if confidence < confidence_threshold:
            continue

        esc50_class = ESC50_CLASSES[idx]

        category = SOUNDS_DICT.get(
            esc50_class,
            DEFAULT_CATEGORY
        )

        results.append({
            "class_name": category,
            "confidence": round(
                confidence,
                2
            )
        })

    return results
