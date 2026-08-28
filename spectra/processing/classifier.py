import numpy as np

from .categories import (
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
    confidence_threshold=0.0
):



    top_indices = np.argsort(
        probabilities
    )[::-1][:max_classes]

    results = []

    for idx in top_indices:

        confidence = float(
            probabilities[idx]
        )
        esc50_class = ESC50_CLASSES[idx]

        #if confidence < confidence_threshold:
        #    print("Low confidence for ", esc50_class)
        #    continue







        results.append({
            "class_name": esc50_class,
            "confidence": round(
                confidence,
                2
            )
        })

    return results
