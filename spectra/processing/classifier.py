import numpy as np

from spectra.processing.categories import (
    ESC50_CLASSES,
    SOUNDS_DICT,
    DEFAULT_CATEGORY,
)


# ==================================================
# DISPLAY LABELS
# ==================================================

CATEGORY_DISPLAY_LABELS = {
    "Clapping": "Clapping",
    "Human": "Human sound",
    "Animal": "Animal",
    "Nature": "Nature",
    "Alert": "Alert",
    "Vehicle": "Vehicle",
    "Background": "Listening",
}


# ==================================================
# NORMALIZED CLASS NAMES
#
# These are used when several ESC-50 classes are
# grouped into one Spectra concept.
#
# We still preserve the original model prediction
# separately as raw_class_name.
# ==================================================

NORMALIZED_CLASSES = {

    # --------------------------------------------------
    # CLAPPING / IMPACT-LIKE SOUNDS
    # --------------------------------------------------

    "clapping": "clapping",
    "door_wood_knock": "clapping",
    "hand_saw": "clapping",
    "brushing_teeth": "clapping",
    "chainsaw": "clapping",

    # --------------------------------------------------
    # ALERT
    # --------------------------------------------------

    "siren": "siren",
    "glass_breaking": "glass_breaking",
    "fireworks": "fireworks",

    # --------------------------------------------------
    # ANIMAL
    # --------------------------------------------------

    "dog": "dog",
    "rooster": "rooster",
    "pig": "pig",
    "cow": "cow",
    "frog": "frog",
    "cat": "cat",
    "hen": "hen",
    "insects": "insects",
    "sheep": "sheep",
    "crow": "crow",
    "crickets": "crickets",
    "chirping_birds": "chirping_birds",

    # --------------------------------------------------
    # NATURE
    # --------------------------------------------------

    "rain": "rain",
    "sea_waves": "sea_waves",
    "crackling_fire": "crackling_fire",
    "water_drops": "water_drops",
    "wind": "wind",
    "pouring_water": "pouring_water",
    "thunderstorm": "thunderstorm",

    # --------------------------------------------------
    # VEHICLES
    # --------------------------------------------------

    "helicopter": "helicopter",
    "car_horn": "car_horn",
    "engine": "engine",
    "train": "train",
    "airplane": "airplane",

    # --------------------------------------------------
    # HUMAN
    # --------------------------------------------------

    "crying_baby": "crying_baby",
    "sneezing": "sneezing",
    "breathing": "breathing",
    "coughing": "coughing",
    "snoring": "snoring",
    "drinking_sipping": "drinking_sipping",
    "laughing": "laughing",
    "footsteps": "footsteps",
}


# ==================================================
# RAW MODEL PROBABILITIES
# ==================================================

def predict_probabilities(
    embedding,
    classifier_model,
):

    probabilities = classifier_model.predict(
        embedding.reshape(
            1,
            -1,
        ),
        verbose=0,
    )[0]

    return probabilities


# ==================================================
# DISPLAY LABEL FORMATTER
# ==================================================

def format_label(
    label,
):

    return (
        label
        .replace(
            "_",
            " ",
        )
        .strip()
        .title()
    )


# ==================================================
# SOUND PREDICTION
# ==================================================

def predict_sound(
    probabilities,
    max_classes=3,
    confidence_threshold=0.0,
):

    top_indices = np.argsort(
        probabilities
    )[::-1][:max_classes]


    results = []


    for idx in top_indices:

        confidence = float(
            probabilities[
                idx
            ]
        )


        if (
            confidence
            < confidence_threshold
        ):
            continue


        # --------------------------------------------------
        # RAW ESC-50 PREDICTION
        # --------------------------------------------------

        raw_class_name = (
            ESC50_CLASSES[
                idx
            ]
        )


        # --------------------------------------------------
        # BROAD SPECTRA CATEGORY
        # --------------------------------------------------

        category = SOUNDS_DICT.get(
            raw_class_name,
            DEFAULT_CATEGORY,
        )


        # --------------------------------------------------
        # NORMALIZED SPECTRA CLASS
        # --------------------------------------------------

        class_name = (
            NORMALIZED_CLASSES.get(
                raw_class_name,
                raw_class_name,
            )
        )


        # --------------------------------------------------
        # USER-FACING LABEL
        # --------------------------------------------------

        if category == "Clapping":

            display_label = (
                "Clapping"
            )

        else:

            display_label = (
                format_label(
                    class_name
                )
            )


        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        results.append(
            {
                "class_name":
                    class_name,

                "raw_class_name":
                    raw_class_name,

                "display_label":
                    display_label,

                "category":
                    category,

                "confidence":
                    round(
                        confidence,
                        2,
                    ),
            }
        )


    return results
