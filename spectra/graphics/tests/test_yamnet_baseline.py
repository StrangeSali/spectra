from spectra.models.classifier import classify_wav
from spectra.processing.category_mapping import map_to_category
from spectra.processing.category_mapping import get_top_category


def test_clapping_baseline():

    filepath = "raw_data/test_audio/clapping.wav"

    predictions = classify_wav(
        filepath,
        top_n=5
    )

    print("\nYAMNet predictions:")

    for class_name, confidence in predictions:
        print(
            f"{class_name}: {confidence:.2f}"
        )

    # Take YAMNet's strongest prediction
    predicted_category = get_top_category(
    predictions
)

    print(
        f"Expected: Human | "
        f"Predicted: {predicted_category}"
    )

    assert predicted_category == "Human"
