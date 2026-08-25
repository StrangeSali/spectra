import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub

# Load model and class names once at module level
print("Loading YAMNet model...")
YAMNET_MODEL = hub.load("https://tfhub.dev/google/yamnet/1")

_class_map_path = YAMNET_MODEL.class_map_path().numpy().decode("utf-8")
CLASS_NAMES = pd.read_csv(_class_map_path)["display_name"].tolist()
print("YAMNet model loaded successfully.")


def load_audio(filepath: str) -> np.ndarray:
    """Load a .wav file and ensure it matches YAMNet's expected format:

    mono, 16kHz, float32.
    """
    waveform, _ = librosa.load(filepath, sr=16000, mono=True, dtype=np.float32)
    return waveform


def classify_wav(filepath: str, top_n: int = 5) -> list[tuple[str, float]]:
    """Run YAMNet inference on a .wav file and return the top N predicted classes

    with their confidence scores.
    """
    waveform = load_audio(filepath)

    # Run inference using global YAMNet instance
    scores, embeddings, spectrogram = YAMNET_MODEL(waveform)

    # Average scores across time frames
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Get indices of top N predictions
    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    results = [
        (CLASS_NAMES[i], float(mean_scores[i])) for i in top_indices
    ]

    return results


if __name__ == "__main__":
    # Quick manual test when running this script directly
    test_filepath = (
        "/home/pablo/code/spectra/raw_data/esc50_dataset/audio/1-69641-A-3.wav"
    )
    predictions = classify_wav(test_filepath)

    print(f"\nTop predictions for {test_filepath}:")
    for class_name, confidence in predictions:
        print(f"  {class_name}: {confidence:.4f}")
