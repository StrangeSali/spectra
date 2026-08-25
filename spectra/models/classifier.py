import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import soundfile as sf


def inicialize_model():
    # Load YAMNet once when this module is imported (not on every call)
    print("Loading YAMNet model...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

    # Load the 521 AudioSet class names
    class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = pd.read_csv(class_map_path)['display_name'].tolist()

    print("YAMNet model loaded successfully.")
    return yamnet_model

def load_audio(filepath):
    """
    Load a .wav file and ensure it matches YAMNet's expected format:
    mono, 16kHz, float32.
    """
    waveform, sample_rate = sf.read(filepath, dtype='float32')

    # Convert stereo to mono if needed
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Warn if sample rate doesn't match YAMNet's expected 16kHz
    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz sample rate, got {sample_rate} Hz. "
            f"Resampling not yet implemented."
        )

    return waveform
def classify_wav(filepath, top_n=5):
    """
    Run YAMNet inference on a .wav file and return the top N predicted classes
    with their confidence scores.

    Args:
        filepath (str): path to the .wav file
        top_n (int): how many top predictions to return

    Returns:
        list of (class_name, confidence_score) tuples, sorted by confidence desc
    """
    waveform = load_audio(filepath)

    scores, embeddings, spectrogram = yamnet_model(waveform)

    # Average scores across all time frames to get one prediction per class
    mean_scores = np.mean(scores.numpy(), axis=0)

    # Get indices of the top N highest scoring classes
    top_indices = np.argsort(mean_scores)[::-1][:top_n]

    results = [(class_names[i], float(mean_scores[i])) for i in top_indices]

    return results


if __name__ == "__main__":
    # Quick manual test when running this script directly
    test_filepath = "../raw_data/test_tone.wav"
    predictions = classify_wav(test_filepath)

    print(f"\nTop predictions for {test_filepath}:")
    for class_name, confidence in predictions:
        print(f"  {class_name}: {confidence:.4f}")
