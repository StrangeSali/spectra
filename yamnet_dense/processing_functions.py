import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def extract_embedding(filepath):
    """
    Convert one audio file into a 1024-dimensional YAMNet embedding.
    """

    waveform, _ = librosa.load(
        filepath,
        sr=16000,
        mono=True
    )

    waveform = tf.convert_to_tensor(
        waveform,
        dtype=tf.float32
    )

    _, embeddings, _ = yamnet_model(waveform)

    return tf.reduce_mean(
        embeddings,
        axis=0
    ).numpy()


def load_and_preprocess_esc50(data_path, test_fold=5):
    """
    Returns:
        X_train : (n_train, 1024)
        X_test  : (n_test, 1024)
        y_train : (n_train,)
        y_test  : (n_test,)
    """

    df = pd.read_csv(
        Path(data_path) / "meta" / "esc50.csv"
    )

    df["filepath"] = df["filename"].apply(
        lambda x: str(Path(data_path) / "audio" / x)
    )

    train_df = df[df["fold"] != test_fold]
    test_df = df[df["fold"] == test_fold]

    X_train = np.array([
        extract_embedding(path)
        for path in train_df["filepath"]
    ])

    X_test = np.array([
        extract_embedding(path)
        for path in test_df["filepath"]
    ])

    y_train = train_df["target"].values
    y_test = test_df["target"].values

    return X_train, X_test, y_train, y_test
