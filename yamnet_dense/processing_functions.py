import io
import os
import zipfile
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from google.cloud import storage
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# --- 1. INITIALIZE GCS IN-MEMORY BUFFER ---

BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
ARCHIVE_NAME = os.getenv("GCS_ARCHIVE_NAME")

print(f"Streaming data directly from bucket '{BUCKET_NAME}' into RAM...")

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(ARCHIVE_NAME)

# Download archive directly into RAM
gcs_data_buffer = io.BytesIO()
blob.download_to_file(gcs_data_buffer)
gcs_data_buffer.seek(0)

# Open ZIP archive from RAM
archive = zipfile.ZipFile(gcs_data_buffer)


# --- 2. EXTRACT METADATA IN-MEMORY ---

print("Extracting metadata table...")

metadata_file_name = "ESC-50-master/meta/esc50.csv"

with archive.open(metadata_file_name) as csv_file:
    metadata = pd.read_csv(csv_file)


# Build paths to audio files inside the ZIP
metadata["filepath"] = metadata["filename"].apply(
    lambda x: f"ESC-50-master/audio/{x}"
)

X_paths = metadata["filepath"].values
y = metadata["target"].values


# --- 3. TRAIN / TEST SPLIT ---

X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    X_paths,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --- 4. LOAD YAMNET ---

print("Loading YAMNet...")

yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1"
)


# --- 5. CREATE QUICK LOOKUPS ---

train_set = set(X_train_paths)
test_set = set(X_test_paths)


# --- 6. CACHE AUDIO BYTES IN RAM ---

audio_bytes_cache = {}

print("Caching target audio bytes from archive...")

for member_name in tqdm(archive.namelist()):

    if member_name.endswith(".wav"):

        if member_name in train_set or member_name in test_set:

            audio_bytes_cache[member_name] = archive.read(member_name)


# Close ZIP archive
archive.close()


# --- 7. AUDIO LOADER ---

def load_audio(file_path):
    """
    Load WAV file as mono waveform at 16kHz
    directly from the in-memory cache.
    """

    raw_bytes = audio_bytes_cache[file_path]

    waveform, sr = librosa.load(
        io.BytesIO(raw_bytes),
        sr=16000,
        mono=True
    )

    return waveform.astype(np.float32)


# --- 8. EXTRACT YAMNET EMBEDDING ---

def extract_embedding(file_path):
    """
    Extract a single 1024-dimensional YAMNet embedding
    from an audio file.
    """

    waveform = load_audio(file_path)

    scores, embeddings, spectrogram = yamnet_model(
        waveform
    )

    feature_vector = tf.reduce_mean(
        embeddings,
        axis=0
    )

    return feature_vector.numpy()


# --- 9. EXTRACT TRAINING EMBEDDINGS ---

print("\nExtracting training embeddings...")

X_train = np.array([
    extract_embedding(path)
    for path in tqdm(X_train_paths)
])


# --- 10. EXTRACT TEST EMBEDDINGS ---

print("\nExtracting test embeddings...")

X_test = np.array([
    extract_embedding(path)
    for path in tqdm(X_test_paths)
])


# --- 11. FINAL OUTPUT ---

print("\nAll processing completed entirely in-memory!")

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of classes: {len(np.unique(y))}")
