import io
import os
import tarfile  # Standard streaming archive reader
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

# Download the archive bytes directly into a seekable RAM buffer
# This handles the "seeking backwards" issue cleanly with zero disk writes
gcs_data_buffer = io.BytesIO()
blob.download_to_file(gcs_data_buffer)
gcs_data_buffer.seek(0)  # Reset buffer position to the beginning

# Open the archive from the seekable RAM space
tar = tarfile.open(fileobj=gcs_data_buffer, mode="r:gz")

# --- 2. EXTRACT METADATA IN-MEMORY ---
print("Extracting metadata table...")
metadata_file_name = "UrbanSound8K/metadata/UrbanSound8K.csv"
csv_file = tar.extractfile(metadata_file_name)
metadata = pd.read_csv(io.BytesIO(csv_file.read()))

# Build the structural internal path names
metadata["filepath"] = metadata.apply(
    lambda row: f"UrbanSound8K/audio/fold{row['fold']}/{row['slice_file_name']}",
    axis=1
)

X_paths = metadata["filepath"].values
y = metadata["classID"].values

# Train test split
X_train_paths, X_test_paths, y_train, y_test = train_test_split(
    X_paths, y, test_size=0.2, random_state=42, stratify=y
)

# Load Yamnet (UPDATED: Points to the correct full directory endpoint)
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

# Create quick lookups
train_set = set(X_train_paths)
test_set = set(X_test_paths)

# This dictionary stores our in-memory audio bytes mapping
# Key: 'UrbanSound8K/audio/fold1/xxx.wav' -> Value: bytes
audio_bytes_cache = {}

# --- 3. PARSE CHUNKS FROM RAM CACHE ---
print("Caching target audio bytes from archive...")
for member in tqdm(tar, total=len(metadata)):
    if member.isfile() and member.name.endswith(".wav"):
        if member.name in train_set or member.name in test_set:
            # Pull bytes out of network stream block
            audio_bytes_cache[member.name] = tar.extractfile(member).read()

# Safely close network connections now that data is processed
tar.close()

# Audio loader reads from RAM instead of your local disk
def load_audio(file_path):
    """
    Load wav file as mono waveform at 16kHz directly from RAM cache.
    """
    raw_bytes = audio_bytes_cache[file_path]

    waveform, sr = librosa.load(
        io.BytesIO(raw_bytes),  # Uses in-memory bytes wrapper instead of path string
        sr=16000,
        mono=True
    )
    return waveform.astype(np.float32)

# Extract embedding
def extract_embedding(file_path):
    """
    Extract a single 1024-dimensional embedding
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

print("\nExtracting training embeddings...")
X_train = np.array([
    extract_embedding(path)
    for path in tqdm(X_train_paths)
])

print("\nExtracting test embeddings...")
X_test = np.array([
    extract_embedding(path)
    for path in tqdm(X_test_paths)
])

print("\nAll processing completed entirely in-memory!")
print(f"X_train shape: {X_train.shape}")
