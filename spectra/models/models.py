import os

import tensorflow_hub as hub
from tf_keras.models import load_model
from google.cloud import storage


def load_models():

    print("Loading YAMNet...")

    yamnet_model = hub.load(
    "https://tfhub.dev/google/yamnet/1")

    print("Loading classifier...")

    bucket_name = os.getenv(
        "MODEL_BUCKET_NAME"
    )

    model_name = os.getenv(
        "MODEL_NAME"
    )

    local_path = "/tmp/dense_50.keras"

    storage_client = storage.Client()

    bucket = storage_client.bucket(
        bucket_name
    )

    blob = bucket.blob(
        model_name
    )

    blob.download_to_filename(
        local_path
    )

    classifier_model = load_model(
        local_path
    )

    print("Models loaded.")

    return (
        yamnet_model,
        classifier_model
    )
