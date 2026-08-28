import os
import tempfile
import numpy as np
import tensorflow_hub as hub
from tf_keras.models import load_model
from google.cloud import storage
from fastapi import FastAPI, UploadFile, File, HTTPException
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()

from spectra.processing.audio import (
    load_audio_file,
    split_audio_into_windows,
    preprocess_audio
)
from spectra.processing.yamnet_utils import extract_features
from spectra.processing.classifier import predict_probabilities, predict_sound

app = FastAPI(title="Spectra Audio Classifier API")

# --------------------------------------------------
# Carga de modelos
# --------------------------------------------------
MODEL_BUCKET_NAME = os.getenv("MODEL_BUCKET_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
LOCAL_MODEL_PATH = "dense_50.keras"

print("Loading YAMNet...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

print("Loading Classifier...")
classifier_model = None

# Intenta cargar desde GCS si las variables de entorno están presentes
if MODEL_BUCKET_NAME and MODEL_NAME:
    try:
        print(f"Intentando descargar desde GCS: {MODEL_BUCKET_NAME}/{MODEL_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(MODEL_BUCKET_NAME)
        blob = bucket.blob(MODEL_NAME)
        temp_model_path = "/tmp/dense_50.keras"
        blob.download_to_filename(temp_model_path)
        classifier_model = load_model(temp_model_path)
        print("Modelo descargado exitosamente desde GCS.")
    except Exception as e:
        print(f"No se pudo cargar desde GCS ({e}). Se intentará modelo local...")

# Fallback al modelo local
if classifier_model is None and os.path.exists(LOCAL_MODEL_PATH):
    print(f"Cargando modelo local desde: {LOCAL_MODEL_PATH}")
    classifier_model = load_model(LOCAL_MODEL_PATH)

if classifier_model is None:
    print("ADVERTENCIA: No se encontró ningún modelo (ni en GCS ni local).")

# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Spectra AI API is running"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):

    if classifier_model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file not loaded on server."
        )

    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        waveform = load_audio_file(tmp_path)
        windows = split_audio_into_windows(
            waveform,
            sample_rate=16000,
            window_seconds=1.0,
            overlap=0.5
        )

        all_probabilities = []
        for window in windows:
            processed_window, _ = preprocess_audio(window)
            _, embedding = extract_features(yamnet_model, processed_window)
            probabilities = predict_probabilities(embedding, classifier_model)
            all_probabilities.append(probabilities)


        mean_probabilities = np.array(all_probabilities).mean(axis=0)
        results = predict_sound(
            mean_probabilities,
            max_classes=3,
            confidence_threshold=0.20
        )

        return {
            "filename": file.filename,
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
