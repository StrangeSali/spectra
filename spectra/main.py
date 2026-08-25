from spectra.processing.audio_processing import sound_to_sample
from models.classifier import inicialize_model
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import soundfile as sf

model = inicialize_model()
sound_to_sample(model)

