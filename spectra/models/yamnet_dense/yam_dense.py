import os
# Force legacy Keras configuration before any TF imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from spectra.models.yamnet_dense.processing_functions import X_train, X_test, y_train, y_test
from tf_keras import layers, models, callbacks
import tensorflow_hub as hub

# 1. Load the model from Kaggle/TFHub
yamnet_layer = hub.load("https://tfhub.dev/google/yamnet/1", trainable=False)

# 2. Build the functional architecture
audio_input = layers.Input(shape=(None,), dtype=tf.float32, name="audio_waveform")
_, embeddings, _ = yamnet_layer(audio_input)

# Custom dense feature processing
x = layers.Dense(128, activation='relu')(embeddings)
x = layers.Dense(156, activation='relu')(x)
x = layers.Dropout(0.2)(x)

# FIX: Pool the frames together so the model predicts ONE label per whole audio file
x = layers.GlobalAveragePooling1D()(x)

# Final classification layer (50 ESC-50 targets)
outputs = layers.Dense(50, activation='softmax')(x)

# 3. Instantiate and compile model
model = models.Model(inputs=audio_input, outputs=outputs)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 4. Set up early stopping callback
es = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 5. Fit the model using your exact processing variables
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=1000,
    verbose=1,
    callbacks=[es],
    validation_split=0.2  # FIX: Corrected argument for 20% validation split
)

# 6. Evaluate on test set
model.evaluate(X_test, y_test)
