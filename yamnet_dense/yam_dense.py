import tensorflow as tf
from processing_functions import X_train, X_test, y_train, y_test
from tf_keras import layers, Input, models
import tensorflow_hub as hub
from tf_keras.callbacks import EarlyStopping

#model yamnet - set to not be trained
yamnet_layer = hub.KerasLayer("https://tfhub.dev/google/yamnet/1", trainable=False)

audio_input = layers.Input(shape=(None,), dtype=tf.float32, name="audio_waveform")

_, embeddings, _ = yamnet_layer(audio_input)

# 4. Your Dense Layers (Connected sequentially in Functional style)
x = layers.Dense(128, activation='relu')(embeddings)
x = layers.Dense(156, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(50, activation='softmax')(x)

#Build and compile the final model
model = models.Model(inputs=audio_input, outputs=outputs)

#Model Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#EarlyStopping
es = EarlyStopping(patience=5, restore_best_weights=True)

model.fit(X_train, y_train, batch_size=32, epochs=1000, verbose=1, callbacks=[es], validation_batch_size=0.2)

model.evaluate(X_test,y_test)
