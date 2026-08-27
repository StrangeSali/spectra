import tensorflow as tf


def extract_features(yamnet_model, waveform):

    scores, embeddings, spectrogram = yamnet_model(waveform)

    embedding = tf.reduce_mean(
        embeddings,
        axis=0
    ).numpy()

    return scores, embedding
