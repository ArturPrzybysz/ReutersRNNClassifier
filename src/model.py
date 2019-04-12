from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Input, Embedding
from ReutersRNNClassifier.src.config import max_len, max_words
import numpy as np


def model(word_matrix, output_size: int):
    inputs = Input(name='inputs', shape=[max_len])

    embeddings = _pretrained_embedding_layer(word_matrix)(inputs)

    X = LSTM(128)(embeddings)
    X = Dropout(0.5)(X)

    X = LSTM(128)(X)
    X = Dropout(0.5)(X)

    X = Dense(256, name='FC1', activation="relu")(X)
    X = Dropout(0.8)(X)

    X = Dense(output_size, name='out_layer', activation='softmax')(X)

    return Model(inputs=inputs, outputs=X)


def _pretrained_embedding_layer(word_matrix: np.array):
    vocab_len = np.shape(word_matrix)[0]
    emb_dim = np.shape(word_matrix)[1]

    embedding_layer = Embedding(vocab_len,
                                emb_dim,
                                trainable=False,
                                weights=word_matrix,
                                input_length=max_len)
    embedding_layer.build((None,))
    embedding_layer.set_weights([word_matrix])

    return embedding_layer
