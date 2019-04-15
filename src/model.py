from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Input, Embedding
from ReutersRNNClassifier.src.config import MAX_SEQ_LENGTH, EMBEDDING_DIM
import numpy as np


def model(word_matrix, output_size: int, num_words: int):
    inputs = Input(name='inputs', shape=[MAX_SEQ_LENGTH])

    embeddings = _pretrained_embedding_layer(word_matrix, num_words)(inputs)

    X = LSTM(128)(embeddings)
    X = Dropout(0.4)(X)

    X = Dense(32, name='Dense1', activation="relu")(X)
    X = Dropout(0.3)(X)

    X = Dense(output_size, name='out_layer', activation='softmax')(X)

    return Model(inputs=inputs, outputs=X)


def _pretrained_embedding_layer(embedding_matrix: np.array, num_words):
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQ_LENGTH,
                                trainable=False)

    return embedding_layer
