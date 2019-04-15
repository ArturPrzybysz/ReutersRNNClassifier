import numpy as np
import pandas as pd

from ReutersRNNClassifier.src.config import MAX_NUM_WORDS, EMBEDDING_DIM


def prepare_embedding_matrix(word_index, words: pd.DataFrame):
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue

        if word in words.index:
            embedding_matrix[i] = words.loc[word]

    return embedding_matrix, num_words
