import csv

import numpy as np

import pandas as pd


def embeddings(glove_model_path: str):
    words = pd.read_csv(glove_model_path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    vocab_len = len(words) + 1
    emb_dim = words.iloc[0].shape[0]

    emb_matrix = np.zeros((vocab_len, emb_dim))
    for i in np.arange(vocab_len - 1):
        emb_matrix[i, :] = words.iloc[i]

    return emb_matrix, words
