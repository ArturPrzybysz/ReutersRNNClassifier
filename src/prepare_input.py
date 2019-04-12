import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.utils

from ReutersRNNClassifier.paths import DATASET_PATH
from ReutersRNNClassifier.src.config import *
from ReutersRNNClassifier.src.data_operations.create_dataset import combine_files_from_dir
from ReutersRNNClassifier.src.preprocessing.preprocess_articles import preprocess_articles


def prepare_input(word_df: pd.DataFrame):
    learn_df = combine_files_from_dir(DATASET_PATH)
    learn_df = preprocess_articles(learn_df)
    learn_df = sklearn.utils.shuffle(learn_df)

    X = learn_df.body
    Y = learn_df.places

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    Y = Y.reshape(-1, 1)  # ?
    label_mapping = label_encoder.inverse_transform([0, 1, 2, 3, 4, 5])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    X_train_matrix = bodies_to_vec(word_df, X_train)
    X_test_matrix = bodies_to_vec(word_df, X_test)

    return X_train_matrix, X_test_matrix, Y_train, Y_test, label_mapping


def bodies_to_vec(word_df, X):
    vec_length = word_df.iloc[0].shape[0]
    body_count = len(X)

    X_matrix = np.zeros((max_words * vec_length, body_count))
    for i in np.arange(body_count):
        words = X[i].split()
        for j in np.arange(min(len(words), max_words)):
            if words[j] in word_df.index:
                X_matrix[j * vec_length: j * vec_length + vec_length, i] = word_df.loc[words[j]]
    return X_matrix
