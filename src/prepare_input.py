import numpy as np

from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.utils

from ReutersRNNClassifier.paths import DATASET_PATH
from ReutersRNNClassifier.src.config import MAX_NUM_WORDS, MAX_SEQ_LENGTH
from ReutersRNNClassifier.src.data_operations.create_dataset import combine_files_from_dir
from ReutersRNNClassifier.src.preprocessing.preprocess_articles import preprocess_articles


def prepare_input():
    learn_df = combine_files_from_dir(DATASET_PATH)
    learn_df = preprocess_articles(learn_df)
    learn_df = sklearn.utils.shuffle(learn_df)

    X = learn_df.body
    Y = learn_df.places

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    Y = Y.reshape(-1, 1)
    label_mapping = label_encoder.inverse_transform(np.arange(6))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    tok = Tokenizer(num_words=MAX_NUM_WORDS)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH)

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = pad_sequences(test_sequences, maxlen=MAX_SEQ_LENGTH)

    word_index = tok.word_index
    print('Found %s unique tokens.' % len(word_index))

    return sequences_matrix, test_sequences_matrix, Y_train, Y_test, label_mapping, word_index
