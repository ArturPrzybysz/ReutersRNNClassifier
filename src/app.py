import numpy
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.python.keras.optimizers import Adam

from ReutersRNNClassifier.paths import GLOVE_50D_PATH
from ReutersRNNClassifier.src.embeddings import embeddings
from ReutersRNNClassifier.src.prepare_embedding_matrix import prepare_embedding_matrix
from ReutersRNNClassifier.src.prepare_input import prepare_input
from ReutersRNNClassifier.src.model import model
from sklearn.metrics import confusion_matrix

emb_matrix, words = embeddings(GLOVE_50D_PATH)

sequences_matrix, test_sequences_matrix, Y_train, Y_test, label_mapping, word_index = prepare_input()

embedding_matrix, num_words = prepare_embedding_matrix(word_index, words)

model = model(word_matrix=embedding_matrix, output_size=len(label_mapping), num_words=num_words)
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(),
              metrics=["sparse_categorical_accuracy", "accuracy"])
history = model.fit(sequences_matrix, Y_train, batch_size=128, epochs=25,
                    validation_split=0.15, callbacks=[CSVLogger('training.log')])

accr = model.evaluate(test_sequences_matrix, Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
print(label_mapping)
y_pred = model.predict(test_sequences_matrix)
y_pred = numpy.argmax(y_pred, axis=1).T
print('Confusion Matrix')
print(confusion_matrix(y_pred=y_pred, y_true=Y_test))
