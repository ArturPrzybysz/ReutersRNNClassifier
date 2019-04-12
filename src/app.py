import numpy
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam

from ReutersRNNClassifier.paths import GLOVE_50D_PATH
from ReutersRNNClassifier.src.embeddings import embeddings
from ReutersRNNClassifier.src.prepare_input import prepare_input
from ReutersRNNClassifier.src.model import model
from sklearn.metrics import classification_report, confusion_matrix

word_matrix, words = embeddings(GLOVE_50D_PATH)

sequences_matrix, test_sequences_matrix, Y_train, Y_test, label_mapping = prepare_input(words)

model = model(word_matrix=word_matrix, output_size=6)
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=["sparse_categorical_accuracy"])
model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])

accr = model.evaluate(test_sequences_matrix, Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
print(label_mapping)
y_pred = model.predict(test_sequences_matrix)
y_pred = numpy.argmax(y_pred, axis=1).T
print('Confusion Matrix')
print(confusion_matrix(y_pred=y_pred, y_true=Y_test))
# print('Classification Report')
# print(classification_report(label_mapping, y_pred))
