import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

history = pd.read_csv('training.log')

acc = history.acc
val_acc = history.val_acc

loss = history.loss
val_loss = history.val_loss

plt.plot(np.arange(len(acc[:150])), acc[:150], 'b')
plt.plot(np.arange(len(val_acc[:150])), val_acc[:150], 'r')
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.show()

plt.plot(np.arange(len(acc[:150])), loss[:150], 'b')
plt.plot(np.arange(len(val_acc[:150])), val_loss[:150], 'r')
plt.xlabel("Epoka")
plt.ylabel("Loss")

plt.show()
