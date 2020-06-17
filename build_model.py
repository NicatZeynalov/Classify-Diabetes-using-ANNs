#import libraries

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

model = tf.keras.models.Sequential()
#first layer
model.add(tf.keras.layers.Dense(200, activation='relu',input_shape=(X_train.shape[1],)))
#dropout
model.add(tf.keras.layers.Dropout(0.5))
#second layer
model.add(tf.keras.layers.Dense(200, activation='relu'))
#third dropout
model.add(tf.keras.layers.Dropout(0.5))
#output layer
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
#model compile
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model fit
history = model.fit(X_train, y_train, epochs =200, validation_data=(X_test, y_test) )

#plot model accuracy
acc = history.history['loss']
val_acc = history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Loss')
plt.legend(loc=0)
plt.figure()
plt.show()

#trainig set performance
y_train_pred = model.predict(X_train)
y_train_pred = (y_train_pred>0.5)
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True)

#testing set performance
cm = confusion_matrix(y_test, y_pred)