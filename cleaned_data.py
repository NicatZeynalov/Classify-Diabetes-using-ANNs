#import libraries

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#import dataset
diabets_df = pd.read_csv('diabets.csv')
#diabets_df.head()
#diabets_df.tail()

X = diabets_df.iloc[:, 0:8].values
y = diabets_df.iloc[:, 8].values

#StandartScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Split into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#print(X_train.shape[1])

