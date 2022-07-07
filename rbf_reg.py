import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import utils
import rbf_layer

# ---------------------------------------------------------------------------- #
#                                   read data                                  #
# ---------------------------------------------------------------------------- #

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
data = "boston.csv"
raw_df = pd.read_csv(data, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# ---------------------------------------------------------------------------- #
#                      split data to training and testing                      #
# ---------------------------------------------------------------------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

# ---------------------------------------------------------------------------- #
#                               set nn attributes                              #
# ---------------------------------------------------------------------------- #

n_train = X_train.shape[0]

# hidden_size_1 = int(0.1 * n_train)
# hidden_size_2 = int(0.5 * n_train)
# hidden_size_3 = int(0.9 * n_train)

hidden_size = int(0.1 * n_train)

rbf_layer_ = rbf_layer.RBF(hidden_size)
rbf_layer_.compute_params(X_train)

model = keras.Sequential()
model.add(Input(shape=(13,)))
model.add(rbf_layer_)
# model.add(layers.Dense(128))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=150, batch_size=10)
print(model.predict(X_test[0:1]).shape)
# model.summary()

# _, accuracy = model.evaluate(X_train, y_train)
# print('Accuracy: %.2f' % (accuracy*100))

