import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import preprocessing
import tensorflow as tf 
from tensorflow import keras
from keras import Input
from keras import layers
from keras.optimizers import RMSprop
import utils
import rbf_layer

# ---------------------------------------------------------------------------- #
#                                   read data                                  #
# ---------------------------------------------------------------------------- #

data = "http://lib.stat.cmu.edu/datasets/boston"
# data = "boston.csv"
raw_df = pd.read_csv(data, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]

# ---------------------------------------------------------------------------- #
#                                normalize data                                #
# ---------------------------------------------------------------------------- #

preprocessing.scale(X, copy=False)
preprocessing.scale(y, copy=False)

# ---------------------------------------------------------------------------- #
#                      split data to training and testing                      #
# ---------------------------------------------------------------------------- #

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

# ---------------------------------------------------------------------------- #
#                            compute rbf layer size                            #
# ---------------------------------------------------------------------------- #

n_train = X_train.shape[0]
hidden_size = int(0.1 * n_train)

# ---------------------------------------------------------------------------- #
#                                 create model                                 #
# ---------------------------------------------------------------------------- #

rbf_layer_ = rbf_layer.RBF(hidden_size)
rbf_layer_.compute_params(X_train)

model = keras.Sequential()
model.add(Input(shape=(13,)))
model.add(rbf_layer_)
model.add(layers.Dense(8, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# ---------------------------------------------------------------------------- #
#                                  train model                                 #
# ---------------------------------------------------------------------------- #

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, batch_size=10)
model.summary()