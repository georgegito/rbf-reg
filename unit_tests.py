import numpy as np
from lib import utils
from lib import rbf_layer

# ---------------------------------------------------------------------------- #
#                           test param initialization                          #
# ---------------------------------------------------------------------------- #

from tensorflow import keras

num_of_samples = 6
np.random.seed(0)
x = np.random.randint(low=0, high=100, size=(num_of_samples, 2))
hidden_size = 3

rbf_layer_ = rbf_layer.RBF(hidden_size, utils.InitCentersKMeans(x))
model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(rbf_layer_)

if abs(rbf_layer_.sigma - 27.83349966683499) < 0.00001:
    print("Sigma test: OK")
else:
    print("Sigma test: FAILED")

# ---------------------------------------------------------------------------- #
#                             test rbf layer output                            #
# ---------------------------------------------------------------------------- #

import utils
import tensorflow as tf
from tensorflow import keras
from keras import layers

num_of_samples = 6
np.random.seed(0)
x = np.random.rand(num_of_samples, 2)
hidden_size = 3

rbf_layer_ = rbf_layer.RBF(hidden_size, utils.InitCentersKMeans(x))
model = keras.Sequential()
model.add(keras.Input(shape=(2,)))
model.add(rbf_layer_)
# print(rbf_layer_.centers)

centers_assert = tf.constant([[0.4375872, 0.891773 ],
                              [0.8776939, 0.4561682],
                              [0.5250772, 0.6353222]])

if (np.all(centers_assert == rbf_layer_.centers)): 
    print("Centers test: OK")
else:
    print("Centers test: FAILED")

y = model(x)
# print(y)

y_assert = tf.constant([[7.4903636, 6.566074 , 7.769674 ],
                        [6.7500825, 7.1973524, 7.7133765],
                        [7.3633857, 6.1412168, 7.7428675],
                        [7.8238,    5.3319926, 7.269946 ],
                        [4.581443 , 7.725222,  6.0579367],
                        [6.0501084, 7.725222,  7.2047644]])

if (np.all(y_assert == y)): 
    print("Output test: OK")
else:
    print("Output test: FAILED")
