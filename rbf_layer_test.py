import numpy as np
import rbf_layer


def test_params(d_max, sigma, err):
    if abs(d_max - 68.1778719396713) < err and abs(sigma - 27.83349966683499) < err:
        return True
    else:
        return False

# ---------------------------------------------------------------------------- #
#                           test param initialization                          #
# ---------------------------------------------------------------------------- #

# import utils
# from tensorflow import keras

# num_of_samples = 6
# np.random.seed(0)
# x = np.random.randint(low=0, high=100, size=(num_of_samples, 2))
# hidden_size = 3

# rbf_layer_ = rbf_layer.RBF(hidden_size, utils.InitCentersKMeans(x))
# # rbf_layer_.compute_params(x)
# model = keras.Sequential()
# model.add(keras.Input(shape=(2,)))
# model.add(rbf_layer_)

# if (test_params(rbf_layer_.d_max, rbf_layer_.sigma, 0.00001)):
#     print("Test passed")
# else:
#     print("Test failed")

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
print(rbf_layer_.centers)

centers_assert = tf.constant([[0.4375872, 0.891773 ],
                              [0.8776939, 0.4561682],
                              [0.5250772, 0.6353222]])

if (np.all(centers_assert == rbf_layer_.centers)): 
    print("Centers test: OK")
else:
    print("Centers test: FAILED")

y = model(x)

y_assert = tf.constant([[64.9723,   56.9549,   67.395065],
                        [58.551006, 62.43068,  66.90674 ],
                        [63.870872, 53.269638, 67.162544],
                        [67.86456,  46.25033,  63.060368],
                        [39.739975, 67.00948,  52.547253],
                        [52.479355, 67.00948,  62.494972]])

if (np.all(y_assert == y)): 
    print("Output test: OK")
else:
    print("Output test: FAILED")

print(y_assert, y)