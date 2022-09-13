import math
from sklearn.cluster import KMeans
import tensorflow as tf 
from tensorflow import keras
from scipy.spatial import distance
import utils

class RBF(keras.layers.Layer):

  def __init__(self, num_units, centers_initializer):
    self.num_units = num_units
    self.centers_initializer = centers_initializer
    super(RBF, self).__init__()

  def build(self, input_shape):
    self.centers = self.add_weight(name="centers", shape=(self.num_units, input_shape[1]), initializer=self.centers_initializer, trainable=False)
    self.sigma = utils.init_sigma(self.centers)
    super(RBF, self).build(input_shape)

  def call(self, inputs):
    return utils.RBF_kernel(inputs, self.centers, self.sigma)