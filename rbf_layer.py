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
    self.centers = self.add_weight(shape=(self.num_units, input_shape[1]), initializer=self.centers_initializer, trainable=False) # TODO use custom initializer
    # print(self.centers)
    # self.sigma = self.add_weight(shape=(1, 1), initializer=self.sigma_initializer, trainable=False) # TODO use custom initializer
    # self.sigma = self.add_weight(shape=(1, 1), initializer=tf.initializers.RandomNormal, trainable=False)
    self.sigma = utils.InitSigma(self.centers)
    super(RBF, self).build(input_shape)

  def call(self, inputs):
    return utils.RBF_kernel(inputs, self.centers, self.sigma)

  # def compute_params(self, inputs):
  #   # TODO this does not update self.centers - Need to use custom initializer
  #   kmeans = KMeans(n_clusters=self.num_units, random_state=0, copy_x=True).fit(inputs)
  #   print(kmeans.cluster_centers_)
  #   self.centers = tf.constant(kmeans.cluster_centers_)
  #   self.d_max = utils.ComputeMaxDistance(self.centers, self.centers) # stored as member variable for testing purposes
  #   self.sigma = self.d_max / (math.sqrt(2 * self.centers.shape[0]))