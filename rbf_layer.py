import math
from sklearn.cluster import KMeans
import tensorflow as tf 
from tensorflow import keras
from scipy.spatial import distance
import utils


class RBF(keras.layers.Layer):

  def __init__(self, num_units):
    self.num_units = num_units
    super(RBF, self).__init__()

  def build(self, input_shape):
    self.centers = self.add_weight(shape=(input_shape[1], self.num_units), initializer=tf.initializers.RandomNormal, trainable=False) # TODO use custom initializer
    self.sigma = self.add_weight(shape=(1, 1), initializer=tf.initializers.RandomNormal, trainable=False) # TODO use custom initializer
    super(RBF, self).build(input_shape)

  def call(self, inputs):
    return utils.RBF_kernel(inputs, self.centers, self.sigma)

  def get_config(self):
    base_config = super(RBF, self).get.config
    base_config['num_units'] = self.num_units
    return base_config

  def compute_params(self, inputs):
    kmeans = KMeans(n_clusters=self.num_units, random_state=0, copy_x=True).fit(inputs)
    self.centers = kmeans.cluster_centers_
    d_max = utils.ComputeMaxDistance(self.centers, self.centers)
    self.sigma = d_max / (math.sqrt(2 * self.centers.shape[0]))