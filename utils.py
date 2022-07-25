from scipy.spatial import distance
import tensorflow as tf
import numpy as np

def ComputeMaxDistance(X1, X2):
  max_dist = 0
  for x1 in X1:
    for x2 in X2:
      dist = distance.euclidean(x1, x2)
      max_dist = max(dist, max_dist)
  return max_dist

def RBF_kernel(inputs, centers, sigma):
  # print(centers)
  X_ = tf.expand_dims(inputs, axis=1)
  C_ = tf.expand_dims(centers, axis=0)
  D_ = X_ - C_
  # return tf.exp(-tf.norm(D_, axis=2)**2) / (2 * sigma**2)
  return tf.exp(-tf.norm(D_, axis=2)**2) / (2 * 1**2) # TODO change 

from sklearn.cluster import KMeans

class InitCentersKMeans(tf.keras.initializers.Initializer):

    def __init__(self, inputs):
      self.inputs = inputs

    def __call__(self, shape, dtype=None):
      # print(shape[0], shape[1], self.inputs.shape[0], self.inputs.shape[1], self.inputs.shape[2])
      assert shape[1] == self.inputs.shape[1]
      num_units = shape[0]
      kmeans = KMeans(n_clusters=num_units, random_state=0, copy_x=True).fit(self.inputs)
      # centers = tf.constant(kmeans.cluster_centers_)
      # d_max = utils.ComputeMaxDistance(self.centers, self.centers) # stored as member variable for testing purposes
      # sigma = self.d_max / (math.sqrt(2 * self.centers.shape[0]))
      # return tf.random.normal(
        # shape, mean=self.mean, stddev=self.stddev, dtype=dtype)
      return tf.cast(kmeans.cluster_centers_, dtype="float32")