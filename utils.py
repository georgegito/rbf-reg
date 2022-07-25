from scipy.spatial import distance
import tensorflow as tf
from sklearn.cluster import KMeans
import math

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
  return tf.exp(-tf.norm(D_, axis=2)**2) / (2 * sigma**2)

class InitCentersKMeans(tf.keras.initializers.Initializer):

    def __init__(self, inputs):
      self.inputs = inputs

    def __call__(self, shape, dtype=None):
      assert shape[1] == self.inputs.shape[1]
      num_units = shape[0]
      kmeans = KMeans(n_clusters=num_units, random_state=0, copy_x=True).fit(self.inputs)
      return tf.cast(kmeans.cluster_centers_, dtype="float32")

def InitSigma(centers):
  d_max = ComputeMaxDistance(centers, centers)
  sigma = d_max / (math.sqrt(2 * centers.shape[0]))
  return sigma