from scipy.spatial import distance
import tensorflow as tf
from sklearn.cluster import KMeans
import math
import numpy as np
from keras import backend as K

def compute_max_distance(X1, X2):
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

def init_sigma(centers):
  d_max = compute_max_distance(centers, centers)
  sigma = d_max / (math.sqrt(2 * centers.shape[0]))
  return sigma

def data_summary(arr):
  shape = np.shape(arr)
  min = np.amin(arr)
  max = np.amax(arr)
  range = np.ptp(arr)
  variance = np.var(arr)
  sd = np.std(arr)
  print("Shape =", shape)
  print("Minimum =", min)
  print("Maximum =", max)
  print("Range =", range)
  print("Variance =", variance)
  print("Standard Deviation =", sd)
  print()

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))