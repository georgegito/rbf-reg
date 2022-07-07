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

def RBF_kernel(x, centers, sigma):
  return [math.exp(-np.linalg.norm(x - c)**2) / (2 * sigma**2) for c in centers]

def RBF_layer_output(x, centers, sigma):
  return [RBF_kernel(x_, centers, sigma) for x_ in x]

class RBF(keras.layers.Layer):
  def __init__(self, units):
    super(RBF, self).__init__()
    self.units = units

  def build(self, input_shape):
    # w and b will be removed
    self.w = self.add_weight(
      shape=(input_shape[-1], self.units), initializer="ones", trainable=False
    )
    self.b = self.add_weight(shape=(self.units,), initializer="zeros", trainable=False) 

  def call(self, inputs):
    # return [0 for i in range(37)]
    return RBF_layer_output(inputs, self.centers, self.sigma)
    # return tf.matmul(float(inputs), float(self.w)) + float(self.b) # TODO rbf
    # return [0, 0, 0]

  def compute_params(self, inputs):
    kmeans = KMeans(n_clusters=self.units, random_state=0, copy_x=True).fit(inputs)
    self.centers = kmeans.cluster_centers_
    self.d_max = utils.ComputeMaxDistance(self.centers, self.centers)
    self.sigma = self.d_max / (math.sqrt(2 * self.centers.shape[0]))