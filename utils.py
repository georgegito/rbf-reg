from scipy.spatial import distance
import tensorflow as tf

def ComputeMaxDistance(X1, X2):
    max_dist = 0
    for x1 in X1:
        for x2 in X2:
            dist = distance.euclidean(x1, x2)
            max_dist = max(dist, max_dist)
    return max_dist

def RBF_kernel(inputs, centers, sigma):
  X_ = tf.expand_dims(inputs, axis=1)
  C_ = tf.expand_dims(tf.transpose(centers), axis=0)
  D_ = X_ - C_
  return tf.exp(-tf.norm(D_, axis=2)**2) / (2 * sigma**2) 