import numpy as np
from scipy.spatial import distance

def ComputeMaxDistance(X1, X2):
    
    max_dist = 0
    for x1 in X1:
        for x2 in X2:
            dist = distance.euclidean(x1, x2)
            max_dist = max(dist, max_dist)
    
    return max_dist