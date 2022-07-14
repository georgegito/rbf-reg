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

# num_of_samples = 6
# np.random.seed(0)
# x = np.random.randint(low=0, high=100, size=(num_of_samples, 2))
# hidden_size = 3

# rbf_layer_ = rbf_layer.RBF(hidden_size)
# rbf_layer_.compute_params(x)

# if (test_params(rbf_layer_.d_max, rbf_layer_.sigma, 0.00001)):
#     print("Test passed")
# else:
#     print("Test failed")

# ---------------------------------------------------------------------------- #
#                             test rbf layer output                            #
# ---------------------------------------------------------------------------- #

num_of_samples = 6
np.random.seed(0)
x = np.random.rand(num_of_samples, 2)
hidden_size = 3

rbf_layer_ = rbf_layer.RBF(hidden_size)
rbf_layer_.compute_params(x)

print(rbf_layer_.centers)

y = rbf_layer_(x)

# print(y)

# y = rbf_layer_(x)

# print(y)
