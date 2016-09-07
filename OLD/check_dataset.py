# import os
# os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
import datetime
import numpy as np

# import theano
# from lasagne.updates import adam, nesterov_momentum
# from greedyNET.utils_fcts import join_dict
# from mod_nolearn.nets.modNeuralNet import AdjustVariable
# from mod_nolearn.utils import float32



# import matplotlib.pyplot as plt

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, _ = get_cityscapes_data()

train = 1000
x_train, y_train = data_X[:train], data_y[:train]

valid = 100
y_valid = data_y[train:train+valid]


# Percentuals of cars:
n_pixels = y_train.shape[1]*y_train.shape[2]
active_pixels_train = y_train.sum(axis=(1,2)).astype(np.float32)
active_pixels_valid = y_valid.sum(axis=(1,2)).astype(np.float32)

mean_train = (active_pixels_train/n_pixels).mean()
mean_valid = (active_pixels_valid/n_pixels).mean()

print mean_train, mean_valid


# Check x_train:
print x_train[0,0,:10,:10]

# Plot some images:
from mod_nolearn.visualize import plot_images, plot_GrTruth
fig = plot_images(x_train[:4])
fig.savefig("images_ere.pdf")
fig = plot_GrTruth(y_train[:4])
fig.savefig("images_ere_GrTruth.pdf")
