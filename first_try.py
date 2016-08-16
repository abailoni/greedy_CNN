import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high"

# import matplotlib.pyplot as plt

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()


used_data = 1000

X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]


# # Don't forget to use right data types!
# # To convert the inputs (they are float64 and int64) use
# X, y = X.astype(np.float32), y.astype(np.int32)
# print y.dtype


# -------------------------------------
# Design the network with nolearn
# -------------------------------------
# from lasagne import layers as lasLayers
from lasagne.updates import adam
# from lasagne.nonlinearities import sigmoid
# from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

# import sys
# sys.path.insert(0, '/Users/alberto-mac/sshfs_vol/rep/greedy_CNN/')
from greedyNET.nets.logRegres import LogRegr

logRegr_params = {
    'imgShape': X.shape[-2:],
    'channels_input': 3,
    'batch_size' = 100,
    'xy_input': X.shape[-2:]
}

my_first_net = LogRegr(
    update=adam,
    update_learning_rate=0.01,
    update_beta1=0.9,
    train_split=TrainSplit(eval_size=0),
    regression=False,
    max_epochs=5,
    verbose=1,
    **logRegr_params
)

# quit()


# -------------------------------------
# Visualize initial informations:
# -------------------------------------

# # Print network in a pdf:
# from nolearn.lasagne.visualize import draw_to_file
# draw_to_file(my_first_net.net,"prova_net.pdf")

# # Really useful information about the health status of the net:
# from nolearn.lasagne import PrintLayerInfo
# layer_info = PrintLayerInfo()
# layer_info(my_first_net.net)


# -------------------------------------
# Testing the training procedure:
# -------------------------------------
my_first_net.net.fit(X, y_mod)


# -------------------------------------
# Visualize some useful stuff:
# -------------------------------------
# from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
# from nolearn.lasagne.visualize import plot_conv_activity
# from nolearn.lasagne.visualize import plot_occlusion
# from nolearn.lasagne.visualize import plot_saliency

from mod_nolearn.visualize import plot_loss
plot = plot_loss(my_first_net.net, "loss_first_try.pdf")

plot = plot_conv_weights(my_first_net.net.layers_["convLayer"])
plot.savefig("weights.pdf")
