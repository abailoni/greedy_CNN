import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high,device=gpu0"

# import matplotlib.pyplot as plt

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()

used_data = 100
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = 100
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]


# -------------------------------------
# Set the main logRegr network:
# -------------------------------------
from lasagne.updates import adam
import greedyNET.nets.logRegres as logRegr
import greedyNET.nets.net2 as net2

logRegr_params = {
    'filter_size': 11,
    'imgShape': X_small.shape[-2:],
    'xy_input': X_small.shape[-2:],
    'channels_image': 3,
    'DCT_size': None,
    'batch_size': 100,
    'eval_size': 0.2
}

my_first_net = logRegr.Boost_LogRegr(
    update=adam,
    update_learning_rate=1e-2,
    update_beta1=0.9,
    max_epochs=5,
    verbose=1,
    **logRegr_params
)

# -------------------------------------
# Visualize initial informations:
# -------------------------------------
# # Print network-structure in a pdf:
from nolearn.lasagne.visualize import draw_to_file
draw_to_file(my_first_net.net,"prova_net.pdf")


# -------------------------------------
# Testing the training procedure:
# overfitting mini-images
# -------------------------------------
from mod_nolearn.visualize import plot_conv_weights_mod, plot_images
fig = plot_conv_weights_mod(my_first_net.net.layers_["convLayer"])
fig.savefig("weights_before_boost1.pdf")

my_first_net.net.fit(X_small, y_small,epochs=2)
# my_first_net.net.update_learning_rate = 1e-3
# my_first_net.net.fit(X_small, y_small,epochs=2)
# my_first_net.net.update_learning_rate = 1e-4
# my_first_net.net.fit(X_small, y_small,epochs=10)

fig = plot_conv_weights_mod(my_first_net.net.layers_["convLayer"])
fig.savefig("weights_after_boost1.pdf")

from mod_nolearn.visualize import plot_loss
plot = plot_loss(my_first_net.net, "loss_boostLogRegr.pdf")

# import lasagne.layers as layers
# print layers.get_all_param_values(my_first_net.net.layers_['convLayer'])[0]
second_network = my_first_net.clone(reset=True, setClassifier=True)
second_network.net.update_learning_rate = 1e-2
print "Training second Log"
# print layers.get_all_param_values(second_network.net.layers_['convLayer'])[0]


second_network.net.fit(X_small, y_small,epochs=2)

# ----------------------
# INITIALIZE NETWORK 2:
# ----------------------
net2_params = {
    'filter_size': 11,
    'batch_size': 100,
    'batchShuffle': True,
    'num_nodes': 5
}

my_first_net2 = net2.Network2(
    my_first_net,
    update=adam,
    update_learning_rate=1e-2,
    update_beta1=0.9,
    max_epochs=5,
    verbose=2,
    **net2_params
)

my_first_net2.insert_weights(second_network)

# from nolearn.lasagne.visualize import plot_conv_weights
# fig = plot_conv_weights(my_first_net2.net.layers_["conv2"])
# fig.savefig("weights_net2_conv2_before.pdf")
# fig = plot_conv_weights(my_first_net2.net.layers_["conv_fixedRegr"])
# fig.savefig("weights_net2_convFixRegr_before.pdf")

# CON2_PRIMA = my_first_net2.net.layers_["conv2"].W.get_value()

my_first_net2.net.fit(X_small, y_small, epochs=4)
my_first_net2.net.update_learning_rate = 1e-3
my_first_net2.net.fit(X_small, y_small, epochs=4)
my_first_net2.net.update_learning_rate = 1e-4
my_first_net2.net.fit(X_small, y_small, epochs=4)

my_first_net2.insert_weights(second_network)
my_first_net2.net.fit(X_small, y_small, epochs=4)


# print "DOPO: "
# print "Conv2 W:"
# print my_first_net2.net.layers_["conv2"].W.get_value()- CON2_PRIMA


