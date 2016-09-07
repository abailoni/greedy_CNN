# import os
# os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
from copy import deepcopy

from lasagne.updates import adam
from greedyNET.utils_fcts import join_dict


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



# ------ # ------ # ------- # ------- #
#        MAIN GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.main_net import greedyRoutine

eval_size = 0.1
nolearn_kwargs = {
    'update': adam,
    'update_learning_rate': 1e-2,
    'update_beta1': 0.9,
    'max_epochs': 5,
    'verbose': 1,
}
num_VGG16_layers = 2

greedy_routine = greedyRoutine(
    num_VGG16_layers,
    eval_size=eval_size,
    **nolearn_kwargs
)

# -----------------------------------------
# Nets params:
# -----------------------------------------
logRegr_params = {
    'filter_size': 11,
    # 'imgShape': X_small.shape[-2:],
    # 'xy_input': X_small.shape[-2:],
    # 'channels_image': 3,
    'batch_size': 100,
    'eval_size': eval_size
}
net2_params = {
    'filter_size': 11,
    'batch_size': 100,
    'batchShuffle': True,
    'num_nodes': 5
}

# --------------------------
# Nets fitting routines:
# --------------------------
def fit_naiveRoutine_logRegr(net):
    net.fit(X_small, y_small,epochs=10)
def fit_naiveRoutine_net2(net):
    net.fit(X_small, y_small,epochs=10)

# --------------------------
# Train one layer:
# --------------------------
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,5,join_dict(nolearn_kwargs, logRegr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, net2_params))
)


# -------------------
# Set main network:
# -------------------



# -------------------------------------
# Set the main logRegr network:
# -------------------------------------


my_first_net = logRegr.Boost_LogRegr(
    greedyNET,
    **join_dict(nolearn_kwargs, logRegr_params)
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

my_first_net.net.fit(X_small, y_small,epochs=2)
# my_first_net.net.update_learning_rate = 1e-3
# my_first_net.net.fit(X_small, y_small,epochs=2)
# my_first_net.net.update_learning_rate = 1e-4
# my_first_net.net.fit(X_small, y_small,epochs=10)


from mod_nolearn.visualize import plot_loss
plot = plot_loss(my_first_net.net, "loss_boostLogRegr.pdf")

# import lasagne.layers as layers
# print layers.get_all_param_values(my_first_net.net.layers_['convLayer'])[0]
second_network = my_first_net.clone(reset=True, setClassifier=True)
second_network.net.update_learning_rate = 1e-2
print "Training second Log"
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
    **join_dict(nolearn_kwargs, net2_params)
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
my_first_net2.net.fit(X_small, y_small, epochs=1)

print my_first_net2.active_nodes

my_first_net2.insert_weights(second_network)
my_first_net2.insert_weights(second_network)
my_first_net2.net.fit(X_small, y_small, epochs=2)

greedyNET.insert_new_layer(my_first_net2)

my_first_net_2 = logRegr.Boost_LogRegr(
    greedyNET,
    **join_dict(nolearn_kwargs, logRegr_params)
)

my_first_net_2.net.fit(X_small, y_small,epochs=2)

