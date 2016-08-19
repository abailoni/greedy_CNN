import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high,device=gpu0"

# import matplotlib.pyplot as plt

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()

used_data = 1000
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = 100
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]


# -------------------------------------
# Set the main logRegr network:
# -------------------------------------
from lasagne.updates import adam
from greedyNET.nets.logRegres import LogRegr

logRegr_params = {
    'filter_size': 11,
    'imgShape': X_small.shape[-2:],
    'xy_input': X_small.shape[-2:],
    'channels_image': 3,
    'DCT_size': None,
    'batch_size': 100,
    'eval_size': 0.2
}

my_first_net = LogRegr(
    update=adam,
    update_learning_rate=1e-2,
    update_beta1=0.9,
    regression=False,
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
from mod_nolearn.visualize import plot_conv_weights, plot_images
fig = plot_conv_weights(my_first_net.net.layers_["convLayer"])
fig.savefig("weights_before_boost1.pdf")


my_first_net.net.fit(X_small, y_mod_small,epochs=4)
my_first_net.net.update_learning_rate = 1e-3
my_first_net.net.fit(X_small, y_mod_small,epochs=2)
# my_first_net.net.update_learning_rate = 1e-4
# my_first_net.net.fit(X_small, y_mod_small,epochs=10)

fig = plot_conv_weights(my_first_net.net.layers_["convLayer"])
fig.savefig("weights_after_boost1.pdf")

from mod_nolearn.visualize import plot_loss
plot = plot_loss(my_first_net.net, "loss_boostLogRegr.pdf")

second_network = my_first_net.clone(reset=True, setClassifier=True)
second_network.net.update_learning_rate = 1e-2
print "Training second Log"
second_network.net.fit(X_small, y_mod_small,epochs=4)




# # -------------------------------------
# # Visualize some useful stuff:
# # -------------------------------------
# # from nolearn.lasagne.visualize import plot_loss
# # from nolearn.lasagne.visualize import plot_conv_weights
# # from nolearn.lasagne.visualize import plot_conv_activity
# # from nolearn.lasagne.visualize import plot_occlusion
# # from nolearn.lasagne.visualize import plot_saliency



# fig = plot_images(X[:4])
# fig.savefig("images_or.pdf")
# fig = plot_images(X_small)
# fig.savefig("images.pdf")
# fig = plot_conv_weights(my_first_net.net.layers_["convLayer"])
# fig.savefig("weights.pdf")
