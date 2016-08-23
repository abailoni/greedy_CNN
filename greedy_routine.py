import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high"

from lasagne.updates import adam
from greedyNET.utils_fcts import join_dict


# import matplotlib.pyplot as plt

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()

used_data = -1
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = -1
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
    'batch_size': 50,
    'eval_size': eval_size
}
net2_params = {
    'filter_size': 11,
    'batch_size': 50,
    'batchShuffle': True,
}

# --------------------------
# Nets fitting routines:
# --------------------------
def fit_naiveRoutine_logRegr(net):
    net.fit(X_small, y_small,epochs=5)
    return net
def fit_naiveRoutine_net2(net):
    net.fit(X_small, y_small,epochs=5)
    return net

# --------------------------
# Train one layer:
# --------------------------
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,3,join_dict(nolearn_kwargs, logRegr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, net2_params))
)
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,3,join_dict(nolearn_kwargs, logRegr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, net2_params))
)

from nolearn.lasagne.visualize import draw_to_file
draw_to_file(greedy_routine.net,"FIRST_GREEDY_NET.pdf")



