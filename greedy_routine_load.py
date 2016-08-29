import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high, device=cpu"
import datetime


import theano
from lasagne.updates import adam, nesterov_momentum
from greedyNET.utils_fcts import join_dict
from mod_nolearn.nets.modNeuralNet import AdjustVariable
from mod_nolearn.utils import float32
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
CROP = 10
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]

# # Plot some images:
# from mod_nolearn.visualize import plot_images
# fig = plot_images(X[:4])
# fig.savefig("images_or.pdf")

# raise Warning("Stop here!")

# ------ # ------ # ------- # ------- #
#        LOAD GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.greedyNet import restore_greedyModel

nolearn_kwargs = {
}

eval_size = 0.1
nolearn_kwargs = {
}

greedy_kwargs = {
    'update': adam,
    'update_learning_rate': 0.01,
    'update_beta1': 0.9,
}

num_VGG16_layers = 2

now = datetime.datetime.now()




greedy_routine = restore_greedyModel('NET_2.0_ole')
greedy_routine.update_name('NET_2.0_ole_B')

# -----------------------------------------
# subNets params:
# -----------------------------------------
lrn_rate_rgr = [0.0001, 0.001]
lrn_rate = [0.00005, 0.001]
regr_params = {
    'max_epochs': 3,
    'update': adam,
    'numIter_subLog': 10,
    # 'subLog_filename': 'prova_subLog.txt',
    # 'livePlot': True,
    'num_filters1': 5,
    'filter_size1': 11,
    'filter_size2': 11,
    'update_learning_rate': theano.shared(float32(lrn_rate_rgr[0])),
    'update_beta1': theano.shared(float32(0.9)),
    'on_epoch_finished': [
        AdjustVariable('update_learning_rate', start=lrn_rate_rgr[0], mode='linear', decay_rate=lrn_rate_rgr[1]),
        # AdjustVariable('update_momentum', start=0.9, mode='linear', stop=0.9),
        ],
    'batch_size': 10,
    'verbose': 1,
    'eval_size': eval_size,
    'batchShuffle': True,
    # Check weights:
    'trackWeights_freq': 30,
    'trackWeights_layerName': 'conv1',
}
convSoftmax_params = {
    'max_epochs': 3,
    'update': adam,
    'update_learning_rate': theano.shared(float32(lrn_rate[0])),
    'numIter_subLog': 10,
    'update_beta1': theano.shared(float32(0.9)),
    'on_epoch_finished': [
        AdjustVariable('update_learning_rate', start=lrn_rate[0], mode='linear', decay_rate=lrn_rate[1]),
        # AdjustVariable('update_momentum', start=0.9, mode='linear', stop=0.9),
        ],
    'batch_size': 5,
    # 'trackWeights_freq': 30,
    'trackWeights_layerName': 'conv2',
    'batchShuffle': True,
    'verbose': 1
}


# --------------------------
# Nets fitting routines:
# --------------------------
def fit_naiveRoutine_logRegr(net):
    net.fit(X_small, y_small, epochs=10)
    return net
def fit_naiveRoutine_net2(net):
    net.update_learning_rate.set_value(float32(0.01))
    net.fit(X_small, y_small, epochs=10)
    return net
def finetune_naiveRoutine_net2(net):
    net.update_learning_rate.set_value(float32(0.0001))
    net.fit(X_small, y_small, epochs=10)
    return net

# --------------------------
# Train one layer:
# --------------------------
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,5,join_dict(nolearn_kwargs, regr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, convSoftmax_params)),
    finetune_naiveRoutine_net2
)
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,5,join_dict(nolearn_kwargs, regr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, convSoftmax_params)),
    finetune_naiveRoutine_net2
)
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,5,join_dict(nolearn_kwargs, regr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, convSoftmax_params)),
    finetune_naiveRoutine_net2
)


from nolearn.lasagne.visualize import draw_to_file
draw_to_file(greedy_routine.net,"FIRST_GREEDY_NET.pdf")



