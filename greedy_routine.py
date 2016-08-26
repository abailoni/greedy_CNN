import os
os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
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

used_data = 500
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = -1
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]

# # Plot some images:
# from mod_nolearn.visualize import plot_images
# fig = plot_images(X[:4])
# fig.savefig("images_or.pdf")

# raise Warning("Stop here!")

# ------ # ------ # ------- # ------- #
#        MAIN GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.main_net import greedyRoutine

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

greedy_routine = greedyRoutine(
    num_VGG16_layers,
    eval_size=eval_size,
    model_name='first_regr_'+now.strftime("%m-%d_%H-%M"),
    **join_dict(nolearn_kwargs, greedy_kwargs)
)

# -----------------------------------------
# Nets params:
# -----------------------------------------
lrn_rate = [0.00005, 0.001]
logRegr_params = {
    'max_epochs': 17,
    'update': adam,
    'numIter_subLog': 10,
    # 'subLog_filename': 'prova_subLog.txt',
    'livePlot': True,
    'filter_size': 11,
    'update_learning_rate': theano.shared(float32(lrn_rate[0])),
    'update_beta1': theano.shared(float32(0.9)),
    'on_epoch_finished': [
        AdjustVariable('update_learning_rate', start=lrn_rate[0], mode='linear', decay_rate=lrn_rate[1]),
        # AdjustVariable('update_momentum', start=0.9, mode='linear', stop=0.9),
        ],
    'batch_size': 20,
    'verbose': 1,
    'eval_size': eval_size,
    # Check weights:
    'trackWeights_freq': 30,
    'trackWeights_layerName': 'convLayer',
    'trackWeights_pdfName': 'logs/weights.txt'
}
net2_params = {
    'update': adam,
    'update_learning_rate': theano.shared(float32(0.01)),
    'update_beta1': theano.shared(float32(0.9)),
    # 'on_epoch_finished': [
    #     AdjustVariable('update_learning_rate', start=0.03, stop=0.00001, mode='log'),
    #     AdjustVariable('update_beta1', start=0.9, stop=0.999, mode='linear'),
    #     ],
    'filter_size': 11,
    'batch_size': 40,
    'batchShuffle': True,
    'max_epochs': 5
}


# --------------------------
# Nets fitting routines:
# --------------------------
def fit_naiveRoutine_logRegr(net):
    net.fit(X_small, y_small)
    return net
def fit_naiveRoutine_net2(net):
    net.fit(X_small, y_small,epochs=5)
    return net

# --------------------------
# Train one layer:
# --------------------------
greedy_routine.train_new_layer(
    (fit_naiveRoutine_logRegr,2,join_dict(nolearn_kwargs, logRegr_params)),
    (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, net2_params))
)


# greedy_routine.train_new_layer(
#     (fit_naiveRoutine_logRegr,3,join_dict(nolearn_kwargs, logRegr_params)),
#     (fit_naiveRoutine_net2,1,join_dict(nolearn_kwargs, net2_params))
# )

from nolearn.lasagne.visualize import draw_to_file
draw_to_file(greedy_routine.net,"FIRST_GREEDY_NET.pdf")



