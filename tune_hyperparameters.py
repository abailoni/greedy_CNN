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

used_data = 100
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = 10
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]


# ------ # ------ # ------- # ------- #
#        MAIN GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.greedyNet import greedyRoutine

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

# greedy_routine = greedyRoutine(
#     num_VGG16_layers,
#     eval_size=eval_size,
#     # model_name='first_regr_'+now.strftime("%m-%d_%H-%M"),
#     model_name="NET_2.0_ole",
#     **join_dict(nolearn_kwargs, greedy_kwargs)
# )


# -----------------------------------------
# SubNets params:
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
def fit_naiveRoutine_convSoft(net):
    net.fit(X_small, y_small, epochs=10)
    return net


# ------ # ------ # ------- # ------- #
#        TUNE HYPERPARAMETER:         #
# ------ # ------ # ------- # ------- #

import numpy as np
from mod_nolearn.tuneHyper import tune_hyperparams

class tune_lrn_rate(tune_hyperparams):
    def fit_model(self, param_values):
        # ----------------------
        # Set values:
        # ----------------------
        lrn_rate, init_weights = param_values['lrn_rate'], param_values['init_weights']
        convSoftmax_params['update_learning_rate'] = theano.shared(float32(lrn_rate))

        # ----------------------
        # Init and train net:
        # ----------------------
        greedy_routine = greedyRoutine(
            num_VGG16_layers,
            eval_size=eval_size,
            model_name="tuning_%.3f_%.3f" %(lrn_rate,init_weights),
            **join_dict(nolearn_kwargs, greedy_kwargs)
        )
        net_name = "cnv_L0_G0"
        greedy_routine.init_convSoftmax(net_name, convSoftmax_params, 3)
        greedy_routine.train_convSoftmax(net_name, fit_naiveRoutine_convSoft, None, 0, 0)

        # ----------------------
        # Collect results:
        # ----------------------
        # WRONG HERE, NOT TAKING THE BEST...
        results = {
            'val_loss': greedy_routine.convSoftmax[net_name].net.train_history_[-1]['train_loss'],
            'val_acc': greedy_routine.convSoftmax[net_name].net.train_history_[-1]['val pixelAcc'],
        }
        return results



first = tune_lrn_rate(
    (('lrn_rate', 0.01, 0.00001, 'log', np.float32),
    ('init_weights', 0.01, 0.0001, 'linear', np.float32)),
    num_iterations = 3,
    name = 'FIRST',
    path_outputs = 'first_tuning/'
)

# FIRST ONE, OLE!
first()


