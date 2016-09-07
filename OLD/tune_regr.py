# import os
# os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
import datetime
import numpy as np

from copy import copy

import theano
from lasagne.updates import adam, nesterov_momentum
from greedyNET.utils_fcts import join_dict
from mod_nolearn.nets.modNeuralNet import AdjustVariable
from mod_nolearn.utils import float32
# import matplotlib.pyplot as plt
from mod_nolearn.segmentFcts import mean_IoU


# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()

used_data = 1000
X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

# Mini-images, just for test:
CROP = -1
X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]


# ------ # ------ # ------- # ------- #
#        MAIN GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.greedyNet import restore_greedyModel

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
# lrn_rate = [0.00005, 0.001]
# convSoftmax_params = {
#     'num_filters1': 5,
#     'filter_size1': 11,
#     'filter_size2': 11,
#     'eval_size': eval_size,

#     'max_epochs': 16,
#     'update': adam,
#     'update_learning_rate': theano.shared(float32(lrn_rate[0])),
#     # 'weights_HeGain': np.sqrt(2),
#     'numIter_subLog': 5,
#     'update_beta1': theano.shared(float32(0.9)),
#     'on_epoch_finished': [
#         AdjustVariable('update_learning_rate', start=lrn_rate[0], mode='linear', decay_rate=lrn_rate[1]),
#         # AdjustVariable('update_momentum', start=0.9, mode='linear', stop=0.9),
#         ],
#     'batch_size': 40,
#     # 'trackWeights_freq': 30,
#     'trackWeights_layerName': 'conv2_newNode',
#     'batchShuffle': True,
#     'verbose': 1
# }


lrn_rate_rgr = [0.0001, 0.2]
regr_params = {
    'max_epochs': 10,
    'update': adam,
    'numIter_subLog': 2,
    # 'subLog_filename': 'prova_subLog.txt',
    # 'livePlot': True,

    'update_beta1': theano.shared(float32(0.9)),
    # 'update_learning_rate': theano.shared(float32(lrn_rate_rgr[0])),
    # 'on_epoch_finished': [
    #     AdjustVariable('update_learning_rate', start=lrn_rate_rgr[0], mode='log', decay_rate=lrn_rate_rgr[1]),
    #     # AdjustVariable('update_momentum', start=0.9, mode='linear', stop=0.9),
    #     ],
    'batch_size': 20,
    'verbose': 1,
    'batchShuffle': True,
    # Check weights:
    # 'trackWeights_freq': 30,
    # 'trackWeights_layerName': 'conv1',
}


# --------------------------
# Nets fitting routines:
# --------------------------
def fit_naiveRoutine_convSoft(net):
    net.fit(X_small, y_small)
    return net
def fit_naiveRoutine_regr(net):
    net.fit(X_small, y_small)
    return net

# ------ # ------ # ------- # ------- #
#        TUNE HYPERPARAMETER:         #
# ------ # ------ # ------- # ------- #

from mod_nolearn.tuneHyper import tune_hyperparams

class tune_lrn_rate(tune_hyperparams):
    def fit_model(self, param_values, model_name):
        # ----------------------
        # Set values:
        # ----------------------
        lrn_rate, L2 = param_values['lrn_rate'], param_values['L2']
        regr_params['L2'] = float(L2)
        regr_params['update_learning_rate'] = theano.shared(float32(lrn_rate))
        # REALLY BAD , TEMP:
        regr_params['on_epoch_finished'] = [
            AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
            AdjustVariable('update_learning_rate', start=lrn_rate, mode='log', decay_rate=1.62766e-02)]

        # ----------------------
        # Init and train net:
        # ----------------------
        greedy_routine = restore_greedyModel('model_553589', 'tuning/node1_DEF/')
        greedy_routine.update_all_paths(model_name, self.path_out)

        convSoftmax_name = "cnv_L0_G0"
        regr_name = "regr_L0G0N1"

        greedy_routine.init_regr(regr_name, regr_params, convSoftmax_name)
        greedy_routine.train_regr(regr_name, fit_naiveRoutine_regr, 0, 1)
        # ----------------------
        # Collect results:
        # ----------------------
        val_loss = np.array([greedy_routine.regr[regr_name].net.train_history_[i]['valid_loss'] for i in range(len(greedy_routine.regr[regr_name].net.train_history_))])
        best = np.argmin(val_loss)
        results = {
            'train_loss': greedy_routine.regr[regr_name].net.train_history_[best]['train_loss'],
            'valid_loss': greedy_routine.regr[regr_name].net.train_history_[best]['valid_loss'],
        }
        return results



first = tune_lrn_rate(
    (
        ('lrn_rate',  1e-6, 8e-1, 'log', np.float32),
        ('L2', 1e-6, 9e-1, 'log', np.float32)
    ),
    ['train_loss', 'valid_loss' ],
    num_iterations = 40,
    name = "regr1_DEF",
    folder_out = 'tuning',
    plot=False
)

# FIRST ONE, OLE!
first()


