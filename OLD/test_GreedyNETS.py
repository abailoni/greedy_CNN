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

# -------------------------------------
# Import cityscape:
# -------------------------------------
from greedyNET.data_utils import get_cityscapes_data

# Load the (preprocessed) CIFAR10 data.
data_X, data_y, data_y_mod = get_cityscapes_data()

used_data = 2000
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
    'max_epochs': 15,
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
    'batch_size': 40,
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
def fit_naiveRoutine_convSoft_finetune(net):
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
        # Init net:
        # ----------------------
        greedy_routine = restore_greedyModel('model_725251', 'tuning/tune_first_regr2/')
        greedy_routine.update_all_paths(model_name, self.path_out)
        convSoftmax_name = "cnv_L0_G0"
        regr_name = "regr_L0G0N1"



        # ----------------------
        # Set values:
        # ----------------------
        lrn_rate, decay_rate = param_values['lrn_rate'], param_values['decay_rate']
        greedy_routine.convSoftmax[convSoftmax_name].net.update_learning_rate.set_value(float32(lrn_rate))
        new_objects = [
            AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
            AdjustVariable('update_learning_rate', start=lrn_rate, mode='log', decay_rate=decay_rate)]
        greedy_routine.convSoftmax[convSoftmax_name].net.update_AdjustObjects(new_objects)


        # ----------------------
        # Fit net:
        # ----------------------
        greedy_routine.convSoftmax[convSoftmax_name].insert_weights(greedy_routine.regr[regr_name])
        # Test modification weights:
        # from lasagne.layers import get_all_param_values
        # main_before = get_all_param_values(greedy_routine.convSoftmax[convSoftmax_name].net.layers_['conv2'])
        # new_before = get_all_param_values(greedy_routine.convSoftmax[convSoftmax_name].net.layers_['conv2_newNode'])
        greedy_routine.train_convSoftmax(convSoftmax_name, 0, fit_naiveRoutine_convSoft, fit_naiveRoutine_convSoft_finetune, 1)
        # main_after = get_all_param_values(greedy_routine.convSoftmax[convSoftmax_name].net.layers_['conv2'])
        # new_after = get_all_param_values(greedy_routine.convSoftmax[convSoftmax_name].net.layers_['conv2_newNode'])

        # print "W_main 1:"
        # print main_before[0]-main_after[0], main_before[0]
        # print "W_main 2:"
        # print main_before[3]-main_after[3], main_before[3]
        # print "W_new 1:"
        # print new_before[0]-new_after[0], new_before[0]
        # print "W_new 2:"
        # print new_before[2]-new_after[2], new_before[2]

        # ----------------------
        # Collect results:
        # ----------------------
        val_loss = np.array([greedy_routine.convSoftmax[convSoftmax_name].net.train_history_[i]['train_loss'] for i in range(len(greedy_routine.convSoftmax[convSoftmax_name].net.train_history_))])
        best = np.argmin(val_loss)
        results = {
            'train_loss': greedy_routine.convSoftmax[convSoftmax_name].net.train_history_[best]['train_loss'],
            'valid_loss': greedy_routine.convSoftmax[convSoftmax_name].net.train_history_[best]['valid_loss'],
        }
        return results



first = tune_lrn_rate(
    (
        ('lrn_rate', 0.00901404, None, None, np.float32),
        ('decay_rate', 1.62766e-02, None, 'log', np.float32)
    ),
    ['train_loss', 'valid_loss'],
    num_iterations = 1,
    name = "test_convSoft2",
    folder_out = 'test',
    plot=False
)

# FIRST ONE, OLE!
first()


