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
data_X_train, data_y_train, data_X_val, data_y_val = get_cityscapes_data()

# print data_X_train.shape[0], data_y_train.shape[0], data_X_val.shape[0], data_y_val.shape[0]
size_val = data_X_val.shape[0]
used_data = 1000
X_train, y_train = data_X_train[:used_data], data_y_train[:used_data]

# Variable that will cointain both:
X, y = 0, 0

# ------ # ------ # ------- # ------- #
#        MAIN GREEDY ROUTINE:         #
# ------ # ------ # ------- # ------- #
from greedyNET.nets.greedyNet import greedyRoutine, restore_greedyModel

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
lrn_conv = 1.67401e-04
decay_conv = 1.90961e-02
# lrn_regr = 1.61034e-02
# decay_regr = 2.27008e-01


convSoftmax_params = {
    'num_filters1': 5,
    'filter_size1': 3,
    'filter_size2': 3,
    'L2': 1e-6,

    'max_epochs': 40,
    'update': adam,
    'update_learning_rate': theano.shared(float32(lrn_conv)),
    # 'weights_HeGain': np.sqrt(2),

    'numIter_subLog': 2,
    # 'livePlot': True,


    'update_beta1': theano.shared(float32(0.9)),
    'on_epoch_finished': [
        # mean_IoU(X_train[:40], data_X_val[:40], y_train[:40], data_y_val[:40]),
        AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
        AdjustVariable('update_learning_rate', start=lrn_conv, mode='log', decay_rate=decay_conv)
        ],
    'batch_size': 20,
    # 'trackWeights_freq': 30,
    # 'trackWeights_layerName': 'conv2_newNode',
    'batchShuffle': True,
    'verbose': 1
}

# regr_params = {
#     'max_epochs': 5,
#     'update': adam,
#     'numIter_subLog': 2,
#     # 'subLog_filename': 'prova_subLog.txt',
#     # 'livePlot': True,

#     'update_beta1': theano.shared(float32(0.9)),
#     'update_learning_rate': theano.shared(float32(lrn_regr)),
#     'on_epoch_finished': [
#         AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
#         AdjustVariable('update_learning_rate', start=lrn_regr, mode='log', decay_rate=decay_regr)
#     ],
#     'batch_size': 20,
#     'verbose': 1,
#     'batchShuffle': True,
#     # Check weights:
#     # 'trackWeights_freq': 30,
#     # 'trackWeights_layerName': 'conv1',
# }


# --------------------------
# Nets fitting routines:
# --------------------------
eval_size = 0.3
X_data = (X_train[:420], data_X_val[:180])
y_data = (y_train[:420], data_y_val[:180])
X = np.concatenate(X_data)
y = np.concatenate(y_data)

def fit_naiveRoutine_convSoft(net):
    net.fit(X, y, epochs=8)
    return net
def fit_naiveRoutine_convSoft_finetune(net):
    # net.update_learning_rate.set_value(float32(lrn_conv))
    # net.update_AdjustObjects([
    #     AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
    #     AdjustVariable('update_learning_rate', start=lrn_conv, mode='log', decay_rate=decay_conv)
    # ])
    # net.fit(X_small, y_small, epochs =10)
    return net
# def fit_naiveRoutine_regr(net):
#     net.fit(X_small, y_small)
#     return net

# ------ # ------ # ------- # ------- #
#        TUNE HYPERPARAMETER:         #
# ------ # ------ # ------- # ------- #

def adjust_convSoft(convSoft):
    convSoft.net.update_learning_rate.set_value(float32(lrn_conv))
    convSoft.net.update_AdjustObjects([
        AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
        AdjustVariable('update_learning_rate', start=lrn_conv, mode='log', decay_rate=decay_conv)
    ])
    return convSoft


from mod_nolearn.tuneHyper import tune_hyperparams



class tune_lrn_rate(tune_hyperparams):
    def fit_model(self, param_values, model_name):
        # # ----------------------
        # # Set size values:
        # # ----------------------
        # size_train = param_values['size_data']
        # new_size_val = size_val
        # if size_train<size_val:
        #     new_size_val = size_train
        # global X, y
        # prop = new_size_val*1./(size_train+new_size_val)
        # print "Eval size: %f" %prop

        # # if prop==0.5:
        # #     prop = 0.6

        # X = np.concatenate((X_train[:size_train], data_X_val[:new_size_val]))
        # y = np.concatenate((y_train[:size_train], data_y_val[:new_size_val]))
        # print "Total data: %d" %X.shape[0]

        # ----------------------
        # Set values:
        # ----------------------
        lrn_rate, init_weight = param_values['lrn_rate'], param_values['init_weight']
        convSoftmax_params['init_weight'] = float(init_weight)
        convSoftmax_params['update_learning_rate'] = theano.shared(float32(lrn_rate))
        convSoftmax_params['on_epoch_finished'] = [
            AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
            AdjustVariable('update_learning_rate', start=lrn_rate, mode='log', decay_rate=decay_conv)]

        convSoftmax_params['eval_size'] = eval_size

        # ----------------------
        # Init and train net:
        # ----------------------
        greedy_routine = greedyRoutine(
            num_VGG16_layers,
            eval_size=eval_size,
            model_name=model_name,
            BASE_PATH_LOG=self.path_out,
            **join_dict(nolearn_kwargs, greedy_kwargs)
        )
        net_name = "cnv_L0_G0"
        greedy_routine.init_convSoftmax(net_name, convSoftmax_params, 5)

        # # Check ititial performance:
        # pred_train_A, pred_val_A = greedy_routine.convSoftmax[net_name].net.predict_proba(X[:size_train]), greedy_routine.convSoftmax[net_name].net.predict_proba(data_X_val[:30])
        # print pred_train_A[:10,:,0,0]
        # # print "Statistics:"
        # # print np.unique(pred_train_A, return_counts=True)
        # # np.savetxt("basta.txt", np.unique(pred_val_A, return_counts=True)[0])
        # print "Before accuracy:"
        # print pixel_accuracy_np(pred_train_A,y[:size_train]), pixel_accuracy_np(pred_val_A,data_y_val[:30])
        # print "Before IoU:"
        # print compute_mean_IoU_logRegr(pred_train_A,y[:size_train]), compute_mean_IoU_logRegr(pred_val_A,data_y_val[:30])

        greedy_routine.train_convSoftmax(net_name, 0, fit_naiveRoutine_convSoft, fit_naiveRoutine_convSoft_finetune,  0)



        # # HERE WE ARE:
        # print "Statistics:"
        # print np.unique(pred_train_B, return_counts=True), np.unique(pred_val_B, return_counts=True)

        # print "After accuracy:"
        # print pixel_accuracy_np(pred_train_B,y[:size_train]), pixel_accuracy_np(pred_val_B,data_y_val[:200])


        # ----------------------
        # Collect results:
        # ----------------------
        if len(greedy_routine.convSoftmax[net_name].net.train_history_):
            # Compute IoU:
            from mod_nolearn.segmentFcts import  compute_mean_IoU_logRegr
            mini_slice = slice(0,30)
            pred_train_B, pred_val_B = greedy_routine.convSoftmax[net_name].net.predict(X_data[0][mini_slice]), greedy_routine.convSoftmax[net_name].net.predict(X_data[1][mini_slice])
            IoU_tr, IoU_val = compute_mean_IoU_logRegr(pred_train_B,y_data[0][mini_slice]), compute_mean_IoU_logRegr(pred_val_B,y_data[1][mini_slice])


            val_loss = np.array([greedy_routine.convSoftmax[net_name].net.train_history_[i]['valid_loss'] for i in range(len(greedy_routine.convSoftmax[net_name].net.train_history_))])
            best = np.argmin(val_loss)
            results = {
                'train_loss': greedy_routine.convSoftmax[net_name].net.train_history_[best]['train_loss'],
                'valid_loss': greedy_routine.convSoftmax[net_name].net.train_history_[best]['valid_loss'],
                'trn pixelAcc': greedy_routine.convSoftmax[net_name].net.train_history_[best]['trn pixelAcc'],
                'val pixelAcc': greedy_routine.convSoftmax[net_name].net.train_history_[best]['valid_accuracy'],
                'Train IoU': IoU_tr,
                'Valid IoU': IoU_val,
            }
        else:
            results = {
                'train_loss': np.nan,
                'valid_loss': np.nan,
                'trn pixelAcc': np.nan,
                'val pixelAcc': np.nan,
                'Train IoU': np.nan,
                'Valid IoU': np.nan,
            }
        return results

# range(10,100,10)+range(100,1001,100)

first = tune_lrn_rate(
    (
        ('lrn_rate', 1e-4, 1e-3, 'log', np.float32),
        ('init_weight', 1e-4, 1e-2, 'log', np.float32)
    ),
    # ['train_loss', 'valid_loss', 'trn pixelAcc', 'val pixelAcc'],
    ['train_loss', 'valid_loss', 'trn pixelAcc', 'val pixelAcc', 'Train IoU', 'Valid IoU' ],
    num_iterations = 40,
    name = "node1_lrn_init",
    folder_out = 'tuning',
    plot=False
)

first()


