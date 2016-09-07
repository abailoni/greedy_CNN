import logging

from greedyNET.nets.full_net import fullNet
from copy import deepcopy

import mod_nolearn.utils as utils

def main():
    # import os
    # os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
    import datetime
    import numpy as np

    from copy import copy

    import theano1
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

    used_data = 1000
    X, y, y_mod = data_X[:used_data], data_y[:used_data], data_y_mod[:used_data]

    # Mini-images, just for test:
    CROP = -1
    X_small, y_small, y_mod_small = X[:,:,:CROP,:CROP], y[:,:CROP,:CROP], y_mod[:,:,:CROP,:CROP]


    # ------ # ------ # ------- # ------- #
    #        MAIN GREEDY ROUTINE:         #
    # ------ # ------ # ------- # ------- #
    from greedyNET.nets.greedyNet import greedyRoutine, restore_greedyModel

    eval_size = 0.2
    nolearn_kwargs = {
    }

    greedy_kwargs = {
        'update': adam,
        'update_learning_rate': 0.01,
        'update_beta1': 0.9,
    }

    num_VGG16_layers = 2

    # now = datetime.datetime.now()

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
    lrn_conv = 6.94683e-02
    decay_conv = 1.90961e-02
    lrn_regr = 1.61034e-02
    decay_regr = 2.27008e-01

    # 'lrn_rate': 0.0073923096, 'L2': 0.0051840702

    fullNet_params = {
        'num_filters1': 20,
        'filter_size1': 3,
        'filter_size2': 3,
        'eval_size': eval_size,
        'L2': 0.051840702,

        'max_epochs': 15,
        'update': adam,
        'update_learning_rate': theano.shared(float32(lrn_conv)),
        # 'weights_HeGain': np.sqrt(2),

        'numIter_subLog': 2,
        # 'livePlot': True,


        'update_beta1': theano.shared(float32(0.9)),
        'on_epoch_finished': [
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
    #     'max_epochs': 10,
    #     'update': adam,
    #     'numIter_subLog': 2,
    #     'L2': 0.0051840702,
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


    # # --------------------------
    # # Nets fitting routines:
    # # --------------------------
    # def fit_naiveRoutine_convSoft(net):
    #     net.fit(X_small, y_small, epochs=10)
    #     return net
    # def fit_naiveRoutine_convSoft_finetune(net):
    #     net.update_learning_rate.set_value(float32(lrn_conv))
    #     net.update_AdjustObjects([
    #         AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
    #         AdjustVariable('update_learning_rate', start=lrn_conv, mode='log', decay_rate=decay_conv)
    #     ])
    #     net.fit(X_small, y_small, epochs =10)
    #     return net
    # def fit_naiveRoutine_regr(net):
    #     net.fit(X_small, y_small, epochs =10)
    #     return net

    # ------ # ------ # ------- # ------- #
    #        TUNE HYPERPARAMETER:         #
    # ------ # ------ # ------- # ------- #

    # def adjust_convSoft(convSoft):
    #     convSoft.net.update_learning_rate.set_value(float32(lrn_conv))
    #     convSoft.net.update_AdjustObjects([
    #         AdjustVariable('update_beta1', start=0.9, mode='linear', stop=0.999),
    #         AdjustVariable('update_learning_rate', start=lrn_conv, mode='log', decay_rate=decay_conv)
    #     ])
    #     return convSoft


    from mod_nolearn.tuneHyper import tune_hyperparams

    class tune_lrn_rate(tune_hyperparams):
        def fit_model(self, param_values, model_name):
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

            # ----------------------
            # Init full net:
            # ----------------------
            logs_path = self.BASE_PATH_LOG_MODEL+'full_net/'
            utils.create_dir(logs_path)
            params = deepcopy(fullNet_params)
            params['log_frequency'] = 1
            params['log_filename'] = 'log.txt'
            params['subLog_filename'] = 'sub_log.txt'
            params['pickleModel_mode'] = 'on_epoch_finished'
            params['trackWeights_pdfName'] = 'weights.txt'
            params['pickle_filename'] = 'model.pickle'
            params['logs_path'] = logs_path
            full_net = fullNet(
                greedy_routine.net,
                greedy_routine.output_channels,
                num_nodes=6,
                **params
            )

            full_net.net.fit(X_small, y_small, epochs=20)



            return {}



    first = tune_lrn_rate(
        (
            ('lrn_rate', 0.0, None, None, np.float32),
            ('decay_rate', 0.0, None, 'log', np.float32)
        ),
        ['train_loss', 'valid_loss'],
        num_iterations = 1,
        name = "full_net_5layers",
        folder_out = 'tuning',
        plot=False
    )

    # FIRST ONE, OLE!
    first()

logging.basicConfig(level=logging.DEBUG, filename='complete_train/ole/error.log')

# try:
main()
# except:
#     logging.exception("Oops:")
