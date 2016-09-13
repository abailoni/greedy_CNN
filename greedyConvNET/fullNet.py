from copy import deepcopy
import json

from lasagne.init import Normal
import theano.tensor as T
from lasagne import layers
from lasagne.nonlinearities import rectify


import pretr_nets.vgg16 as vgg16


import various.utils as utils
from mod_nolearn import BatchIterator
from mod_nolearn.segm import segmNeuralNet
import mod_nolearn.segm.segm_utils as segm_utils


class fullNet(object):
    def __init__(self,num_VGG16_layers,num_conv_layers,**kwargs):
        info = deepcopy(kwargs)
        # -----------------
        # General attributes:
        # -----------------
        self.filter_size = kwargs.pop('filter_size', 3)
        self.num_filters = kwargs.pop('num_filters', 65)
        self.num_classes = kwargs.pop('num_classes', 2)
        self.num_conv_layers = kwargs.pop('num_conv_layers', 5)
        self.xy_input = kwargs.pop('xy_input', (None, None))
        self.eval_size = kwargs.pop('eval_size', 0.2)

        self.model_name = kwargs.pop('model_name', 'greedyNET')
        self.BASE_PATH_LOG = kwargs.pop('BASE_PATH_LOG', "./logs/")
        self.BASE_PATH_LOG_MODEL = self.BASE_PATH_LOG+self.model_name+'/'



        # -------------------------------------
        # Specific parameters:
        # -------------------------------------
        # self.weights_HeGainin = kwargs.pop('weights_HeGain', 1.)
        kwargs.setdefault('name', 'fullNet')
        self.init_weight = kwargs.pop('init_weight', 1e-3)
        self.batch_size = kwargs.pop('batch_size', 20)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.active_nodes = 0

        customBatchIterator = BatchIterator(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
        )

        # -------------------------------------
        # CONSTRUCT NETWORK:
        # -------------------------------------
        self.num_VGG16_layers = int(num_VGG16_layers)
        self.layers = vgg16.nolearn_vgg16_layers(data_size=(None, 3, self.xy_input[0], self.xy_input[1]))[:self.num_VGG16_layers+1]
        self.layers += [
            (layers.Conv2DLayer, {
                'name': 'conv%d' %i,
                'num_filters': self.num_filters,
                'filter_size': self.filter_size,
                'W': Normal(std=self.init_weight),
                'pad':'same',
                'nonlinearity': rectify}) for i in range(self.num_conv_layers)] + [
            (layers.Conv2DLayer, {
                'name': 'conv_last',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size,
                'W': Normal(std=self.init_weight),
                'pad':'same',
                'nonlinearity': segm_utils.softmax_segm}),
        ]

        self.net = segmNeuralNet(
            layers=self.layers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = segm_utils.categorical_crossentropy_segm,
            scores_train = [('trn pixelAcc', segm_utils.pixel_accuracy)],
            # scores_valid = [('val pixelAcc', pixel_accuracy_sigmoid)],
            y_tensor_type = T.ltensor3,
            eval_size=self.eval_size,
            regression = False,
            logs_path= self.BASE_PATH_LOG_MODEL,
            **kwargs
        )


        self.net.initialize()

        # --------------------
        # Copy vgg16 weights:
        # --------------------
        self.net = vgg16.nolearn_insert_weights_vgg16(self.net,self.num_VGG16_layers)


        # -------------------------------------
        # SAVE INFO NET:
        # -------------------------------------
        info['num_classes'] = self.num_classes
        info.pop('update', None)
        info.pop('on_epoch_finished', None)
        info.pop('on_batch_finished', None)
        info.pop('on_training_finished', None)
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        utils.create_dir(self.BASE_PATH_LOG_MODEL)
        json.dump(info, file(self.BASE_PATH_LOG_MODEL+'info-net.txt', 'w'))

