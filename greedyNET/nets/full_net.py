import time
import numpy as np
from copy import deepcopy
import json

from lasagne.init import HeNormal
import theano.tensor as T
from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.nonlinearities import rectify, identity
import lasagne.init

from lasagne.init import HeNormal

import greedyNET.greedy_utils as greedy_utils
import mod_nolearn.nets.segmNet as segmNet
from mod_nolearn.segmentFcts import pixel_accuracy_sigmoid


class fullNet(object):
    def __init__(self,previous_layers,input_filters,**kwargs):
        info = deepcopy(kwargs)
        # -----------------
        # General attributes:
        # -----------------
        self.filter_size1 = kwargs.pop('filter_size1', 7)
        self.filter_size2 = kwargs.pop('filter_size2', 7)
        self.num_filters1 = kwargs.pop('num_filters1', 5)
        self.input_filters = input_filters
        self.previous_layers = previous_layers
        self.num_classes = 1
        self.xy_input = kwargs.pop('xy_input', (None, None))
        self.eval_size = kwargs.pop('eval_size', 0.1)
        # Checks:
        if "num_classes" in kwargs:
            raise ValueError('Multy-class classification boosting not implemented for the moment. "num_classes" parameter is fixed to 2')
        if "train_split" in kwargs:
            raise ValueError('The option train_split is not used. Use eval_size instead.')


        # -------------------------------------
        # Specific parameters:
        # -------------------------------------
        # self.weights_HeGainin = kwargs.pop('weights_HeGain', 1.)
        kwargs.setdefault('name', 'convSoftmax')
        self.num_nodes = kwargs.pop('num_nodes', 5)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.active_nodes = 0

        customBatchIterator = greedy_utils.BatchIterator_Greedy(
            batch_size=self.batch_size,
            shuffle=self.batchShuffle,
            previous_layers=self.previous_layers
        )

        # -------------------------------------
        # CONSTRUCT NETWORK:
        # -------------------------------------
        netLayers = [
            # layer dealing with the input data
            (layers.InputLayer, {
                'name': 'inputLayer',
                'shape': (None, self.input_filters, self.xy_input[0], self.xy_input[1])}),
            (layers.Conv2DLayer, {
                'name': 'conv1',
                'num_filters': self.num_nodes*self.num_filters1*self.num_classes,
                'filter_size': self.filter_size1,
                'W': HeNormal(np.sqrt(2)),
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_nodes*self.num_filters1*self.num_classes,
                'filter_size': self.filter_size1,
                'W': HeNormal(np.sqrt(2)),
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv3',
                'num_filters': self.num_nodes*self.num_filters1*self.num_classes,
                'filter_size': self.filter_size1,
                'W': HeNormal(np.sqrt(2)),
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv3',
                'num_filters': self.num_nodes*self.num_filters1*self.num_classes,
                'filter_size': self.filter_size1,
                'W': HeNormal(np.sqrt(2)),
                'pad':'same',
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv_last',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'W': HeNormal(np.sqrt(2)),
                'pad':'same',
                'nonlinearity': segmNet.sigmoid_segm}),
        ]

        self.net = segmNet.segmNeuralNet(
            layers=netLayers,
            batch_iterator_train = customBatchIterator,
            batch_iterator_test = customBatchIterator,
            objective_loss_function = segmNet.binary_crossentropy_segm,
            scores_train = [('trn pixelAcc', pixel_accuracy_sigmoid)],
            scores_valid = [('val pixelAcc', pixel_accuracy_sigmoid)],
            y_tensor_type = T.ftensor3,
            eval_size=self.eval_size,
            regression = True,
            **kwargs
        )

        self.net.initialize()


        # -------------------------------------
        # SAVE INFO NET:
        # -------------------------------------
        info['num_classes'] = 1
        info.pop('update', None)
        info.pop('on_epoch_finished', None)
        info.pop('on_batch_finished', None)
        info.pop('on_training_finished', None)
        for key in [key for key in info if 'update_' in key]:
            info[key] = info[key].get_value().item()
        json.dump(info, file(info['logs_path']+'/info-net.txt', 'w'))

