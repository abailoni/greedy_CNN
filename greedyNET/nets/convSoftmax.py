# ------------------------------
# For the moment just a LogRegr
# ------------------------------
import time
import numpy as np
from copy import deepcopy
import json

import theano.tensor as T
from lasagne import layers
from lasagne.layers import set_all_param_values, get_all_param_values
from lasagne.nonlinearities import rectify, identity
import lasagne.init

from lasagne.init import HeNormal

import greedyNET.greedy_utils as greedy_utils
import mod_nolearn.nets.segmNet as segmNet
from mod_nolearn.segmentFcts import pixel_accuracy_sigmoid


class convSoftmax_routine(object):
    '''
    TO BE COMPLETELY UPDATED

    This class contains the network that put together the LogRegr networks computed in the boosting procedure.

    Inputs and options:
        - instance of class Boost_LogRegr (mandatory): the first trained node. Many parameters of Net2 are inherited by this network
        - batch_size (100)
        - batchShuffle (True)
        - filter_size (7): the dimension of the filter of the second convolutional level
        - num_nodes (5): number of nodes trained in a boosting matter
        - usual NeuralNet parameters (update, learning_rate...)

    Names of the layers:
        - conv1
        - mask
        - conv2
    '''
    def __init__(self,previous_layers,input_filters,**kwargs):
        info = deepcopy(kwargs)
        info['logs_path'] = kwargs.pop('logs_path', './logs/')
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
        self.name = kwargs.pop('name', 'boostRegr')
        self.batch_size = kwargs.pop('batch_size', 100)
        self.batchShuffle = kwargs.pop('batchShuffle', True)
        self.num_nodes = kwargs.pop('num_nodes', 5)

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
                'pad':'same',
                'nonlinearity': rectify}),
            (MaskLayer,{
                'name': 'mask',
                'num_filters1': self.num_filters1,
                'num_classes': self.num_classes
                }),
            (layers.Conv2DLayer, {
                'name': 'conv2',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'pad':'same',
                'nonlinearity': identity}),
            # New node:
            (layers.Conv2DLayer, {
                'incoming': 'inputLayer',
                'name': 'conv1_newNode',
                'num_filters': self.num_filters1*self.num_classes,
                'filter_size': self.filter_size1,
                'pad':'same',
                'W': HeNormal(np.sqrt(2)),
                'nonlinearity': rectify}),
            (layers.Conv2DLayer, {
                'name': 'conv2_newNode',
                'num_filters': self.num_classes,
                'filter_size': self.filter_size2,
                'pad':'same',
                'W': HeNormal(1.),
                'nonlinearity': identity}),
            (layers.ElemwiseMergeLayer, {
                'incomings': ['conv2', 'conv2_newNode'],
                'merge_function': T.add}),
            (layers.NonlinearityLayer,{
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

        # Set the first layer as not trainable:
        self.net._output_layer = self.net.initialize_layers()
        self.net.layers_['conv1'].params[self.net.layers_['conv1'].W].remove('trainable')
        self.net.layers_['conv1'].params[self.net.layers_['conv1'].b].remove('trainable')
        self.net.layers_['conv1_newNode'].params[self.net.layers_['conv1_newNode'].W].remove('trainable')
        self.net.layers_['conv1_newNode'].params[self.net.layers_['conv1_newNode'].b].remove('trainable')


        # print "\n\n---------------------------\nCompiling Network 2...\n---------------------------"
        # tick = time.time()
        self.net.initialize()
        # tock = time.time()
        # print "Done! (%f sec.)\n\n\n" %(tock-tick)


        # # Insert the weights of the first network:
        # self.insert_weights(regrNode1)

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
        json.dump(info, file(info['logs_path']+self.name+'/info-net.txt', 'w'))



    def insert_weights(self, regr):
        '''
        In order the following operations are done:
         - Update mask main part: activate another node
         - Copy the 'new_node' weights in the main part of the net
         - Copy the regr weights in the 'new_node' part
         - Recompile the net

        Structure of parameters:
         - W1: (num_classes*num_filters1*num_nodes, num_inputs, filter_length1)
         - b1: (num_classes*num_filters1*num_nodes, )
         - W2: (num_classes, num_classes*num_filters1*num_nodes, filter_length2)
         - b2: (num_classes,)
        '''
        # ------------------
        # Update mask:
        # ------------------
        self.net.layers_['mask'].add_node()
        actNode = self.net.layers_['mask'].active_nodes
        self.active_nodes = actNode

        # ------------------
        # Get weights:
        # ------------------
        W1, b1, maskParam, W2, b2 = layers.get_all_param_values(self.net.layers_['conv2'])
        newNode_W1, newNode_b1, newNode_W2, newNode_b2 = layers.get_all_param_values(self.net.layers_['conv2_newNode'])
        reg_W1, reg_b1, reg_W2, reg_b2 = layers.get_all_param_values(regr.net.layers_['conv2'])
        # --------------------
        # Update main part:
        # --------------------
        nNodes = self.num_classes*self.num_filters1
        start = nNodes*(actNode-1)
        stop = nNodes*actNode
        slice_weights = slice(start,stop)
        W1[slice_weights,:,:], b1[slice_weights] = newNode_W1, newNode_b1
        # For the moment I don't touch b2... Not sure about this...
        W2[:,slice_weights,:], b2 = newNode_W2, b2+newNode_b2
        layers.set_all_param_values(self.net.layers_['conv2'], [W1, b1, maskParam, W2, b2])
        # --------------------
        # Insert new node:
        # --------------------
        newNode_W1, newNode_b1, newNode_W2, newNode_b2 = reg_W1, reg_b1, reg_W2, reg_b2
        layers.set_all_param_values(self.net.layers_['conv2_newNode'], [newNode_W1, newNode_b1, newNode_W2, newNode_b2])

        # --------------------
        # Set layer conv2 not-trainable:
        # --------------------
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].W].remove('trainable')
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].b].remove('trainable')
        self.net.initialize()

    def activate_nodes(self):
        '''
        Makes the active weights of the main part of the net (conv2 layer) trainable. Then recompile the net.
        '''
        # Set layer conv2 trainable:
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].W].add('trainable')
        self.net.layers_['conv2'].params[self.net.layers_['conv2'].b].add('trainable')
        self.net.initialize()


    # def clone(self,**kwargs):
    #     '''
    #     Options:
    #         - reset (True): for resetting the weights of the new Net

    #     Return the cloned object.
    #     '''
    #     if self.xy_input is not (None, None):
    #         raise Warning("Cloning with xy-image-inputs already set...")
    #     kwargs.setdefault('reset', True)
    #     kwargs.setdefault('setClassifier', False)
    #     newObj = deepcopy(self)
    #     if kwargs['reset']:
    #         # Reset some stuff:
    #         if newObj.net.verbose:
    #             # newObj.net.on_epoch_finished.append(PrintLog())
    #             pass
    #         newObj.net.train_history_[:] = []
    #         newObj._reset_weights()
    #     return newObj

    # def _reset_weights(self):
    #     W_regr, b_regr, mask, W_conv2, b_conv2 = get_all_param_values(self.net.layers_['conv2'])
    #     glorot, constant = lasagne.init.GlorotNormal(), lasagne.init.Constant()
    #     new_weights = [glorot.sample(W_regr.shape), constant.sample(b_regr.shape), mask, glorot.sample(W_conv2.shape), constant.sample(b_conv2.shape)]
    #     set_all_param_values(self.net.layers_['convLayer'], new_weights)
    #     self.net.layers_['mask'].active_nodes = 0
    #     self.active_nodes = 0

class MaskLayer(layers.Layer):
    '''
    --------------------------
    Subclass of lasagne.layers.Layer:
    --------------------------

    The received input should be in the form: (N, num_classes*num_nodes, dim_x, dim_y)

    Inputs:
     - num_filters1 (5)
     - num_classes (1)

    The only parameter of the layer is a 2-dim array containing the slice extremes
    deciding the active nodes. When initialized, no nodes are active.
    '''

    def __init__(self, incoming, *args, **kwargs):
        self.num_filters1 = kwargs.pop('num_filters1', 5)
        self.num_classes = kwargs.pop('num_classes', 1)
        super(MaskLayer, self).__init__(incoming, *args, **kwargs)
        self.active_nodes = 0
        self.active_nodes_slice = self.add_param(np.ones(1, dtype=np.int8), (1,), name='active_nodes_slice', trainable=False, regularizable=False)
        #
        self.active_nodes_slice.set_value([0])


    def add_node(self):
        '''
        Add one node and make all the others not trainable.
        '''
        self.active_nodes += 1
        actNods, nClas, nRegNodes = self.active_nodes, self.num_classes, self.num_filters1
        self.active_nodes_slice.set_value([actNods*nClas*nRegNodes])


    def get_output_for(self, input, **kwargs):
        return T.set_subtensor(input[:,self.active_nodes_slice[0]:,:,:], 0.)




